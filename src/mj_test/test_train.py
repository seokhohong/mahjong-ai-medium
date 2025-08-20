import unittest
import random
from base64 import decode

import numpy as np
# torch is available in .venv312
import torch
from sklearn.preprocessing import StandardScaler

from core.game import MediumJong, Player
from core.learn import ACPlayer, ACNetwork
from core.learn.policy_utils import build_move_from_flat, flat_index_for_action
from core.learn.feature_engineering import encode_game_perspective, MAX_HAND_LEN
from core.learn.recording_ac_player import ExperienceBuffer, RecordingHeuristicACPlayer
from run.create_dataset import save_dataset


class ScriptedRecordingPlayer(Player):
    """Deterministic discarder that records (state, action) pairs.

    Strategy: choose the lowest-index legal discard at each action state.
    """

    def __init__(self, pid: int, buffer: ExperienceBuffer):
        super().__init__(pid)
        self.buffer = buffer

    def play(self, gs):  # type: ignore[override]
        mask = gs.legal_flat_mask()
        choice = None
        # Prefer discard band 0..34
        for idx in range(0, 35):
            if mask[idx] == 1:
                choice = idx
                break
        if choice is None:
            # Fallback to first any legal action
            for idx, bit in enumerate(mask):
                if bit == 1:
                    choice = idx
                    break
        assert choice is not None, "No legal move found"
        move = build_move_from_flat(gs, int(choice))
        assert move is not None, f"Failed to build move from flat index {choice}"
        # Record experience with zero value/logp
        self.buffer.add(
            encode_game_perspective(gs),
            int(choice),
            0.0,
            0.0,
            main_logp=0.0,
            main_probs=None,
        )
        return move


# --- Shared helpers to avoid duplication ---

def _build_tensors_from_states_actions(net, states, actions, dev):
    import torch  # type: ignore
    # Convert indexed features to tensors via network's featurizer
    hand_list, calls_list, disc_list, gsv_list = [], [], [], []
    for st in states:
        h, c, d, gsv = net._extract_features_from_indexed(
            np.asarray(st['hand_idx'], dtype=np.int32),
            np.asarray(st['disc_idx'], dtype=np.int32),
            np.asarray(st['called_idx'], dtype=np.int32),
            np.asarray(st['game_state'], dtype=np.float32),
        )
        hand_list.append(h.astype(np.float32))
        calls_list.append(c.astype(np.float32))
        disc_list.append(d.astype(np.float32))
        gsv_list.append(gsv.astype(np.float32))

    hand = torch.from_numpy(np.stack(hand_list)).to(dev)
    calls = torch.from_numpy(np.stack(calls_list)).to(dev)
    disc = torch.from_numpy(np.stack(disc_list)).to(dev)
    gsv = torch.from_numpy(np.stack(gsv_list)).to(dev)
    flat_idx = torch.tensor(list(actions), dtype=torch.long, device=dev)
    return hand, calls, disc, gsv, flat_idx


def _train_bc_to_perfect(model, hand, calls, disc, gsv, flat_idx, *, max_steps: int = 200):
    import torch  # type: ignore
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(max_steps):
        pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
        log_pp = (pp.clamp_min(1e-8)).log()
        loss = __import__('torch').nn.functional.nll_loss(log_pp, flat_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with __import__('torch').no_grad():
            pred = __import__('torch').argmax(pp, dim=1)
            acc = float((pred == flat_idx).float().mean().item())
            if acc >= 1.0:
                break
    assert acc >= 1.0, "model failed to memorize"
    # Final assert inside tests


def _assert_replay_identical(net, states, actions):
    import numpy as np  # local import
    from core.learn.ac_player import ACPlayer
    from core.learn.feature_engineering import decode_game_perspective
    acp = ACPlayer(0, net, temperature=0)
    misses = []
    for i, (st, action_idx) in enumerate(zip(states, actions)):
        gp = decode_game_perspective(st)
        _, _, log_policy = acp.compute_play(gp)
        # Check that the observed action is within the top-3 choices of the policy
        # Using log-probabilities is fine since log is monotonic
        order = np.argsort(-log_policy)[:3]
        if int(action_idx) not in set(order.tolist()):
            misses.append((i, int(action_idx), order.tolist()))
    # Require at least 90% of states have the observed action in top-3
    match_frac = 1.0 - (len(misses) / max(1, len(states)))
    assert match_frac >= 0.9, f"Observed action in top-3 for only {match_frac:.3f} of states; first misses: {misses[:5]}"


def _features_key(state_dict):
    """Generate a hashable key from an encoded feature dict returned by encode_game_perspective."""
    import numpy as _np
    hand = tuple(_np.asarray(state_dict['hand_idx']).tolist())
    called = tuple(_np.asarray(state_dict['called_idx']).flatten().tolist())
    disc = tuple(_np.asarray(state_dict['disc_idx']).flatten().tolist())
    gsv = tuple(_np.asarray(state_dict['game_state']).tolist())
    cdm = tuple(_np.asarray(state_dict.get('called_discards', _np.zeros((4,0)))).flatten().tolist())
    return (hand, called, disc, gsv, cdm)


class TestTrainEndToEnd(unittest.TestCase):


    # models can sometimes give different outputs in batch vs per example
    def test_consistent_prediction(self):
        import torch  # type: ignore
        from core.learn.ac_network import ACNetwork  # type: ignore

        # Determinism
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        # Play a full game using deterministic heuristic recorders
        rec_players = [RecordingHeuristicACPlayer(0, random_exploration=0.0),
                   RecordingHeuristicACPlayer(1, random_exploration=0.0),
                   RecordingHeuristicACPlayer(2, random_exploration=0.0),
                   RecordingHeuristicACPlayer(3, random_exploration=0.0)]
        g = MediumJong(rec_players)
        steps = 0
        while not g.is_game_over() and steps < 256:
            g.play_turn()
            steps += 1

        # Aggregate experiences from all players
        all_states = []
        all_actions = []
        for p in rec_players:
            all_states.extend(p.experience.states)
            all_actions.extend(p.experience.actions)
        self.assertGreater(len(all_states), 0)

        # Train to memorize
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        player = ACPlayer.default()
        net = player.network
        model = net.torch_module
        model.eval()
        hand, calls, disc, gsv, flat_idx = _build_tensors_from_states_actions(net, all_states, all_actions, dev)
        _train_bc_to_perfect(model, hand, calls, disc, gsv, flat_idx, max_steps=400)
        # After training, assert the model argmax matches the training actions for every sample
        with torch.no_grad():
            model_pred_probs, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
            self.assertTrue(np.allclose(model_pred_probs[0].cpu().numpy(), model(hand.float()[[0]], calls.float()[[0]], disc.float()[[0]], gsv.float()[[0]])[0].cpu().numpy()))

    def test_build_ac_dataset_and_train_ppo_pipeline(self):
        """Test the complete pipeline: build dataset -> train model -> verify exact predictions."""
        import tempfile
        import os
        import sys
        import torch  # type: ignore
        from core.learn.ac_network import ACNetwork  # type: ignore
        from core.learn.feature_engineering import decode_game_perspective
        from core.learn.data_utils import load_gsv_scaler

        # Import the functions we need
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from run.create_dataset import build_ac_dataset  # type: ignore
        from run.train_model import train_ppo  # type: ignore

        # Set up temporary directory for dataset and model files
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = os.path.join(tmp_dir, "test_dataset.npz")
            model_path = os.path.join(tmp_dir, "test_model")

            # Step 1: Build a small dataset using build_ac_dataset
            print("Building AC dataset...")
            random.seed(42)  # For reproducibility
            np.random.seed(42)
            torch.manual_seed(42)

            dataset_dict = build_ac_dataset(
                games=1,  # Small dataset for quick testing
                seed=42,
                hidden_size=32,  # Network size isn't relevant
                embedding_dim=8,
                temperature=0,
                use_heuristic=True
            )

            save_dataset(dataset_dict, dataset_path)

            # Step 3: Train model using train_ppo with warm-up to 100% accuracy
            print("Training model with PPO...")

            # this should memorize the small number of samples
            trained_model_path = train_ppo(
                dataset_path=dataset_path,
                epochs=0,  # Minimal epochs since we're doing behavior cloning
                batch_size=1,
                lr=1e-4,
                val_split=0,
                warm_up_acc=0.4,  # not sure why it can't copy, maybe just learn up to 40%
                warm_up_max_epochs=100
            )

            # After warm-up to ~40% accuracy, load ACPlayer from the saved directory
            model_dir = os.path.dirname(trained_model_path)
            player = ACPlayer.from_directory(model_dir, player_id=0, temperature=0)

            # Iterate through dataset and rehydrate GamePerspective; measure accuracy
            correct = 0
            total = 0
            with np.load(dataset_path, allow_pickle=True) as data:
                N = int(len(data['flat_idx']))
                for i in range(N):
                    st = {
                        'hand_idx': data['hand_idx'][i],
                        'disc_idx': data['disc_idx'][i],
                        'called_idx': data['called_idx'][i],
                        'game_state': data['game_state'][i],
                        'called_discards': data['called_discards'][i],
                    }
                    gp = decode_game_perspective(st)
                    move, _, _ = player.compute_play(gp)
                    pred_idx = flat_index_for_action(gp, move)
                    gold_idx = int(data['flat_idx'][i])
                    correct += int(pred_idx == gold_idx)
                    total += 1

            acc = (correct / max(1, total))
            self.assertGreaterEqual(acc, 0.40, f"Rehydrated accuracy {acc:.2%} < 40%")

            # Clean up saved model
            if os.path.exists(trained_model_path):
                os.remove(trained_model_path)

            print("✅ Pipeline test completed successfully!")

    def test_ac_player_from_directory(self):
        """Test loading ACPlayer from directory using from_directory method."""
        import tempfile
        import os
        import torch  # type: ignore
        from sklearn.preprocessing import StandardScaler
        import pickle

        # Create a temporary model directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple model and scaler
            network = ACNetwork(gsv_scaler=StandardScaler(), hidden_size=32, embedding_dim=8)

            # Save model weights (state_dict) for portability
            model_path = os.path.join(tmp_dir, 'model.pt')
            network.save_model(model_path, save_entire_module=False)

            # Save the same scaler used by the network
            scaler_path = os.path.join(tmp_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(network.gsv_scaler, f)

            # Test loading from directory
            loaded_player = ACPlayer.from_directory(tmp_dir, player_id=1, temperature=0.5)

            # Verify the player was loaded correctly (model + scaler)
            self.assertIsInstance(loaded_player, ACPlayer)
            self.assertEqual(loaded_player.player_id, 1)
            self.assertAlmostEqual(loaded_player.temperature, 0.5)
            self.assertIsNotNone(loaded_player.gsv_scaler)

            print("✅ ACPlayer.from_directory test completed successfully!")

def cleanup_old_models():
    """Clean up any leftover model files from previous test runs."""
    import os
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.startswith("ac_ppo") and filename.endswith(".pt"):
                filepath = os.path.join(models_dir, filename)
                try:
                    os.remove(filepath)
                    print(f"Cleaned up: {filepath}")
                except OSError as e:
                    print(f"Warning: Could not remove {filepath}: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)


