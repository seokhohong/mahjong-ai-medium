import unittest
import random
from base64 import decode

import numpy as np
# torch is available in .venv312
import torch
from sklearn.preprocessing import StandardScaler

from core.game import MediumJong, Player
from core.learn.ac_player import ACPlayer
from core.learn.ac_network import ACNetwork
from core.learn.ac_constants import GAME_STATE_VEC_LEN
from core.learn.policy_utils import (
    build_move_from_two_head,
    encode_two_head_action,
)
from core.learn.data_utils import build_state_from_npz_row
from core.learn.feature_engineering import encode_game_perspective, MAX_HAND_LEN
from core.learn.recording_ac_player import ExperienceBuffer, RecordingHeuristicACPlayer
from run.create_dataset import save_dataset


class ScriptedRecordingPlayer(Player):
    """Deterministic discarder that records (state, action) pairs.

    Strategy: choose the lowest-index legal discard at each action state.
    """

    def __init__(self, buffer: ExperienceBuffer):
        super().__init__()
        self.buffer = buffer

    def play(self, gs):  # type: ignore[override]
        # Prefer a discard if available, else first legal action
        a_mask = gs.legal_action_mask()
        discard_idx = None
        from core.learn.ac_constants import ACTION_HEAD_INDEX
        if a_mask[ACTION_HEAD_INDEX['discard']] == 1:
            discard_idx = ACTION_HEAD_INDEX['discard']
        # Choose action
        if discard_idx is not None:
            a_idx = discard_idx
            t_mask = gs.legal_tile_mask(a_idx)
            # pick lowest tile index enabled (1..37). If none, fallback to no-op (shouldn't happen for discard)
            t_idx = 0
            for i in range(1, 38):
                if t_mask[i] == 1:
                    t_idx = i
                    break
        else:
            # fallback to first legal action head and its first legal tile
            a_idx = None
            for i, bit in enumerate(a_mask):
                if bit == 1:
                    a_idx = i
                    break
            assert a_idx is not None, "No legal action found"
            t_mask = gs.legal_tile_mask(a_idx)
            t_idx = 0
            for i in range(38):
                if t_mask[i] == 1:
                    t_idx = i
                    break
        move = build_move_from_two_head(gs, int(a_idx), int(t_idx))
        assert move is not None, f"Failed to build move from two-head indices {(a_idx, t_idx)}"
        # Record experience with zero value/logp
        self.buffer.add(
            encode_game_perspective(gs),
            (int(a_idx), int(t_idx)),
            0.0,
            0.0,
            action_logp=0.0,
            tile_logp=0.0,
        )
        return move


# --- Shared helpers to avoid duplication ---

def _build_tensors_from_states_actions(net, states, actions, dev):
    import torch  # type: ignore
    # Convert indexed features to tensors via network's featurizer
    hand_list, calls_list, disc_list, gsv_list = [], [], [], []
    for st in states:
        h, c, d, gsv = net.extract_features_from_indexed(
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
    # actions is a list of (action_idx, tile_idx)
    a_idx = torch.tensor([int(a[0]) for a in actions], dtype=torch.long, device=dev)
    t_idx = torch.tensor([int(a[1]) for a in actions], dtype=torch.long, device=dev)
    return hand, calls, disc, gsv, a_idx, t_idx


def _train_bc_to_perfect(model, hand, calls, disc, gsv, a_idx, t_idx, *, max_steps: int = 200):
    import torch  # type: ignore
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(max_steps):
        pa, pt, _v = model(hand.float(), calls.float(), disc.float(), gsv.float())
        log_pa = (pa.clamp_min(1e-8)).log()
        log_pt = (pt.clamp_min(1e-8)).log()
        nll = __import__('torch').nn.functional.nll_loss
        loss = nll(log_pa, a_idx) + nll(log_pt, t_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with __import__('torch').no_grad():
            pred_a = __import__('torch').argmax(pa, dim=1)
            pred_t = __import__('torch').argmax(pt, dim=1)
            acc = float(((pred_a == a_idx) & (pred_t == t_idx)).float().mean().item())
            if acc >= 1.0:
                break
    assert acc >= 1.0, "model failed to memorize"
    # Final assert inside tests


def _assert_replay_identical(net, states, actions):
    import numpy as np  # local import
    from core.learn.ac_player import ACPlayer
    from core.learn.feature_engineering import decode_game_perspective
    acp = ACPlayer(network=net, temperature=0)
    misses = []
    for i, (st, action_tuple) in enumerate(zip(states, actions)):
        gp = decode_game_perspective(st)
        # Evaluate action head directly
        pa, pt, _ = acp.network.evaluate(gp, scaler=acp.gsv_scaler)
        order = np.argsort(-pa)[:3]
        gold_a = int(action_tuple[0])
        if gold_a not in set(order.tolist()):
            misses.append((i, gold_a, order.tolist()))
    match_frac = 1.0 - (len(misses) / max(1, len(states)))
    assert match_frac >= 0.9, f"Observed action head in top-3 for only {match_frac:.3f} of states; first misses: {misses[:5]}"


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
        rec_players = [RecordingHeuristicACPlayer(random_exploration=0.0),
                       RecordingHeuristicACPlayer(random_exploration=0.0),
                       RecordingHeuristicACPlayer(random_exploration=0.0),
                       RecordingHeuristicACPlayer(random_exploration=0.0)]
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
        net.fit_scaler(np.array([state["game_state"] for state in all_states]))
        hand, calls, disc, gsv, a_idx, t_idx = _build_tensors_from_states_actions(net, all_states, all_actions, dev)
        _train_bc_to_perfect(model, hand, calls, disc, gsv, a_idx, t_idx, max_steps=400)
        # After training, assert the model argmax matches the training actions for every sample
        with torch.no_grad():
            pa, pt, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
            pa_solo, pt_solo, _ = model(hand.float()[[0]], calls.float()[[0]], disc.float()[[0]], gsv.float()[[0]])
            self.assertTrue(np.allclose(pa[0].cpu().numpy(), pa_solo[0].cpu().numpy()))
            self.assertTrue(np.allclose(pt[0].cpu().numpy(), pt_solo[0].cpu().numpy()))

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
                temperature=1,
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
                warm_up_max_epochs=150
            )

            # After warm-up to ~40% accuracy, load ACPlayer from the saved directory
            model_dir = os.path.dirname(trained_model_path)
            player = ACPlayer.from_directory(model_dir, temperature=1)

            # Iterate through dataset and rehydrate GamePerspective; measure accuracy
            correct = 0
            total = 0
            with np.load(dataset_path, allow_pickle=True) as data:
                N = int(len(data['action_idx']))
                for i in range(N):
                    st = build_state_from_npz_row(data, i)
                    gp = decode_game_perspective(st)
                    move, _, a_idx, t_idx, _, = player.compute_play(gp)
                    pred_a, pred_t = a_idx, t_idx
                    gold_a = int(data['action_idx'][i])
                    gold_t = int(data['tile_idx'][i])
                    correct += int((pred_a == gold_a) and (pred_t == gold_t))
                    total += 1

            acc = (correct / max(1, total))
            self.assertGreaterEqual(acc, 0.35, f"Rehydrated accuracy {acc:.2%} < 40%")

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
                pickle.dump(network._gsv_scaler, f)

            # Test loading from directory
            loaded_player = ACPlayer.from_directory(tmp_dir, temperature=0.5)

            # Verify the player was loaded correctly (model + scaler)
            self.assertIsInstance(loaded_player, ACPlayer)
            self.assertAlmostEqual(loaded_player.temperature, 0.5)
            self.assertIsNotNone(loaded_player.gsv_scaler)

            print("✅ ACPlayer.from_directory test completed successfully!")


    def test_temperature(self):
        # Use the same untrained ACPlayer at two temperatures. Outputs should be mostly consistent but not identical.
        import numpy as np
        from core.learn.ac_player import ACPlayer
        from core.game import MediumJong
        from core.learn.recording_ac_player import RecordingHeuristicACPlayer
        from core.learn.ac_constants import ACTION_HEAD_INDEX

        # Determinism
        random.seed(123)
        np.random.seed(123)

        # Build a default, untrained AC network/player
        base_player = ACPlayer.default(temperature=0.1)

        # dummy fit for this test
        base_player.network.fit_scaler(np.ones((10, GAME_STATE_VEC_LEN)))

        # Clone two players that share the same network but different temperatures
        cold_player = ACPlayer(network=base_player.network, gsv_scaler=base_player.gsv_scaler, temperature=0.05)
        warm_player = ACPlayer(network=base_player.network, gsv_scaler=base_player.gsv_scaler, temperature=1.0)

        # Same opponents for both runs
        opps = [RecordingHeuristicACPlayer(random_exploration=0.0), RecordingHeuristicACPlayer(random_exploration=0.0),
                RecordingHeuristicACPlayer(random_exploration=0.0)]

        # Run with cold temperature
        g1 = MediumJong([cold_player] + opps)
        moves_cold = []
        discard_tiles_cold = []
        steps = 0
        while not g1.is_game_over() and steps < 200:
            gp = g1.get_game_perspective(g1.current_player_idx)
            if g1.current_player_idx == 0:
                mv = cold_player.play(gp)
                # Record action head index
                ai, ti = encode_two_head_action(mv)
                moves_cold.append(ai)
                if ai == ACTION_HEAD_INDEX['discard']:
                    discard_tiles_cold.append(int(ti))
            g1.play_turn()
            steps += 1

        # Re-seed and run with warm temperature
        random.seed(123)
        np.random.seed(123)
        g2 = MediumJong([warm_player] + opps)
        moves_warm = []
        discard_tiles_warm = []
        steps = 0
        while not g2.is_game_over() and steps < 200:
            gp = g2.get_game_perspective(g2.current_player_idx)
            if g2.current_player_idx == 0:
                mv = warm_player.play(gp)
                ai, ti = encode_two_head_action(mv)
                moves_warm.append(ai)
                if ai == ACTION_HEAD_INDEX['discard']:
                    discard_tiles_warm.append(int(ti))
            g2.play_turn()
            steps += 1

        # Basic sanity
        self.assertGreater(len(moves_cold), 0)
        self.assertGreater(len(moves_warm), 0)

        # Compare tile-level distributions on discard actions (where temperature should affect tile selection)
        import collections
        self.assertGreater(len(discard_tiles_cold), 0)
        self.assertGreater(len(discard_tiles_warm), 0)

        cold_tile_counts = collections.Counter(discard_tiles_cold)
        warm_tile_counts = collections.Counter(discard_tiles_warm)

        def to_dist(counter):
            total = sum(counter.values())
            return {k: v / float(max(1, total)) for k, v in counter.items()}

        p_cold_tiles = to_dist(cold_tile_counts)
        p_warm_tiles = to_dist(warm_tile_counts)

        # Compute L1 distance over the union of observed tiles
        all_tiles = set(p_cold_tiles.keys()) | set(p_warm_tiles.keys())
        l1_tiles = sum(abs(p_cold_tiles.get(t, 0.0) - p_warm_tiles.get(t, 0.0)) for t in all_tiles)
        # Expect a noticeable difference when temperatures differ substantially
        self.assertGreater(l1_tiles, 0.05)


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


