import unittest
import random
import os
import tempfile

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
from run.create_dataset_parallel import save_dataset
from run.create_dataset_parallel import compute_n_step_returns
from run.create_dataset_parallel import build_ac_dataset
from run.train_model import train_ppo

class TestComputeNStepReturns(unittest.TestCase):
    def test_intermediate_rewards_only(self):
        # Simple case: intermediate reward followed by zeros, n=2
        rewards = [0.1, 0.0, 0.0]
        values = [0.0, 0.0, 0.0]
        nstep, adv = compute_n_step_returns(rewards, n_step=2, gamma=0.9, values=values)
        self.assertEqual(len(nstep), 3)
        self.assertEqual(len(adv), 3)
        # Returns: [0.1, 0.0, 0.0]
        self.assertAlmostEqual(nstep[0], 0.1, places=7)
        self.assertAlmostEqual(nstep[1], 0.0, places=7)
        self.assertAlmostEqual(nstep[2], 0.0, places=7)
        # Advantages equal returns since values are zeros
        self.assertAlmostEqual(adv[0], 0.1, places=7)
        self.assertAlmostEqual(adv[1], 0.0, places=7)
        self.assertAlmostEqual(adv[2], 0.0, places=7)

    def test_terminal_reward_discounting(self):
        # Terminal reward at the end; ensure proper discounting and truncation
        rewards = [0.0, 0.0, 1.0]
        values = [0.0, 0.0, 0.0]
        nstep, adv = compute_n_step_returns(rewards, n_step=3, gamma=0.99, values=values)
        # t=0: 0 + 0.99*0 + 0.99^2*1 = 0.9801
        # t=1: 0 + 0.99*1 = 0.99
        # t=2: 1
        self.assertAlmostEqual(nstep[0], 0.99**2, places=7)
        self.assertAlmostEqual(nstep[1], 0.99, places=7)
        self.assertAlmostEqual(nstep[2], 1.0, places=7)
        # Advantages equal returns since values are zeros
        self.assertAlmostEqual(adv[0], nstep[0], places=7)
        self.assertAlmostEqual(adv[1], nstep[1], places=7)
        self.assertAlmostEqual(adv[2], nstep[2], places=7)

    def test_truncation_when_trajectory_shorter_than_n(self):
        rewards = [1.0, 1.0]
        values = [0.0, 0.0]
        # n=3 should only sum available rewards
        nstep, _ = compute_n_step_returns(rewards, n_step=3, gamma=0.5, values=values)
        # t=0: 1 + 0.5*1 = 1.5; t=1: 1
        self.assertAlmostEqual(nstep[0], 1.5, places=7)
        self.assertAlmostEqual(nstep[1], 1.0, places=7)

    def test_advantages_subtract_values(self):
        rewards = [1.0, 0.0, 0.0]
        values = [0.2, -1.0, 0.5]
        # n=1 so returns equal immediate rewards
        nstep, adv = compute_n_step_returns(rewards, n_step=1, gamma=0.99, values=values)
        self.assertAlmostEqual(nstep[0], 1.0, places=7)
        self.assertAlmostEqual(nstep[1], 0.0, places=7)
        self.assertAlmostEqual(nstep[2], 0.0, places=7)
        self.assertAlmostEqual(adv[0], 1.0 - 0.2, places=7)
        self.assertAlmostEqual(adv[1], 0.0 - (-1.0), places=7)  # 1.0
        self.assertAlmostEqual(adv[2], 0.0 - 0.5, places=7)     # -0.5


class TestEndToEndTraining(unittest.TestCase):
    def test_build_dataset_and_train_minimal(self):
        """Create a tiny dataset (.npz) and run a minimal PPO training to verify the pipeline executes."""
        # Build 4 heuristic AC players for 1 game
        players = [RecordingHeuristicACPlayer(random_exploration=0.0) for _ in range(4)]
        built = build_ac_dataset(
            games=2,
            seed=123,
            n_step=1,
            gamma=0.99,
            prebuilt_players=players,
        )
        # Basic sanity on built dataset
        self.assertIn('action_idx', built)
        self.assertGreater(len(built['action_idx']), 0)

        # Save to a temporary .npz
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'tiny_dataset.npz')
            save_dataset(built, out_path)
            self.assertTrue(os.path.isfile(out_path))

            # Train a model quickly on CPU; keep everything minimal to just verify it runs
            model_pt_path = train_ppo(
                dataset_path=out_path,
                epochs=1,
                batch_size=32,
                lr=3e-4,
                epsilon=0.2,
                value_coeff=0.5,
                entropy_coeff=0.01,
                bc_fallback_ratio=5.0,
                device='cpu',
                min_delta=1e-3,
                val_split=0.0,  # avoid splitting a single game
                init_model=None,
                warm_up_acc=0.0,
                warm_up_max_epochs=1,
                warm_up_value=False,
                hidden_size=64,
                embedding_dim=8,
                kl_threshold=None,
                patience=0,
                dl_workers=0,  # no worker processes in unit tests
                prefetch_factor=2,
            )
            # Ensure a model file was produced
            self.assertTrue(isinstance(model_pt_path, str))
            self.assertTrue(os.path.isfile(model_pt_path))

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


