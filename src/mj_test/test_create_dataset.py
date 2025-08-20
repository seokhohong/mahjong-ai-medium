import unittest
import os

from core.learn.ac_constants import MAX_TURNS


class TestCreateDataset(unittest.TestCase):
    def test_build_ac_dataset_heuristic_single_game(self):
        from run.create_dataset import build_ac_dataset

        data = build_ac_dataset(
            games=1,
            seed=123,
            use_heuristic=True,
            temperature=0.0,
            n_step=3,
            gamma=0.99,
        )
        self.assertTrue("called_discards" in data)
        # just test that this runs

    def test_records_legal(self):
        # Play a game with recording heuristic players and confirm recorded (state, action) are legal
        import random
        import numpy as np
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from core.game import MediumJong
        from core.learn.recording_ac_player import RecordingHeuristicACPlayer
        from core.learn.feature_engineering import decode_game_perspective
        from core.learn.policy_utils import build_move_from_flat

        random.seed(123)
        np.random.seed(123)

        players = [
            RecordingHeuristicACPlayer(0, random_exploration=0.0),
            RecordingHeuristicACPlayer(1, random_exploration=0.0),
            RecordingHeuristicACPlayer(2, random_exploration=0.0),
            RecordingHeuristicACPlayer(3, random_exploration=0.0),
        ]
        g = MediumJong(players)

        steps = 0
        while not g.is_game_over() and steps < MAX_TURNS:
            g.play_turn()
            steps += 1

        # Aggregate experiences from all players
        total = 0
        for p in players:
            self.assertEqual(len(p.experience.states), len(p.experience.actions))
            for st, act in zip(p.experience.states, p.experience.actions):
                gp = decode_game_perspective(st)
                move = build_move_from_flat(gp, int(act))
                self.assertIsNotNone(move)
                self.assertTrue(gp.is_legal(move))
                total += 1
        self.assertGreater(total, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)


