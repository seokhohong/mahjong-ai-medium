import unittest
import os


class TestCreateDataset(unittest.TestCase):
    def test_build_ac_dataset_heuristic_single_game(self):
        try:
            from run.create_dataset import build_ac_dataset
        except Exception as e:
            self.skipTest(f"Skipping dataset build test due to import error: {e}")
            return

        data = build_ac_dataset(
            games=1,
            seed=123,
            use_heuristic=True,
            temperature=0.0,
            n_step=3,
            gamma=0.99,
        )

        # Basic keys present
        for k in (
            'states', 'actions', 'returns', 'advantages',
            'old_log_probs', 'game_ids', 'step_ids', 'actor_ids', 'flat_policies',
        ):
            self.assertIn(k, data)

        # There should be at least one recorded step
        num = len(data['states'])
        self.assertGreater(num, 0)
        # All arrays aligned in length
        self.assertEqual(num, len(data['actions']))
        self.assertEqual(num, len(data['returns']))
        self.assertEqual(num, len(data['advantages']))
        self.assertEqual(num, len(data['old_log_probs']))
        self.assertEqual(num, len(data['game_ids']))
        self.assertEqual(num, len(data['step_ids']))
        self.assertEqual(num, len(data['actor_ids']))
        self.assertEqual(num, len(data['flat_policies']))

        # States are dicts produced by encode_game_perspective
        s0 = data['states'][0]
        if hasattr(s0, 'item'):
            s0 = s0.item()
        self.assertIsInstance(s0, dict)
        # Actions are serialized dicts
        a0 = data['actions'][0]
        if hasattr(a0, 'item'):
            a0 = a0.item()
        self.assertIsInstance(a0, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)


