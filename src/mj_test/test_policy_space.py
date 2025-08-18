import unittest
import numpy as np

from core.learn.ac_constants import FLAT_POLICY_SIZE
from core.game import MediumJong, Player
from core.learn.policy_utils import flat_index_for_action


class TestPolicyActionSpace(unittest.TestCase):
    def test_flat_policy_size_matches_mask_and_utils(self):
        # Create a fresh game state
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        pid = g.current_player_idx
        gp = g.get_game_perspective(pid)

        # Mask shape should match declared flat policy size
        mask = np.asarray(gp.legal_flat_mask(), dtype=np.int32)
        self.assertEqual(mask.shape[0], FLAT_POLICY_SIZE)

        # Every legal index should reconstruct to a legal move and round-trip to same index
        legal_indices = [i for i, v in enumerate(mask.tolist()) if v == 1]
        for idx in legal_indices:
            from core.learn.policy_utils import build_move_from_flat
            move = build_move_from_flat(gp, idx)
            # Some indices can be ambiguous in reconstruction (should return None). Skip those.
            if move is None:
                continue
            self.assertTrue(g.is_legal(pid, move))
            idx2 = flat_index_for_action(gp, move)
            self.assertEqual(idx2, idx)


if __name__ == '__main__':
    unittest.main(verbosity=2)


