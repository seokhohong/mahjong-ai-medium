import unittest

from core.constants import MAX_CALLS, NUM_PLAYERS
from core.game import MediumJong, Player
from core.learn.policy_utils import build_move_from_flat, flat_index_for_action
from core.game import (
    MediumJong, Player, Tile, TileType, Suit, Honor,
    Discard, Tsumo, Pon, Chi, Riichi,
    KanDaimin, KanAnkan, CalledSet,
)
from core.learn.feature_engineering import encode_game_perspective, decode_game_perspective, MAX_HAND_LEN
from mj_test.test_utils import ForceDiscardPlayer
from core.learn.ac_constants import MAX_TURNS

class TestFeatureEngineering(unittest.TestCase):
    def test_roundtrip_flat_action_space_over_gameplay(self):
        players = [Player(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        steps = 0
        # Exercise a sequence of turns, validating action index <-> move round-trip each time
        while not g.is_game_over() and steps < MAX_TURNS:
            pid = g.current_player_idx
            gp = g.get_game_perspective(pid)
            mask = gp.legal_flat_mask()
            # For every legal index, reconstruct a move and verify legality and index consistency
            for idx, bit in enumerate(mask):
                if bit != 1:
                    continue
                move = build_move_from_flat(gp, idx)
                # Some indices may not reconstruct due to ambiguity; they should yield a valid legal move when possible
                self.assertIsNotNone(move, f"Failed to build move for idx={idx} at step={steps} pid={pid}")
                self.assertTrue(g.is_legal(pid, move), f"Rebuilt move not legal for idx={idx} at step={steps} pid={pid}")
                idx2 = flat_index_for_action(gp, move)
                self.assertEqual(idx2, idx, f"Round-trip mismatch: idx={idx} -> move -> idx2={idx2}")
            # Advance the game one turn
            g.play_turn()
            steps += 1

    def _assert_gp_equals(self, gp, gp2):
        # Hand tiles preserved (ignoring order normalization differences beyond sort)
        self.assertEqual(sorted([str(t) for t in gp.player_hand]), sorted([str(t) for t in gp2.player_hand]))
        # Last discard preserved
        if gp.last_discarded_tile is not None and gp2.last_discarded_tile is not None:
            self.assertEqual(str(gp.last_discarded_tile), str(gp2.last_discarded_tile))
        self.assertEqual(gp.last_discard_player, gp2.last_discard_player)
        # Discards preserved for P0 including aka flags when present
        self.assertEqual([str(t) + ('r' if t.aka else '') for t in gp.player_discards[0]],
                         [str(t) + ('r' if t.aka else '') for t in gp2.player_discards[0]])

    def test_feature_engineering_roundtrip_basic(self):
        # Build a simple game state and ensure encode/decode preserves key fields
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        while not g.is_game_over():
            gp = g.get_game_perspective(g.current_player_idx)
            self._assert_gp_equals(gp, decode_game_perspective(encode_game_perspective(gp)))
            g.play_turn()


    def test_serialization_shapes_fixed_over_game(self):
        # Verify that encode_game_perspective returns fixed-size arrays each turn
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        steps = 0
        while not g.is_game_over() and steps < 100:
            pid = g.current_player_idx
            gp = g.get_game_perspective(pid)
            feat = encode_game_perspective(gp)
            self.assertEqual(feat['hand_idx'].shape, (MAX_HAND_LEN + 1,))
            self.assertEqual(feat['called_idx'].shape[0], NUM_PLAYERS)
            self.assertEqual(feat['disc_idx'].shape[0], NUM_PLAYERS)
            # called_discards is present but ignored by the network; assert shape parity with discards
            self.assertIn('called_discards', feat)
            self.assertEqual(feat['called_discards'].shape, feat['disc_idx'].shape)
            # Assert indices are within valid bounds or PAD (-1)
            import numpy as np
            from core.learn.ac_constants import TILE_INDEX_SIZE
            for arr in (feat['hand_idx'], feat['called_idx'], feat['disc_idx']):
                a = np.asarray(arr)
                valid = (a >= 0)
                self.assertFalse(np.any(a[valid] >= int(TILE_INDEX_SIZE)))
            g.play_turn()
            steps += 1

    def test_perspective_rotation_invariance(self):
        # P0 will discard a specific tile; after the turn completes, from P1's perspective
        # the last discard should appear as player index 3 (left of P1), not index 1.
        target = Tile(Suit.PINZU, TileType.THREE)
        g = MediumJong([ForceDiscardPlayer(0, target), Player(1), Player(2), Player(3)])
        # Ensure P0 has the target in hand; if not, play until it appears or game ends
        # Play one action where P0 discards the target
        gp0 = g.get_game_perspective(0)
        # If target not in hand, skip the test early (non-deterministic decks)
        if target not in gp0.player_hand:
            return
        g.play_turn()
        # Now it's reaction state for players 1..3; get P1 perspective
        gp1 = g.get_game_perspective(1)
        self.assertIsNotNone(gp1.last_discarded_tile)
        self.assertEqual(str(gp1.last_discarded_tile), str(target))
        # Last discard player should be relative index 3 from P1's perspective (left of P1)
        self.assertEqual(gp1.last_discard_player, 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)


