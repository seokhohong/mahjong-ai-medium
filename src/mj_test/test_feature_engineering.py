import unittest

from core.constants import MAX_CALLS, NUM_PLAYERS
from core.game import MediumJong, Player
from core.learn.policy_utils import build_move_from_two_head, encode_two_head_action
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
        players = [Player(), Player(), Player(), Player()]
        g = MediumJong(players)
        steps = 0
        # Exercise a sequence of turns, validating action index <-> move round-trip each time
        while not g.is_game_over() and steps < MAX_TURNS:
            pid = g.current_player_idx
            gp = g.get_game_perspective(pid)
            legal_moves = gp.legal_moves()
            # For every legal index, reconstruct a move and verify legality and index consistency
            for move in legal_moves:
                idx = encode_two_head_action(move)
                # Some indices may not reconstruct due to ambiguity; they should yield a valid legal move when possible
                self.assertIsNotNone(move, f"Failed to build move for idx={idx} at step={steps} pid={pid}")
                self.assertTrue(g.is_legal(pid, move), f"Rebuilt move not legal for idx={idx} at step={steps} pid={pid}")
                idx2 = encode_two_head_action(build_move_from_two_head(gp, idx[0], idx[1]))
                self.assertEqual(idx2, idx, f"Round-trip mismatch: idx={idx} -> move -> idx2={idx2}")
            # Advance the game one turn
            g.play_turn()
            steps += 1

    def _assert_gp_equals(self, gp, gp2):
        # Helper to stringify tiles including aka flag
        def ts(tiles):
            return [str(t) + ('r' if getattr(t, 'aka', False) else '') for t in tiles]

        # Hand tiles preserved (ignoring order)
        self.assertEqual(sorted([str(t) for t in gp.player_hand]), sorted([str(t) for t in gp2.player_hand]))

        # Reactable tile and owner
        self.assertEqual(str(gp.reactable_tile()) if gp.reactable_tile() else None,
                         str(gp2.reactable_tile()) if gp2.reactable_tile() else None)
        self.assertEqual(gp._owner_of_reactable_tile, gp2._owner_of_reactable_tile)

        # Discards preserved for all players
        for pid in range(4):
            self.assertEqual(ts(gp.player_discards.get(pid, [])), ts(gp2.player_discards.get(pid, [])))

        # Called sets preserved (tiles and grouping)
        for pid in range(4):
            cs1 = gp.called_sets.get(pid, [])
            cs2 = gp2.called_sets.get(pid, [])
            self.assertEqual(len(cs1), len(cs2))
            for a, b in zip(cs1, cs2):
                self.assertEqual(ts(a.tiles), ts(b.tiles))

        # Called discards preserved
        self.assertEqual(gp.called_discards, gp2.called_discards)

        # Winds preserved
        self.assertEqual(gp.round_wind.value, gp2.round_wind.value)
        for pid in range(4):
            self.assertEqual(gp.seat_winds[pid].value, gp2.seat_winds[pid].value)

        # Riichi declarations preserved
        self.assertEqual(gp.riichi_declaration_tile, gp2.riichi_declaration_tile)

        # Remaining tiles preserved
        self.assertEqual(gp.remaining_tiles, gp2.remaining_tiles)

        # Newly drawn tile preserved
        self.assertEqual(str(gp.newly_drawn_tile) if gp.newly_drawn_tile else None,
                         str(gp2.newly_drawn_tile) if gp2.newly_drawn_tile else None)

        # Dora indicators preserved (order up to first -1 index)
        self.assertEqual([str(t) for t in getattr(gp, 'dora_indicators', []) or []],
                         [str(t) for t in getattr(gp2, 'dora_indicators', []) or []])

        # Legal action mask preserved
        lam1 = gp.legal_action_mask()
        lam2 = gp2.legal_action_mask()
        self.assertEqual(len(lam1), len(lam2))
        self.assertTrue(all(int(a) == int(b) for a, b in zip(lam1, lam2)))

    def test_feature_engineering_roundtrip_basic(self):
        # Build a simple game state and ensure encode/decode preserves key fields
        g = MediumJong([Player(), Player(), Player(), Player()])
        while not g.is_game_over():
            gp = g.get_game_perspective(g.current_player_idx)
            self._assert_gp_equals(gp, decode_game_perspective(encode_game_perspective(gp)))
            g.play_turn()


    def test_serialization_shapes_fixed_over_game(self):
        # Verify that encode_game_perspective returns fixed-size arrays each turn
        g = MediumJong([Player(), Player(), Player(), Player()])
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
        import random
        random.seed(234)
        target = Tile(Suit.PINZU, TileType.THREE)
        g = MediumJong([ForceDiscardPlayer(target), Player(), Player(), Player()])
        # Ensure P0 has the target in hand; if not, play until it appears or game ends
        # Play one action where P0 discards the target
        gp0 = g.get_game_perspective(0)
        # If target not in hand, skip the test early (non-deterministic decks)
        if target not in gp0.player_hand:
            return
        g.play_turn()
        # Now it's reaction state for players 1..3; get P1 perspective
        gp1 = g.get_game_perspective(1)
        self.assertIsNotNone(gp1._reactable_tile)
        self.assertEqual(str(gp1._reactable_tile), str(target))
        # Last discard player should be relative index 3 from P1's perspective (left of P1)
        self.assertEqual(gp1._owner_of_reactable_tile, 3)
        # test for player 2's perspective as well
        gp2 = g.get_game_perspective(2)
        self.assertEqual(gp2._owner_of_reactable_tile, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)


