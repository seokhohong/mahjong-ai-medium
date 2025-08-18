import unittest

from core.game import MediumJong, Player
from core.learn.policy_utils import build_move_from_flat, flat_index_for_action
from core.game import (
    MediumJong, Player, Tile, TileType, Suit, Honor,
    Discard, Tsumo, Pon, Chi, Riichi,
    KanDaimin, KanAnkan, CalledSet,
)
from core.learn.feature_engineering import encode_game_perspective, decode_game_perspective

class TestFeatureEngineering(unittest.TestCase):
    def test_roundtrip_flat_action_space_over_gameplay(self):
        players = [Player(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        steps = 0
        # Exercise a sequence of turns, validating action index <-> move round-trip each time
        while not g.is_game_over() and steps < 200:
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
                if move is None:
                    continue
                self.assertTrue(g.is_legal(pid, move), f"Rebuilt move not legal for idx={idx} at step={steps} pid={pid}")
                idx2 = flat_index_for_action(gp, move)
                self.assertEqual(idx2, idx, f"Round-trip mismatch: idx={idx} -> move -> idx2={idx2}")
            # Advance the game one turn
            g.play_turn()
            steps += 1

    def test_feature_engineering_roundtrip_basic(self):
        # Build a simple game state and ensure encode/decode preserves key fields
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g.current_player_idx = 0
        # Deterministic: set small known hand and discards
        g._player_hands[0] = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.HONORS, Honor.WHITE),
        ]
        g.player_discards[0] = [Tile(Suit.PINZU, TileType.THREE)]
        g.last_discarded_tile = g.player_discards[0][-1]
        g.last_discard_player = 0
        g.last_drawn_tile = None
        g.last_drawn_player = None
        gp = g.get_game_perspective(0)
        features = encode_game_perspective(gp)
        gp2 = decode_game_perspective(features)
        # Hand tiles preserved (ignoring order normalization differences beyond sort)
        self.assertEqual(sorted([str(t) for t in gp.player_hand]), sorted([str(t) for t in gp2.player_hand]))
        # Last discard preserved
        if gp.last_discarded_tile is not None and gp2.last_discarded_tile is not None:
            self.assertEqual(str(gp.last_discarded_tile), str(gp2.last_discarded_tile))
        self.assertEqual(gp.last_discard_player, gp2.last_discard_player)
        # Discards preserved for P0 including aka flags when present
        self.assertEqual([str(t) + ('r' if t.aka else '') for t in g.player_discards[0]],
                         [str(t) + ('r' if t.aka else '') for t in gp2.player_discards[0]])

    def test_serialization_shapes_fixed_over_game(self):
        # Verify that encode_game_perspective returns fixed-size arrays each turn
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        steps = 0
        while not g.is_game_over() and steps < 100:
            pid = g.current_player_idx
            gp = g.get_game_perspective(pid)
            feat = encode_game_perspective(gp)
            self.assertEqual(feat['hand_idx'].shape, (14,))
            self.assertEqual(feat['called_idx'].shape[0], 4)
            self.assertEqual(feat['disc_idx'].shape[0], 4)
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

if __name__ == '__main__':
    unittest.main(verbosity=2)


