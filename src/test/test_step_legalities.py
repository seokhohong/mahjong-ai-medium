#!/usr/bin/env python3
"""
Unit tests adapted for MediumJong legality helpers
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong, Player, Tile, TileType, Discard, Tsumo, Ron, Suit, Pon, Chi, CalledSet


class TestMediumStepLegalities(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i) for i in range(4)]
        self.game = MediumJong(self.players)

    def test_game_initialization(self):
        self.assertEqual(len(self.game.players), 4)
        # Medium deals 13 tiles initially
        for i in range(MediumJong.NUM_PLAYERS):
            self.assertEqual(len(self.game.hand(i)), 13)

    def test_tile_string_representation(self):
        tile = Tile(Suit.PINZU, TileType.FIVE)
        self.assertEqual(str(tile), "5p")

    def test_action_tsumo_detection(self):
        # Compose a 13-tile closed hand that wins on self-draw with menzen tsumo yaku
        tiles = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
        ]
        self.game._player_hands[0] = tiles.copy()
        self.game.current_player_idx = 0
        # Deterministic draw that completes 456s -> tanyao not needed because menzen tsumo provides yaku
        draw_tile = Tile(Suit.SOUZU, TileType.SIX)
        self.game.tiles = [draw_tile]
        # Simulate turn
        self.game.play_turn()
        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.get_winners(), [0])
        gp = self.game.get_game_perspective(0)
        self.assertIsNotNone(gp)

    def test_reaction_chi_detection_for_left_player(self):
        self.game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        self.game.last_discard_player = 0
        self.game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        rs = self.game.get_game_perspective(1)
        options = rs.get_call_options()
        self.assertGreaterEqual(len(options['chi']), 1)
        # Player 2 not left
        self.game._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        rs2 = self.game.get_game_perspective(2)
        options2 = rs2.get_call_options()
        self.assertEqual(len(options2['chi']), 0)

    def test_reaction_pon_detection(self):
        self.game.last_discarded_tile = Tile(Suit.SOUZU, TileType.FIVE)
        self.game.last_discard_player = 0
        self.game._player_hands[2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)]
        rs = self.game.get_game_perspective(2)
        options = rs.get_call_options()
        self.assertGreaterEqual(len(options['pon']), 1)

    def test_reaction_ron_detection(self):
        self.game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        self.game.last_discard_player = 0
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        # Ensure 13 tiles before ron by adding a pair
        pair = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        self.game._player_hands[1] = base_s + pair + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        rs = self.game.get_game_perspective(1)
        self.assertTrue(rs.can_ron())


if __name__ == '__main__':
    unittest.main(verbosity=2)


