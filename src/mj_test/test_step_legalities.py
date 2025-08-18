#!/usr/bin/env python3
"""
Unit tests adapted for MediumJong legality helpers
"""

import unittest
import sys
import os

# Add this test directory to Python path to import helpers
sys.path.insert(0, os.path.dirname(__file__))

from core.game import MediumJong, Player, Tile, TileType, Discard, Tsumo, Ron, Suit, Pon, Chi, CalledSet
from test_utils import ForceDiscardPlayer


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
        # Deterministic draw that completes 456s -> tanyao not needed because menzen tsumo provides yaku
        draw_tile = Tile(Suit.SOUZU, TileType.SIX)
        self.game.tiles = [draw_tile]
        # Simulate turn
        self.game.play_turn()
        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.get_winners(), [0])


    def test_reaction_chi_on_3p_discard(self):
        # Player 0 will discard 3p
        p0 = ForceDiscardPlayer(0, Tile(Suit.PINZU, TileType.THREE))
        game = MediumJong([p0, Player(1), Player(2), Player(3)])
        # Ensure player 0 has 3p in hand to discard
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        # Ensure player 1 (left of 0) can Chi with 2p and 4p
        game._player_hands[1][:2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Directly perform the discard to enter reaction phase without auto-resolution
        game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Player 1 should be able to Chi
        chi_move = Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])
        self.assertTrue(game.is_legal(1, chi_move))

    def test_reaction_pon_on_3p_discard(self):
        # Player 0 will discard 3p
        p0 = ForceDiscardPlayer(0, Tile(Suit.PINZU, TileType.THREE))
        game = MediumJong([p0, Player(1), Player(2), Player(3)])
        # Ensure player 0 has 3p in hand to discard
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        # Ensure player 2 has two 3p to Pon
        game._player_hands[2][:2] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)]
        # Discard 3p
        game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        pon_move = Pon([Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)])
        self.assertTrue(game.is_legal(2, pon_move))

    def test_reaction_ron_on_3p_discard(self):
        # Player 0 will discard 3p
        p0 = ForceDiscardPlayer(0, Tile(Suit.PINZU, TileType.THREE))
        game = MediumJong([p0, Player(1), Player(2), Player(3)])
        # Ensure player 0 has 3p in hand to discard
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        # Configure player 1 to be closed tenpai waiting on 3p with a yaku (pinfu):
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        pair = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        game._player_hands[1] = base_s + pair + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Discard 3p from player 0
        game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Player 1 should be able to Ron
        self.assertTrue(game.is_legal(1, Ron()))




if __name__ == '__main__':
    unittest.main(verbosity=2)


