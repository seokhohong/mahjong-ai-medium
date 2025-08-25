#!/usr/bin/env python3
import unittest
import sys
import os
from typing import List

from core.game import MediumJong, Player, GamePerspective
from core.tile import Tile, TileType, Suit, Honor
from core.action import Tsumo, Reaction, PassCall
from core.constants import (
    HANEMAN_NON_DEALER_RON_POINTS,
    HANEMAN_NON_DEALER_TSUMO_DEALER_PAYMENT,
    HANEMAN_NON_DEALER_TSUMO_OTHERS_PAYMENT,
    BAIMAN_DEALER_TSUMO_PAYMENT_EACH,
    BAIMAN_DEALER_RON_POINTS,
)
# Ensure this test directory is importable for test_utils
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import ForceDiscardPlayer, ForceActionPlayer, NoReactionPlayer


def simple_tanyao_13():
    # 13-tile closed hand with only simples and ready to win on a simple draw
    # 234m, 345p, 456s, 56m, pair 77p (waiting on 4m or 6m)
    return [
        Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
        Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
        Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
        Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
        Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
    ]


class TestLimitScoring(unittest.TestCase):
    def test_haneman_non_dealer_ron_points(self):
        # Non-dealer (P1) wins by ron with 6 han (tanyao 1 + 5 dora)
        # Simulate via a forced discard and assert using get_points().
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.MANZU, TileType.SEVEN)),  # P0 will discard 6m
            Player(),
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        # Winner is player 1 (non-dealer); use a simple tanyao 13 with ryanmen on 6m/4m
        g._player_hands[1] = simple_tanyao_13()
        # Make 3p the dora by setting 5 indicators 2p -> each grants dora for the single 3p in hand
        g.dora_indicators = [Tile(Suit.PINZU, TileType.TWO) for _ in range(5)]
        # Ensure the forced discarder has the target tile and allow a draw
        g._player_hands[0][0] = Tile(Suit.MANZU, TileType.SEVEN)
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]  # harmless draw for P0 before discard
        # Play the single discard -> P1 rons
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[1], HANEMAN_NON_DEALER_RON_POINTS)
        self.assertEqual(pts[0], -HANEMAN_NON_DEALER_RON_POINTS)

    def test_haneman_non_dealer_tsumo_split(self):
        class ForceTsumoPlayer(Player):
            def __init__(self):
                super().__init__()
                self.action = Tsumo()

            def play(self, gs):  # type: ignore[override]
                if gs.is_legal(self.action):
                    return self.action
                return super().play(gs)

            def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]) -> Reaction:
                return PassCall()

        # Non-dealer tsumo with 6 han -> split per Haneman constants
        g = MediumJong([ForceDiscardPlayer(Honor.NORTH), ForceTsumoPlayer(), Player(), Player()])
        g._player_hands[1] = simple_tanyao_13()
        # 5 indicators 2p => 3p is dora; tanpin tsumo dora 4 = 7 han
        g.dora_indicators = [Tile(Suit.PINZU, TileType.TWO) for _ in range(4)]
        # Make it player 1's turn and draw a winning simple tile (e.g., 6m)
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.HONORS, Honor.NORTH)]
        g.dead_wall = []
        g.play_turn()
        # Play; P1 should tsumo
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[1], HANEMAN_NON_DEALER_TSUMO_DEALER_PAYMENT + 2 * HANEMAN_NON_DEALER_TSUMO_OTHERS_PAYMENT)
        # Losers' totals should sum to negative of winner
        self.assertEqual(sum(pts), 0)

    def test_baiman_dealer_tsumo_total(self):
        # Dealer tsumo with 8 han (tanyao 1 + 7 dora) -> BAIMAN dealer tsumo each * 3
        g = MediumJong([ForceActionPlayer(Tsumo()), Player(), Player(), Player()])
        g._player_hands[0] = simple_tanyao_13()
        # 7 indicators 2p => 3p dora; hand has one 3p -> +7 han with tanyao 1 = 8 han
        g.dora_indicators = [Tile(Suit.PINZU, TileType.TWO) for _ in range(5)]
        # Draw winning tile for dealer
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN)]
        g.dead_wall = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[0], BAIMAN_DEALER_TSUMO_PAYMENT_EACH * 3)

    def test_baiman_dealer_ron_points(self):
        # Dealer ron with 8 han -> BAIMAN dealer ron points
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.HONORS, Honor.NORTH)),
            ForceDiscardPlayer(Tile(Suit.MANZU, TileType.SEVEN)),  # P1 will discard 7m to be ron'd by dealer
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        g._player_hands[0] = simple_tanyao_13()
        # 7 indicators 2p => 3p dora; hand has one 3p -> +7 han + tanyao 1 = 8 han
        g.dora_indicators = [Tile(Suit.PINZU, TileType.TWO) for _ in range(7)]
        # Ensure discarder has 7m and allow a draw
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.HONORS, Honor.NORTH)]
        g.play_turn()
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[0], BAIMAN_DEALER_RON_POINTS)
        self.assertEqual(pts[1], -BAIMAN_DEALER_RON_POINTS)


if __name__ == '__main__':
    unittest.main()
