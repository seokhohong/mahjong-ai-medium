#!/usr/bin/env python3
import unittest

from core.game import MediumJong, Player
from core.tile import Tile, TileType, Suit
from core.action import Tsumo
from core.constants import (
    HANEMAN_NON_DEALER_RON_POINTS,
    HANEMAN_NON_DEALER_TSUMO_DEALER_PAYMENT,
    HANEMAN_NON_DEALER_TSUMO_OTHERS_PAYMENT,
    BAIMAN_DEALER_TSUMO_PAYMENT_EACH,
    BAIMAN_DEALER_RON_POINTS,
)


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
        # Non-dealer wins by ron with 6 han (tanyao 1 + 5 dora)
        g = MediumJong([Player(), Player(), Player(), Player()])
        # Winner is player 1 (non-dealer)
        g._player_hands[1] = simple_tanyao_13()
        # Ensure there are multiple copies of 5p in hand for dora counting
        # Hand already contains one 5p; add a second 5p by swapping a tile
        g._player_hands[1][2] = Tile(Suit.PINZU, TileType.FIVE)
        # Set 5 indicators to make 5p the dora tile (indicator 4p -> dora 5p)
        g.dora_indicators = [Tile(Suit.PINZU, TileType.FOUR) for _ in range(5)]
        # Configure loser context (ron)
        g._owner_of_reactable_tile = 0
        g.loser = 0
        s = g._score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['tsumo'], False)
        self.assertEqual(s['points'], HANEMAN_NON_DEALER_RON_POINTS)

    def test_haneman_non_dealer_tsumo_split(self):
        # Non-dealer tsumo with 6 han -> split per Haneman constants
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[1] = simple_tanyao_13()
        # Make dora to reach 6 han (tanyao 1 + 5 dora)
        g.dora_indicators = [Tile(Suit.PINZU, TileType.FOUR) for _ in range(5)]
        # Make it player 1's turn and draw winning tile
        g.current_player_idx = 1
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN)]  # any simple tile to allow tsumo action
        # Force tsumo
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g._score_hand(1, win_by_tsumo=True)
        self.assertTrue(s['tsumo'])
        self.assertIn('payments', s)
        self.assertEqual(s['payments']['from_dealer'], HANEMAN_NON_DEALER_TSUMO_DEALER_PAYMENT)
        self.assertEqual(s['payments']['from_others'], HANEMAN_NON_DEALER_TSUMO_OTHERS_PAYMENT)

    def test_baiman_dealer_tsumo_total(self):
        # Dealer tsumo with 8 han (tanyao 1 + 7 dora) -> BAIMAN dealer tsumo each * 3
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = simple_tanyao_13()
        # Indicators to make 5p dora; use 7 to reach 8 han
        g.dora_indicators = [Tile(Suit.PINZU, TileType.FOUR) for _ in range(7)]
        # Draw winning tile for dealer
        g.current_player_idx = 0
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g._score_hand(0, win_by_tsumo=True)
        self.assertTrue(s['tsumo'])
        self.assertEqual(s['points'], BAIMAN_DEALER_TSUMO_PAYMENT_EACH * 3)

    def test_baiman_dealer_ron_points(self):
        # Dealer ron with 8 han -> BAIMAN dealer ron points
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = simple_tanyao_13()
        g.dora_indicators = [Tile(Suit.PINZU, TileType.FOUR) for _ in range(7)]
        g._owner_of_reactable_tile = 1
        g.loser = 1
        s = g._score_hand(0, win_by_tsumo=False)
        self.assertFalse(s['tsumo'])
        self.assertEqual(s['points'], BAIMAN_DEALER_RON_POINTS)


if __name__ == '__main__':
    unittest.main()
