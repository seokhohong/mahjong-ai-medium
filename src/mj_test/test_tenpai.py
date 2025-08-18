#!/usr/bin/env python3
"""
Unit tests for tenpai detection helpers.
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import Tile, TileType, Suit, Honor, hand_is_tenpai, hand_is_tenpai_for_tiles


class TestTenpaiHelpers(unittest.TestCase):
    def test_standard_single_wait_tenpai(self):
        # Base 1-9 souzu (3 sequences), pair 77m, and 2p 4p -> wait on 3p
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        hand = base_s + [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
                         Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        self.assertEqual(len(hand), 13)
        self.assertTrue(hand_is_tenpai_for_tiles(hand))
        self.assertTrue(hand_is_tenpai(hand))

    def test_chiitoi_tenpai(self):
        # Six pairs + one singleton -> chiitoitsu tenpai
        hand = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.HONORS, Honor.WHITE),
        ]
        self.assertEqual(len(hand), 13)
        self.assertTrue(hand_is_tenpai_for_tiles(hand))
        self.assertTrue(hand_is_tenpai(hand))

    def test_multi_wait_tenpai_two_waits(self):
        # 1-2-3m, 4-5-6m, 1-2-3p, 7-8m, pair EE -> waits on 6m or 9m
        hand = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST),
        ]
        self.assertEqual(len(hand), 13)
        self.assertTrue(hand_is_tenpai_for_tiles(hand))
        self.assertTrue(hand_is_tenpai(hand))

    def test_not_tenpai(self):
        # A random 13-tile hand that is not one away from completion
        hand = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST), Tile(Suit.HONORS, Honor.NORTH),
        ]
        self.assertEqual(len(hand), 13)
        self.assertFalse(hand_is_tenpai_for_tiles(hand))
        self.assertFalse(hand_is_tenpai(hand))


if __name__ == '__main__':
    unittest.main(verbosity=2)


