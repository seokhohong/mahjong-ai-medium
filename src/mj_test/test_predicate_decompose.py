#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import _decompose_standard_with_pred
from core.tile import Tile, TileType, Suit, Honor


class TestDecomposeWithPred(unittest.TestCase):
    def test_honor_pair_with_standard_melds(self):
        # Hand: pair EAST EAST; melds: 123m, 456p, 789s, 555m
        tiles = [
            # 123m
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            # 456p
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            # 789s
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            # 555m triplet
            Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE),
            # Pair EAST EAST
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST),
        ]

        def pred_meld(meld):
            # Accept any meld (sequence or triplet)
            return True

        def pred_pair(tile):
            # Only accept honors as the pair
            return tile.suit == Suit.HONORS

        self.assertTrue(_decompose_standard_with_pred(tiles, pred_meld, pred_pair))


if __name__ == '__main__':
    unittest.main(verbosity=2)


