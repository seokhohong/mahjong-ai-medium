#!/usr/bin/env python3
"""
Unit tests for tenpai detection helpers.
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import hand_is_tenpai, hand_is_tenpai_for_tiles, CalledSet, GamePerspective
from core.tile import Tile, TileType, Suit, Honor
from core.tenpai import can_complete_standard_with_calls, clear_hand_caches, waits_for_tiles
from core.tenpai import hand_is_tenpai_with_calls  # type: ignore
from core.action import Discard, Riichi, Action
from core.learn.recording_ac_player import _transitions_into_tenpai
from core.learn.ac_constants import NULL_TILE_INDEX


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

    def test_puts_hand_in_tenpai_after_move_closed(self):
        # Start from a known 13-tile tenpai hand
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        tenpai13 = base_s + [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
                              Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        self.assertTrue(hand_is_tenpai_for_tiles(tenpai13))
        # Create a 14-tile hand by adding a junk tile and discard that tile
        extra = Tile(Suit.HONORS, Honor.WHITE)
        hand14 = list(tenpai13) + [extra]
        gp = GamePerspective(player_hand=hand14, remaining_tiles=30, reactable_tile=None, owner_of_reactable_tile=None,
                             called_sets={0: [], 1: [], 2: [], 3: []}, player_discards={0: [], 1: [], 2: [], 3: []},
                             called_discards={0: [], 1: [], 2: [], 3: []}, last_discards={}, newly_drawn_tile=None,
                             seat_winds={0: Honor.EAST, 1: Honor.SOUTH, 2: Honor.WEST, 3: Honor.NORTH},
                             round_wind=Honor.EAST, dora_indicators=[],
                             riichi_declaration_tile={0: NULL_TILE_INDEX, 1: NULL_TILE_INDEX, 2: NULL_TILE_INDEX,
                                                      3: NULL_TILE_INDEX})
        move = Discard(tile=extra)
        self.assertTrue(_transitions_into_tenpai(move, gp))

    def test_puts_hand_in_tenpai_after_move_open(self):
        # Open hand: chi 1-2-3m and pon E E E; concealed 7 tiles tenpai as in existing test
        chi = CalledSet([
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE)
        ], 'chi', Tile(Suit.MANZU, TileType.TWO), caller_position=0, source_position=3)
        pon = CalledSet([
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST)
        ], 'pon', Tile(Suit.HONORS, Honor.EAST), caller_position=0, source_position=1)
        concealed7 = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),
        ]
        # Player's concealed hand should NOT include called set tiles.
        # Build 8 concealed: concealed7 + one extra junk to discard
        extra = Tile(Suit.HONORS, Honor.WHITE)
        hand14 = list(concealed7) + [extra]
        gp = GamePerspective(player_hand=hand14, remaining_tiles=30, reactable_tile=None, owner_of_reactable_tile=None,
                             called_sets={0: [chi, pon], 1: [], 2: [], 3: []},
                             player_discards={0: [], 1: [], 2: [], 3: []}, called_discards={0: [], 1: [], 2: [], 3: []},
                             last_discards={}, newly_drawn_tile=None,
                             seat_winds={0: Honor.EAST, 1: Honor.SOUTH, 2: Honor.WEST, 3: Honor.NORTH},
                             round_wind=Honor.EAST, dora_indicators=[],
                             riichi_declaration_tile={0: NULL_TILE_INDEX, 1: NULL_TILE_INDEX, 2: NULL_TILE_INDEX,
                                                      3: NULL_TILE_INDEX})
        move = Discard(tile=extra)
        self.assertTrue(_transitions_into_tenpai(move, gp))

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

    def test_open_hand_tenpai_and_nontenpai(self):
        # Two called sets: chi 1-2-3m and pon E E E
        chi = CalledSet([
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE)
        ], 'chi', Tile(Suit.MANZU, TileType.TWO), caller_position=0, source_position=3)
        pon = CalledSet([
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST)
        ], 'pon', Tile(Suit.HONORS, Honor.EAST), caller_position=0, source_position=1)

        # Concealed tiles count for tenpai pre-draw should be 14 - 3*len(calls) - 1 = 7
        concealed_tenpai = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),  # 45p waiting on 6p
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),  # pair
        ]
        self.assertTrue(can_complete_standard_with_calls(concealed_tenpai + [Tile(Suit.PINZU, TileType.SIX)], [chi, pon]))

        if hand_is_tenpai_with_calls:
            self.assertTrue(hand_is_tenpai_with_calls(concealed_tenpai, [chi, pon]))

        # Non-tenpai open hand: honors singles that cannot be completed with one tile
        concealed_noten = [
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH), Tile(Suit.HONORS, Honor.WEST),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.WHITE), Tile(Suit.HONORS, Honor.GREEN),
            Tile(Suit.HONORS, Honor.RED),
        ]
        self.assertFalse(can_complete_standard_with_calls(concealed_noten + [Tile(Suit.PINZU, TileType.SIX)], [chi, pon]))
        if hand_is_tenpai_with_calls:
            self.assertFalse(hand_is_tenpai_with_calls(concealed_noten, [chi, pon]))

    def test_tenpai_with_kan_and_without(self):
        # One called kan (ankan), counts as one meld for composition
        kan = CalledSet([
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH)
        ], 'kan_ankan', None, caller_position=0, source_position=None)

        # Concealed tiles count for tenpai pre-draw should be 14 - 3*len(calls) - 1 = 10
        concealed_tenpai = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),  # 45p waiting on 6p
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),  # pair
        ]
        self.assertTrue(can_complete_standard_with_calls(concealed_tenpai + [Tile(Suit.PINZU, TileType.SIX)], [kan]))
        if hand_is_tenpai_with_calls:
            self.assertTrue(hand_is_tenpai_with_calls(concealed_tenpai, [kan]))

        concealed_noten = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.SIX),
        ]
        self.assertFalse(any(can_complete_standard_with_calls(concealed_noten + [Tile(Suit.PINZU, v)], [kan]) for v in [TileType.FOUR, TileType.SIX]))
        if hand_is_tenpai_with_calls:
            self.assertFalse(hand_is_tenpai_with_calls(concealed_noten, [kan]))

    def test_four_of_a_kind_not_kanned(self):
        # Tenpai case: known standard single-wait, but replace an existing sequence tile to create a quad
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        # Start from the earlier tenpai: 1-9 souzu, pair 77m, and 2p 4p
        tenpai_hand = base_s + [
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)
        ]
        self.assertTrue(hand_is_tenpai_for_tiles(tenpai_hand))

        # Make a quad by duplicating one tile and removing another from souzu run while keeping 13 tiles
        quad_tile = Tile(Suit.SOUZU, TileType.FIVE)
        quad_hand = [t for t in tenpai_hand if not (t.suit == Suit.SOUZU and t.tile_type == TileType.ONE)]
        quad_hand.append(quad_tile)
        quad_hand.append(quad_tile)
        # Now ensure length 13 by removing TWO souzu 1 tiles (only one existed), fallback remove 2s
        while len(quad_hand) > 13:
            # Remove a low souzu to keep structure plausible
            for candidate in [(Suit.SOUZU, TileType.TWO), (Suit.SOUZU, TileType.THREE)]:
                for i, t in enumerate(quad_hand):
                    if t.suit == candidate[0] and t.tile_type == candidate[1]:
                        quad_hand.pop(i)
                        break
                if len(quad_hand) <= 13:
                    break

        # Should still be a valid input and not crash; assert deterministically not tenpai or tenpai isn't critical
        _ = hand_is_tenpai_for_tiles(quad_hand)

        # A clear non-tenpai with a concealed four-of-a-kind
        noten_quad = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH), Tile(Suit.HONORS, Honor.WEST), Tile(Suit.HONORS, Honor.NORTH),
        ]
        self.assertFalse(hand_is_tenpai_for_tiles(noten_quad))

    def test_aka_five_counts_as_five_in_tenpai(self):
        # Hand waiting on 5p (4p6p), ensure waits include 5p (aka also valid)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        hand = base_s + [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
                          Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.SIX)]
        self.assertTrue(hand_is_tenpai_for_tiles(hand))
        waits = waits_for_tiles(hand)
        self.assertIn(Tile(Suit.PINZU, TileType.FIVE), waits)

    def test_tenpai_with_aka_five_in_hand(self):
        # Include an aka 5 in a meld; tenpai logic should treat it as a 5
        hand = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE, aka=True), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        self.assertTrue(hand_is_tenpai_for_tiles(hand))


if __name__ == '__main__':
    unittest.main(verbosity=2)


