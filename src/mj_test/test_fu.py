#!/usr/bin/env python3
import unittest

from core.game import MediumJong, Player, CalledSet
from core.action import Discard, Tsumo, KanAnkan
from core.tile import Tile, TileType, Suit, Honor
from mj_test.test_utils import NoReactionPlayer


class TsumoIfPossible(Player):
    def play(self, gs):  # type: ignore[override]
        if gs.can_tsumo():
            return Tsumo()
        return Discard(gs.player_hand[0])


class TestFuScoring(unittest.TestCase):
    def _reset_env(self, g: MediumJong):
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g._reactable_tile = None
        g._owner_of_reactable_tile = None

    def test_fu_pinfu_tsumo_20(self):
        # Closed pinfu tsumo: all sequences, non-yakuhai pair, ryanmen wait
        g = MediumJong([TsumoIfPossible(), Player(), Player(), Player()])
        # Hand: 234m, 345p, 456s, pair 77p, wait on 6m to complete 456m
        tiles_13 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
            Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.MANZU, TileType.FIVE),  # will form 345m after draw to keep all sequences
        ]
        g._player_hands[0] = tiles_13
        self._reset_env(g)
        # Winning draw: 6m to make ryanmen 45m-6m -> 456m (pinfu tsumo waives +2 fu)
        g.tiles = [Tile(Suit.MANZU, TileType.ONE)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(s['fu'], 20)

    def test_fu_pinfu_ron_30(self):
        # Closed pinfu ron: 20 base + 10 menzen ron = 30
        class DiscardThreePin(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.PINZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])
        g = MediumJong([DiscardThreePin(), Player(), Player(), Player()])
        # Player 1 closed pinfu waiting on 3p ryanmen
        g._player_hands[1] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),  # ryanmen 2-4p waiting 3p
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN),  # non-yakuhai pair
        ]
        self._reset_env(g)
        # Ensure player 0 can discard 3p immediately
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g._score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['fu'], 30)

    def test_fu_chiitoitsu_25(self):
        # Seven pairs fixed 25 fu
        g = MediumJong([TsumoIfPossible(), Player(), Player(), Player()])
        g._player_hands[0] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.SIX), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.HONORS, Honor.WHITE),
        ]
        self._reset_env(g)
        g.current_player_idx = 0
        # Draw final pair
        g.tiles = [Tile(Suit.HONORS, Honor.WHITE)]
        g.play_turn()
        s = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(s['fu'], 25)

    def test_fu_closed_kanchan_ron_rounds_40(self):
        # Closed hand with kanchan wait ron: 20 + 2 (kanchan) + 10 menzen ron = 32 -> 40
        class DiscardFiveMan(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.MANZU, TileType.FIVE)
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])
        g = MediumJong([DiscardFiveMan(), Player(), NoReactionPlayer(), NoReactionPlayer()])
        # Player 1 waits KANCHAN on 5m for 4-5-6m
        g._player_hands[1] = [
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.SIX),  # kanchan
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN),
        ]
        self._reset_env(g)
        g.tiles = [Tile(Suit.MANZU, TileType.FIVE)]
        g.play_turn()
        s = g._score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['fu'], 40)

    def test_fu_open_hand_no_fu_sets_to_30(self):
        # Open pinfu-like hand should be set to 30 fu on ron
        class DiscardThreePin(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.PINZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])
        g = MediumJong([DiscardThreePin(), Player(), Player(), Player()])
        # Player 1: call chi to open, keep all sequences, non-yakuhai pair, ryanmen wait
        base_closed = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),  # will wait 3p
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN),
        ]
        g._player_hands[1] = list(base_closed)
        # Pre-open with a chi set 1-2-3m from someone to ensure open status and still no fu sources
        called = CalledSet([
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE)
        ], 'chi', Tile(Suit.MANZU, TileType.TWO), caller_position=1, source_position=0)
        g._player_called_sets[1] = [called]
        self._reset_env(g)
        # Player 0 discards 3p -> Player 1 ron
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        s = g._score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['fu'], 30)

    def test_fu_closed_triplet_simple_adds_4fu(self):
        # Closed ron hand containing a closed triplet of a simple tile adds 4 fu -> 20 + 4 + 10 = 34 -> 40
        class DiscardThreeMan(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.MANZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])
        g = MediumJong([DiscardThreeMan(), NoReactionPlayer(), Player(), Player()])
        # Triplet of 2m closed; rest sequences; ryanmen ron on 3m
        g._player_hands[1] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        self._reset_env(g)
        g._player_hands[0][0] = Tile(Suit.MANZU, TileType.THREE)
        g.tiles = [Tile(Suit.MANZU, TileType.FIVE)]
        g.play_turn()
        s = g._score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['fu'], 40)

    def test_fu_closed_ankan_honor_tsumo_rounds_60(self):
        # Closed ankan of honors (e.g., NORTH): 32 fu, tsumo +2, base 20 -> 54 -> 60
        class AnkanThenTsumo(Player):
            def play(self, gs):  # type: ignore[override]
                # Force Ankan if available, else Tsumo if available, else discard first
                for m in gs.legal_moves():
                    if isinstance(m, KanAnkan):
                        return m
                if gs.can_tsumo():
                    return Tsumo()
                return Discard(gs.player_hand[0])

        g = MediumJong([AnkanThenTsumo(), Player(), Player(), Player()])
        # Start with four NORTH for immediate ankan; rest sequences to avoid other fu
        g._player_hands[0] = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        # Arrange dead wall so rinshan draw completes a sequence without adding fu besides +2 tsumo
        self._reset_env(g)
        g.dead_wall = [Tile(Suit.HONORS, Honor.EAST), Tile(Suit.SOUZU, TileType.SIX)]
        # Provide draw that completes 3p4p5p from base
        g.tiles = [Tile(Suit.PINZU, TileType.FIVE)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(s['fu'], 60)

    def test_fu_pair_dragon_adds_2fu(self):
        # Closed ron, pair is a dragon -> +2 fu; assume ryanmen wait
        class DiscardThreeSou(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.SOUZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])
        g = MediumJong([DiscardThreeSou(), NoReactionPlayer(), Player(), Player()])
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.FOUR),  # ryanmen on 3s
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT), Tile(Suit.MANZU, TileType.NINE),
            Tile(Suit.HONORS, Honor.WHITE), Tile(Suit.HONORS, Honor.WHITE),
        ]
        self._reset_env(g)
        g._player_hands[0][0] = Tile(Suit.SOUZU, TileType.THREE)
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        s = g._score_hand(1, win_by_tsumo=False)
        # 20 base + 2 (pair) + 10 menzen ron = 32 -> 40
        self.assertEqual(s['fu'], 40)


if __name__ == '__main__':
    unittest.main()
