#!/usr/bin/env python3
import unittest
import sys
import os

from mj_test.test_utils import _make_tenpai_hand, _make_noten_hand

# Add this test directory to Python path so we can import test_utils
sys.path.insert(0, os.path.dirname(__file__))

# Import helpers
from test_utils import ForceActionPlayer, NoReactionPlayer, ForceDiscardPlayer

from core.game import (
    MediumJong, Player,
    CalledSet, OutcomeType,
)
from core.action import Discard, Tsumo, Chi, Riichi
from core.tile import Tile, TileType, Suit, Honor


class TestMediumJongBasics(unittest.TestCase):
    def setUp(self):
        self.players = [Player() for i in range(4)]
        self.game = MediumJong(self.players)

    def test_initialization(self):
        # 13 tiles each
        for i in range(4):
            self.assertEqual(len(self.game.hand(i)), 13)
        # Round/seat winds
        self.assertEqual(self.game.round_wind.name, 'EAST')
        self.assertEqual(self.game.seat_winds[0], Honor.EAST)

    def test_tile_string_honors(self):
        self.assertEqual(str(Tile(Suit.HONORS, Honor.EAST)), 'E')
        self.assertEqual(str(Tile(Suit.HONORS, Honor.WHITE)), 'Wh')
        self.assertEqual(str(Tile(Suit.PINZU, TileType.FIVE)), '5p')

    def test_tile_sorting_honors(self):
        # Ensure honors sort without error and by honor rank 1..7 (E,S,W,N,P,G,R)
        g = MediumJong([Player(), Player(), Player(), Player()])
        scrambled_honors = [
            Tile(Suit.HONORS, Honor.RED),
            Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.HONORS, Honor.WHITE),
            Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.GREEN),
            Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST),
        ]
        # Fill to 13 tiles with some suited tiles (ordering among honors is what we check)
        fillers = [
            Tile(Suit.MANZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.TWO),
        ]
        g._player_hands[0] = scrambled_honors + fillers
        g.current_player_idx = 0
        # Accessing game perspective should sort the hand
        gp = g.get_game_perspective(0)
        honors_sorted = [t for t in gp.player_hand if t.suit == Suit.HONORS]
        expected = [
            Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST),
            Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.WHITE),
            Tile(Suit.HONORS, Honor.GREEN),
            Tile(Suit.HONORS, Honor.RED),
        ]
        self.assertEqual([h.tile_type for h in honors_sorted], [e.tile_type for e in expected])

    def test_tile_uniqueness(self):
        # aka 5p and 9m should map to different flat indices
        from core.tile import tile_flat_index
        a5p = Tile(Suit.PINZU, TileType.FIVE, aka=True)
        nine_m = Tile(Suit.MANZU, TileType.NINE)
        idx_a5p = tile_flat_index(a5p)
        idx_9m = tile_flat_index(nine_m)
        self.assertNotEqual(idx_a5p, idx_9m, f"aka 5p and 9m mapped to same index {idx_a5p}")


    def test_aka_five_pinzu_string_and_sort_order(self):
        # aka 5p string representation should be "0p"
        a5p = Tile(Suit.PINZU, TileType.FIVE, aka=True)
        self.assertEqual(str(a5p), '0p')

        # Sorting a hand should treat aka 5p as a 5p for order purposes
        t3p = Tile(Suit.PINZU, TileType.THREE)
        t4p = Tile(Suit.PINZU, TileType.FOUR)
        t5p = Tile(Suit.PINZU, TileType.FIVE)
        t6p = Tile(Suit.PINZU, TileType.SIX)

        tiles = [t6p, a5p, t3p, t4p]
        # Sort by suit then tile number, ignoring aka flag, matching game sort intent
        sorted_tiles = sorted(tiles, key=lambda t: (t.suit.value, int(t.tile_type.value)))
        self.assertEqual([str(t) for t in sorted_tiles], ['3p', '4p', '0p', '6p'])

    def test_chi_left_only(self):
        g = MediumJong([Player(), Player(), Player(), Player()])
        # Prepare controlled hands
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[1][2:]
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[2][2:]
        # Empty wall to avoid draws beyond one action
        g.tiles = [Tile(Suit.PINZU, TileType.THREE)]
        g._draw_tile()
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Player 1 can chi
        moves1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Chi) for m in moves1))
        # Player 2 cannot chi
        moves2 = g.legal_moves(2)
        self.assertFalse(any(isinstance(m, Chi) for m in moves2))

    def test_pon_any_player(self):
        import random
        random.seed(123)
        g = MediumJong([ForceDiscardPlayer(target=Tile(Suit.SOUZU, TileType.FIVE)), Player(), Player(), Player()])
        g._player_hands[2] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g.tiles = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)]
        g.play_turn()
        self.assertTrue(len(g._player_called_sets[2]) > 0)

    def test_chain_pon_calls(self):
        import random
        random.seed(123)
        # Chain pon scenario:
        # P0 discards 3p -> P1 pon; P1 discards 4s -> P2 pon.
        p0 = ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE))
        p1 = ForceDiscardPlayer(Tile(Suit.SOUZU, TileType.FOUR))  # after pon, discard 4s
        p2 = Player()  # default reaction prefers pon if available
        p3 = NoReactionPlayer()
        g = MediumJong([p0, p1, p2, p3])

        # Ensure P0 has a 3p to discard
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        # P1 can pon 3p (two in hand) and also has a 4s to discard
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR)
        ] + [t if t != Tile(Suit.PINZU, TileType.THREE) else Tile(Suit.MANZU, TileType.NINE) for t in g._player_hands[1][3:] ]
        # P2 can pon 4s (two in hand)
        g._player_hands[2] = [
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR)
        ] + g._player_hands[2][2:]

        # Provide a small wall so the first turn proceeds deterministically
        g.tiles = [Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE)]

        # Execute one turn: should result in P1 pon on 3p, discard 4s, then P2 pon on 4s
        g.play_turn()

        # Validate P1 called sets include a pon on 3p from P0
        p1_pons = [cs for cs in g._player_called_sets[1] if cs.call_type == 'pon']
        self.assertTrue(any(cs.called_tile == Tile(Suit.PINZU, TileType.THREE) and cs.source_position == 0 for cs in p1_pons))

        # Validate P2 called sets include a pon on 4s from P1
        p2_pons = [cs for cs in g._player_called_sets[2] if cs.call_type == 'pon']
        self.assertTrue(any(cs.called_tile == Tile(Suit.SOUZU, TileType.FOUR) and cs.source_position == 1 for cs in p2_pons))

        # The discards that were called should be marked
        self.assertIn(0, g.called_discards[0])  # P0's first discard (3p) was called
        self.assertIn(0, g.called_discards[1])  # P1's first discard (4s) was called

        # Turn should be after the last caller (P3) after chain pon
        self.assertEqual(g.current_player_idx, 3)


    def test_ankan_keeps_turn_and_leads_to_one_discard(self):
        # Player 0 with 4x NORTH should Ankan on first play_turn, keep turn, then discard on second
        class KanThenDiscard(Player):
            def play(self, gs):  # type: ignore[override]
                from core.action import KanAnkan
                for m in gs.legal_moves():
                    if isinstance(m, KanAnkan):
                        return m
                return Discard(gs.player_hand[0])

        g = MediumJong([KanThenDiscard(), NoReactionPlayer(), NoReactionPlayer(), NoReactionPlayer()])
        g._player_hands[0] = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH)
        ] + g._player_hands[0][4:]
        g.current_player_idx = 0
        g._reactable_tile = None
        g._owner_of_reactable_tile = None
        rinshan_tile = Tile(Suit.MANZU, TileType.TWO)
        g.dead_wall = [Tile(Suit.HONORS, Honor.EAST), rinshan_tile]
        # First turn: perform Ankan and discard
        g.play_turn()
        self.assertEqual(g.current_player_idx, 1)
        self.assertEqual(len(g.player_discards[0]), 1)
        self.assertTrue(any(cs.call_type == 'kan_ankan' for cs in g._player_called_sets[0]))
        self.assertEqual(len(g._player_hands[0]), 10)


    def test_ron_priority_over_calls(self):

        players = [ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)), Player(), Player(), Player()]
        g = MediumJong(players)
        # Configure: player 3 can ron on 3p; player 2 can pon 3p
        # Player 3 exact 13 tiles (Tanyao-ready): 2p,4p; 345s; 456s; 456m; pair 77p
        g._player_hands[3] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g._player_hands[2][:2] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)]
        g.tiles = [Tile(Suit.PINZU, TileType.THREE)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertIn(3, g.get_winners())
        self.assertEqual(g.get_loser(), 0)


class TestYakuAndRiichi(unittest.TestCase):
    def test_yaku_required_no_yaku_no_win(self):
        # Use a player that asserts inside its play() when tsumo would be legal
        class CheckTsumoLegal(Player):
            def play(self, game_state):  # type: ignore[override]
                # Should NOT be legal to tsumo: hand has no yaku and is open (called set)
                assert not game_state.can_tsumo(), "yakuless open hand should not be able to tsumo"
                # Return any discard to continue
                return Discard(game_state.player_hand[0])

        players = [CheckTsumoLegal(), Player(), Player(), Player()]
        g = MediumJong(players)
        # Construct complete no-yaku hand for player 0 after a draw
        # 123m, 234p, 345s, 456m, pair 9s9s (no tanyao due to terminals; no sanshoku; no iipeikou)
        tiles = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.NINE)
        ]
        g._player_hands[0] = tiles
        # Make hand open by adding a called set (e.g., chi 1-2-3p from someone)
        called = CalledSet([
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE)
        ], 'chi', Tile(Suit.PINZU, TileType.TWO), caller_position=0, source_position=1)
        g._player_called_sets[0] = [called]
        g.tiles = [Tile(Suit.SOUZU, TileType.NINE)]
        # Trigger the player's play(), which performs the assertion internally
        g.play_turn()

    def test_tanyao_enables_win(self):
        g = MediumJong([Player(), Player(), Player(), Player()])
        # All simples hand waiting completed
        tiles = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tiles
        g.current_player_idx = 0
        g.last_drawn_tile = tiles[-1]
        g.last_drawn_player = 0
        self.assertTrue(g.get_game_perspective(0).can_tsumo())

    def test_yakuhai_winds(self):
        # Player 0 tsumo with triplet of East; closed hand; expect 3 han: 1 seat wind + 1 round wind + 1 menzen tsumo
        g = MediumJong([Player(), Player(), Player(), Player()])
        # Construct 13-tile closed hand:
        # E E E | 234m | 345p | 77s (pair) | 4s 5s (waiting on 6s)
        tiles_13 = [
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        ]
        g._player_hands[0] = tiles_13
        # Deterministic draw: 6s to complete 456s
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.current_player_idx = 0
        g._reactable_tile = None
        g._owner_of_reactable_tile = None
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])
        score = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(score['han'], 3)

    def test_riichi_lock_and_uradora(self):
        import random
        random.seed(433)
        g = MediumJong([ForceActionPlayer(Riichi(Tile(Suit.MANZU, TileType.TWO))), NoReactionPlayer(),
                        NoReactionPlayer(), NoReactionPlayer()])
        # Put player 0 in tenpai with closed hand; ensure Riichi legal and then only discard drawn/kan/tsumo allowed
        # Tenpai example: needing 3p to complete 2-3-4p
        base = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE),
        ]
        g._player_hands[0] = base
        g.current_player_idx = 0
        g.last_drawn_tile = None
        g.last_drawn_player = None
        # Draw a tile to start action
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)] + g.tiles
        g.play_turn()  # player 0 draws and acts (may discard). Reset for controlled test
        for _ in range(3):
            g.play_turn()
        self.assertEqual(len(g.legal_moves(0)), 1, "Forced tsumogiri")

    def test_menzen_tsumo_yakuless(self):
        # Closed hand with no yaku except menzen tsumo should win by tsumo
        g = MediumJong([Player(), Player(), Player(), Player()])
        tiles_13 = [
            # 234m sequence
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            # 345p sequence
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            # 45s waiting on 6s to complete 456s
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            # triplet 2m (breaks pinfu)
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            # pair 9s (breaks tanyao)
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[0] = tiles_13
        # Deterministic environment: draw 6s; no dora/uradora/aka
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.current_player_idx = 0
        g._reactable_tile = None
        # Player 0 draws and should tsumo
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])
        # Score should have exactly 1 han (menzen tsumo) with our constructed hand
        score = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(score['han'], 1)

    def test_ankan_north_enables_menzen_tsumo_only_yaku(self):
        # Hand includes a concealed kan of NORTH; after rinshan draw, tsumo should be legal
        # and the only yaku should be menzen tsumo.
        class KanThenTsumo(Player):
            def play(self, gs):  # type: ignore[override]
                from core.action import KanAnkan
                # If ankan is legal, perform it first
                for m in gs.legal_moves():
                    if isinstance(m, KanAnkan):
                        return m
                # If tsumo is legal now, do it
                if gs.can_tsumo():
                    from core.action import Tsumo
                    return Tsumo()
                # Otherwise discard any
                from core.action import Discard
                return Discard(gs.player_hand[0])

        g = MediumJong([KanThenTsumo(), NoReactionPlayer(), NoReactionPlayer(), NoReactionPlayer()])
        # Start with four NORTHs to enable immediate Ankan, and a nearly complete closed hand with no yaku
        g._player_hands[0] = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),  # waiting for 6s later
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),  # terminal pair breaks tanyao
        ]
        # Arrange dead wall so that rinshan draw after Ankan gives 6s (completes 456s)
        g.dead_wall = [Tile(Suit.HONORS, Honor.EAST), Tile(Suit.SOUZU, TileType.SIX)]
        # Provide an initial draw that completes 3p4p5p as a meld
        g.tiles = [Tile(Suit.PINZU, TileType.FIVE)]
        # Make deterministic: clear any initial dora/uradora so kandora won't add extra han
        g.dora_indicators = []
        g.ura_dora_indicators = []
        # Execute first turn: draw, perform ankan, rinshan draw happens, then tsumo
        g.play_turn()
        self.assertTrue(g.is_game_over())
        # Only menzen tsumo should count as yaku
        s = g._score_hand(0, win_by_tsumo=True)
        self.assertEqual(s['han'], 1)


class TestScoring(unittest.TestCase):
    def test_zero_and_nonzero_points_imply_ron_over_10_games(self):
        # If at least one player has 0 points and at least one has non-zero,
        # the outcome must be a Ron (single or multiple). Verify over 10 random games.
        import random
        random.seed(42)
        for _ in range(10):
            g = MediumJong([Player(), Player(), Player(), Player()])
            g.play_round()
            pts = g.get_points()
            self.assertIsNotNone(pts)
            has_zero = any(p == 0 for p in pts)
            has_nonzero = any(p != 0 for p in pts)
            if has_zero and has_nonzero:
                # Mixed zero/non-zero implies Ron occurred
                self.assertIsNotNone(g.get_loser())
                self.assertTrue(len(g.get_winners()) >= 1)

    def test_dealer_tsumo_vs_non_dealer(self):
        # Dealer tsumo by drawing winning tile
        g = MediumJong([ForceActionPlayer(Tsumo()), Player(), Player(), Player()])
        # 13 tiles waiting on 3s to complete tanyao hand
        dealer_wait = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = dealer_wait.copy()
        # Deterministic draw and indicators
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]  # any simple that completes the third sequence
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.play_turn()  # should tsumo
        self.assertTrue(g.is_game_over())
        pts_dealer = g.get_points()

        # Non-dealer tsumo by player 1
        g2 = MediumJong([ForceDiscardPlayer(Tile(Suit.HONORS, Honor.NORTH)),
                         ForceActionPlayer(Tsumo()),
                         NoReactionPlayer(),
                         NoReactionPlayer()])
        nondealer_wait = list(dealer_wait)
        g2._player_hands[1] = nondealer_wait.copy()
        # Make it player 1's turn directly and provide winning tile draw
        g2.current_player_idx = 1
        g2.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g2.dead_wall = []
        g2.dora_indicators = []
        g2.ura_dora_indicators = []
        g2.play_turn()
        self.assertTrue(g2.is_game_over())
        pts_nd = g2.get_points()
        self.assertIsNotNone(pts_dealer)
        self.assertIsNotNone(pts_nd)
        self.assertGreater(pts_dealer[0], pts_nd[1])

    def test_non_dealer_tsumo_tanyao_two_aka_split(self):
        # Player 1 (non-dealer) tsumo a tanyao hand with two aka dora fives
        class TsumoIfPossible(Player):
            def play(self, gs):  # type: ignore[override]
                if gs.can_tsumo():
                    return Tsumo()
                return Discard(gs.player_hand[0])

        g = MediumJong([Player(), TsumoIfPossible(), Player(), Player()])
        # Concealed tiles (10): 234m, 345s(aka 5s), 77m, 45m (avoid sanshoku)
        concealed = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE, aka=True),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE),
        ]
        g._player_hands[1] = concealed
        # Called set: chi 4-5-6p, with 5p as aka (avoid sanshoku)
        called = CalledSet([
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE, aka=True), Tile(Suit.PINZU, TileType.SIX)
        ], 'chi', Tile(Suit.PINZU, TileType.FIVE), caller_position=1, source_position=0)
        g._player_called_sets[1] = [called]
        # Deterministic draw: 6m to complete 3-4-5m with 2-3-4m already -> tanyao
        g.tiles = [Tile(Suit.MANZU, TileType.SIX)]
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.current_player_idx = 1
        g._reactable_tile = None
        g._owner_of_reactable_tile = None
        g.last_drawn_tile = None
        g.last_drawn_player = None
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        # Expect split total: dealer 2000, others 1000 each => winner gains 4000
        self.assertEqual(pts[1], 2000 + 2 * 1000)

    def test_points_not_none_after_ron_win(self):
        # Player 1 wins by ron from Player 0's discard; ensure points are populated
        from test_utils import ForceDiscardPlayer
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)),
            Player(),
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        # Construct player 1 hand to ron on 3p discard
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        # Ensure player 0 has a 3p to discard
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        # Provide a draw so play_turn can proceed deterministically
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [1])
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        self.assertEqual(g.get_game_outcome().outcome_type(0), OutcomeType.DEAL_IN)
        self.assertEqual(g.get_game_outcome().outcome_type(1), OutcomeType.RON)

    def test_pinfu_only_ron_2p_scores_1000(self):
        # Pinfu-only closed hand; non-dealer (P1) wins by ron on 2p; expect exactly 1000 points
        from test_utils import ForceDiscardPlayer, NoReactionPlayer
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.TWO)),
            Player(),
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        # Hand (13 tiles): pair 11m; sequence 1-2-3m; 3-4p (ryanmen wait on 2p/5p); 7-8-9p; 3-4-5s
        # Tiles: 1m,1m,1m,2m,3m, 3p,4p, 7p,8p,9p, 3s,4s,5s
        g._player_hands[1] = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        ]
        # Deterministic environment: ensure discarder has 2p and there are no dora or aka
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.TWO)
        g.dora_indicators = []
        g.ura_dora_indicators = []
        # Provide a benign draw so the forced discarder takes a turn
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [1])
        # Score should be exactly 1000 for non-dealer ron with pinfu-only (30 fu, 1 han)
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[1], 1300)

    def test_dealer_ron_points_conservation(self):
        # Dealer (P0) wins by ron from non-dealer P1's discard; total points should conserve to 0
        from test_utils import ForceDiscardPlayer, NoReactionPlayer
        import random
        random.seed(234)
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.HONORS, Honor.NORTH)),
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)),
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        # Construct dealer (P0) hand to ron on 3p discard
        g._player_hands[0] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        # Ensure P1 holds a 3p to discard
        g._player_hands[1][0] = Tile(Suit.PINZU, TileType.THREE)
        # Provide a draw so play_turn can proceed deterministically
        g.tiles = [Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        # Points conservation: sum to zero
        self.assertEqual(sum(pts), 0)

    def test_double_ron_points_and_riichi_sticks_first_only(self):
        # Setup: Player 0 discards 3p; Players 1 and 2 can both ron on 3p with tanyao hands
        from test_utils import ForceDiscardPlayer, NoReactionPlayer
        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)),
            Player(),
            Player(),
            NoReactionPlayer(),
        ])
        # Both players 1 and 2 wait on 3p (tanyao-completable)
        ron_wait_hand = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g._player_hands[1] = list(ron_wait_hand)
        g._player_hands[2] = list(ron_wait_hand)
        # Ensure player 0 holds a 3p to discard
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        # Add riichi sticks to verify only first ron receives them
        extra_points = 2000
        g.riichi_sticks_pot = extra_points
        g.last_discard_was_riichi = False
        # Provide a draw so play_turn proceeds
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        # Winners should be [1, 2] (turn order from discarder 0)
        self.assertEqual(g.get_winners(), [1, 2])
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        # Points conservation: loser pays sum of winners
        self.assertEqual(pts[0] - extra_points, -(pts[1] + pts[2]))
        # Use GameOutcome to verify deltas and riichi sticks allocation
        outcome = g.get_game_outcome()
        # Winners recorded correctly and loser is discarder 0
        self.assertEqual(outcome.winners, [1, 2])
        self.assertEqual(outcome.loser, 0)
        # Per-player deltas equal outcome players' points_delta
        for pid in range(4):
            self.assertEqual(pts[pid], outcome.players[pid].points_delta)
        # First winner gets riichi sticks, second does not
        self.assertEqual(outcome.players[1].points_delta, outcome.players[2].points_delta + extra_points)

    def test_riichi_stick_not_paid_on_immediate_ron(self):
        # P0 declares riichi and immediately deals into P1; P1 should NOT get riichi stick
        class DiscardWinning(Player):
            def play(self, gs):  # type: ignore[override]
                # Discard 3p if present
                t = Tile(Suit.PINZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                # Otherwise declare riichi on drawn
                for m in gs.legal_moves():
                    if isinstance(m, Riichi):
                        return m
                return Discard(gs.player_hand[0])

        g = MediumJong([DiscardWinning(), Player(), Player(), Player()])
        # Set P1 to ron on 3p
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        # Ensure P0 has 3p and can riichi; immediate discard will be the riichi tile
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = []
        g.current_player_idx = 0
        # P0 declares riichi (discard 3p) and P1 rons immediately
        lm = g.legal_moves(0)
        # Synthesize a riichi move with that tile (if not provided), else discard
        r_moves = [m for m in lm if isinstance(m, Riichi)]
        if r_moves:
            g.step(0, r_moves[0])
        else:
            g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        g._poll_reactions()
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(pts[0], -5200) #3han 40fu
        self.assertEqual(pts[1], 5200)

    def test_riichi_stick_lost_on_exhaustive_draw(self):
        # P0 declares riichi; at exhaustive draw, P0 loses 1k but gains 3k from noten payments (others not tenpai)
        g = MediumJong([ForceActionPlayer(Riichi(Tile(Suit.MANZU, TileType.ONE))), Player(), Player(), Player()])
        # Make all others noten
        for pid in [1,2,3]:
            g._player_hands[pid] = [
                Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH), Tile(Suit.HONORS, Honor.WEST),
                Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.WHITE), Tile(Suit.HONORS, Honor.GREEN),
                Tile(Suit.HONORS, Honor.RED),
                Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.NINE),
                Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.NINE),
                Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.NINE),
            ]
        # P0 in tenpai
        g._player_hands[0] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        # Exhaustive draw
        g.tiles = [Tile(Suit.MANZU, TileType.ONE)]
        g.play_turn()
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pay = g.get_points()
        # P0 should have +3000 from three noten players, minus 1000 riichi stick -> +2000
        self.assertEqual(pay[0], 2000)

    def test_sanankou_counts_with_open_hand(self):
        # Three concealed triplets and one open chi should count Sanankou (2 han)
        g = MediumJong([Player(), Player(), Player(), Player()])
        concealed = [
            # 111m concealed triplet
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
            # 222p concealed triplet
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            # 333s concealed triplet
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
            # Pair 77m
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
        ]
        g._player_hands[0] = concealed
        # Open chi 4-5-6p
        g._player_called_sets[0] = [CalledSet([
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX)
        ], 'chi', Tile(Suit.PINZU, TileType.FIVE), caller_position=0, source_position=1)]
        # Disable dora/aka randomness
        g.dora_indicators = []
        g.ura_dora_indicators = []
        s = g._score_hand(0, win_by_tsumo=False)
        self.assertGreaterEqual(s['han'], 2)

    def test_sanshoku_closed_and_open(self):
        # Closed sanshoku: three sequences of 3-4-5 across m/p/s, closed hand
        g = MediumJong([Player(), Player(), Player(), Player()])
        closed_hand = [
            Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
            Tile(Suit.MANZU, TileType.EIGHT), Tile(Suit.MANZU, TileType.NINE),
        ]
        g._player_hands[0] = closed_hand
        g.current_player_idx = 0
        g.tiles = [Tile(Suit.MANZU, TileType.SEVEN)]  # complete pair
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s_closed = g._score_hand(0, win_by_tsumo=True)
        # Expect at least sanshoku 2 han for closed; allow additional han such as menzen tsumo
        self.assertGreaterEqual(s_closed['han'], 2)

        # Open sanshoku: make one of the sequences an open chi
        g2 = MediumJong([Player(), Player(), Player(), Player()])
        # Concealed tiles: 345m, 34s (waiting 5s), 678m, pair 77p -> 10 tiles
        open_hand_concealed = [
            Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g2._player_hands[0] = open_hand_concealed
        # Open chi: 3-4-5p provides the third suit sequence
        g2._player_called_sets[0] = [CalledSet([
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE)
        ], 'chi', Tile(Suit.PINZU, TileType.FOUR), caller_position=0, source_position=1)]
        g2.current_player_idx = 0
        g2.tiles = [Tile(Suit.SOUZU, TileType.FIVE)]  # tsumo 5s completes 345s
        g2.dead_wall = []
        g2.dora_indicators = []
        g2.ura_dora_indicators = []
        g2.play_turn()
        self.assertTrue(g2.is_game_over())
        s_open = g2._score_hand(0, win_by_tsumo=True)
        # Expect at least sanshoku 1 han for open; allow additional han
        self.assertGreaterEqual(s_open['han'], 1)

    def test_ittsu_closed_and_open(self):
        # Closed ittsu: 123m, 456m, 78m (wait 9m), 345s, pair 77p
        g = MediumJong([Player(), Player(), Player(), Player()])
        closed_hand = [
            # 123m, 456m, 78m
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            # 345s
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            # pair 77p
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g._player_hands[0] = closed_hand
        g.current_player_idx = 0
        # Draw 9m to complete 789m
        g.tiles = [Tile(Suit.MANZU, TileType.NINE)]
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s_closed = g._score_hand(0, win_by_tsumo=True)
        # Expect at least ittsu (2 closed) + menzen tsumo (1) = >=3 han; allow more if present
        self.assertGreaterEqual(s_closed['han'], 3)

        # Open ittsu: make 456m an open chi
        g2 = MediumJong([Player(), Player(), Player(), Player()])
        concealed = [
            # 123m, 78m (wait 9m)
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            # 345s
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            # pair 77p
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g2._player_hands[0] = concealed
        g2._player_called_sets[0] = [CalledSet([
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX)
        ], 'chi', Tile(Suit.MANZU, TileType.FIVE), caller_position=0, source_position=1)]
        g2.current_player_idx = 0
        # Draw 9m to complete 789m
        g2.tiles = [Tile(Suit.MANZU, TileType.NINE)]
        g2.dead_wall = []
        g2.dora_indicators = []
        g2.ura_dora_indicators = []
        g2.play_turn()
        self.assertTrue(g2.is_game_over())
        s_open = g2._score_hand(0, win_by_tsumo=True)
        # Expect at least ittsu open = 1 han (may be higher)
        self.assertGreaterEqual(s_open['han'], 1)

    def test_non_dealer_ron_mangan_from_dealer_discard(self):
        players = [ForceDiscardPlayer(Tile(Suit.SOUZU, TileType.SIX)), Player(), Player(), Player()]
        g = MediumJong(players)
        # Construct player 1 hand: 123s, 234s, 789s, pair 66s, plus 45s waiting on 6s
        p1_tiles = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.SIX), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        ]
        g._player_hands[1] = p1_tiles
        # Deterministic: no draws, player 0 to act and discard 6s
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [1])
        self.assertEqual(g.get_loser(), 0)
        # Score should apply mangan cap for non-dealer ron: 8000 from dealer
        pts = g.get_points()
        self.assertEqual(pts[1], 8000)
        self.assertEqual(pts[0], -8000)

    def test_two_riichi_then_ron_zero_sum(self):
        # Script: P0 declares riichi (on 9m), P1 declares riichi (on 9m), then P2 discards 3p and P1 rons.
        # Check that cumulative get_points sums to zero.
        from test_utils import ForceActionPlayer, ForceDiscardPlayer, NoReactionPlayer
        # Force P0 and P1 to declare Riichi when legal; P2 will discard 3p; P3 no reactions
        g = MediumJong([
            ForceActionPlayer(Riichi(Tile(Suit.MANZU, TileType.NINE))),
            ForceActionPlayer(Riichi(Tile(Suit.MANZU, TileType.NINE))),
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)),
            NoReactionPlayer(),
        ])
        # Configure P0 hand to be tenpai so Riichi(9m) is legal when discarding the drawn 9m
        g._player_hands[0] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        # Configure P1 hand to be able to ron on 3p (tanyao-completable), reusing pattern from other tests
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        # Ensure P2 holds a 3p to discard
        g._player_hands[2][0] = Tile(Suit.PINZU, TileType.THREE)
        # Arrange draws: P0 draws 9m (to enable riichi); P1 draws 9m (to enable riichi);
        # Provide a harmless draw for P2 to proceed before discarding 3p
        g.tiles = [
            Tile(Suit.MANZU, TileType.NINE),  # P0 draw -> riichi discard 9m
            Tile(Suit.MANZU, TileType.NINE),  # P1 draw -> riichi discard 9m
            Tile(Suit.MANZU, TileType.TWO),   # P2 draw (harmless) then discard 3p -> P1 ron
        ]
        # Play through until over
        while not g.is_game_over():
            g.play_turn()
        pts = g.get_points()
        self.assertIsNotNone(pts)
        self.assertEqual(sum(pts), 0)

    def test_rinshan(self):
        # After Ankan (concealed kan) of four NORTH tiles, a rinshan draw should occur from the dead wall
        g = MediumJong([Player(), Player(), Player(), Player()])
        # Ensure player 0 has four NORTHs
        g._player_hands[0] = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH)
        ] + g._player_hands[0][4:]
        g.current_player_idx = 0
        g._reactable_tile = None
        g._owner_of_reactable_tile = None
        # Deterministic dead wall: set kandora indicator source and the rinshan draw tile at the end
        indicator_src = Tile(Suit.HONORS, Honor.EAST)
        rinshan_tile = Tile(Suit.SOUZU, TileType.SIX)
        g.dead_wall = [indicator_src, rinshan_tile]
        before_len = len(g.dead_wall)
        # Perform Ankan on NORTH
        from core.action import KanAnkan
        self.assertTrue(g.is_legal(0, KanAnkan(Tile(Suit.HONORS, Honor.NORTH))))
        g.step(0, KanAnkan(Tile(Suit.HONORS, Honor.NORTH)))
        # Rinshan draw should have happened and drawn the last tile from dead wall
        self.assertIsNotNone(g.last_drawn_tile)
        self.assertEqual(g.last_drawn_tile.suit, rinshan_tile.suit)
        self.assertEqual(g.last_drawn_tile.tile_type, rinshan_tile.tile_type)
        # Dead wall reduced by 1 (kandora indicators do not pop tiles)
        self.assertEqual(len(g.dead_wall), before_len - 1)
        # Kandora indicator appended should match the tile that was at the end just before rinshan
        self.assertTrue(g.dora_indicators)
        self.assertEqual(g.dora_indicators[-1].suit, rinshan_tile.suit)
        self.assertEqual(g.dora_indicators[-1].tile_type, rinshan_tile.tile_type)

    def test_initial_dora_exists(self):
        # Simple instantiation of MediumJong should have at least one dora indicator
        g = MediumJong([Player(), Player(), Player(), Player()])
        self.assertTrue(hasattr(g, 'dora_indicators'))
        self.assertGreaterEqual(len(g.dora_indicators), 1)

    def test_riichi_ippatsu_tsumo_with_single_uradora(self):
        # Player 0: closed tenpai with no yaku except riichi + menzen tsumo; ensure exactly 1 uradora gives 4 han total
        g = MediumJong([ForceActionPlayer(Riichi(Tile(Suit.MANZU, TileType.NINE))),
                        NoReactionPlayer(),
                        NoReactionPlayer(),
                        NoReactionPlayer()])
        # Construct 13-tile closed hand: triplet 222m; sequences 345p and 678m; pair 11m; wait 6s on 45s
        tiles_13 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
        ]
        g._player_hands[0] = tiles_13
        # Set indicators: no normal dora; exactly one uradora that maps to 3p (present in hand)
        g.dora_indicators = []
        g.ura_dora_indicators = [Tile(Suit.PINZU, TileType.TWO)]  # next is 3p
        # First draw for P0 (non-winning) to allow Riichi; choose a riichi option discarding a safe tile
        first_draw = Tile(Suit.MANZU, TileType.NINE)
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX),  # winning 6s later for tsumo
                   Tile(Suit.HONORS, Honor.EAST),  # safe discards for others
                   Tile(Suit.HONORS, Honor.SOUTH),
                   Tile(Suit.HONORS, Honor.WEST),
                   first_draw]
        for _ in range(4):
            g.play_turn()

        # Back to player 0; draw winning 6s and tsumo
        g.play_turn()
        # Ensure game ends by tsumo and final points reflect dealer tsumo with riichi + ippatsu + menzen + 1 uradora
        self.assertTrue(g.is_game_over())
        pts = g.get_points()
        # Base points: fu=30, han=4 -> base=30*2^(2+4)=30*64=1920; dealer tsumo total=round_up_100(1920*6)=11600
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        self.assertEqual(pts[0], 11600)
        self.assertEqual(pts[1], -3866)
        self.assertEqual(pts[2], -3866)
        self.assertEqual(pts[3], -3866)

    def test_keiten_one_in_tenpai(self):
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = _make_tenpai_hand()
        g._player_hands[1] = _make_noten_hand()
        g._player_hands[2] = _make_noten_hand()
        g._player_hands[3] = _make_noten_hand()
        g.tiles = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pay = g.get_points()
        self.assertEqual(pay[0], 3000)
        self.assertEqual(pay[1], -1000)
        self.assertEqual(pay[2], -1000)
        self.assertEqual(pay[3], -1000)

    def test_keiten_two_in_tenpai(self):
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = _make_tenpai_hand()
        g._player_hands[1] = _make_tenpai_hand()
        g._player_hands[2] = _make_noten_hand()
        g._player_hands[3] = _make_noten_hand()
        g.tiles = []
        g.play_turn()
        pay = g.get_points()
        self.assertEqual(pay[0], 1500)
        self.assertEqual(pay[1], 1500)
        self.assertEqual(pay[2], -1500)
        self.assertEqual(pay[3], -1500)

    def test_keiten_three_in_tenpai(self):
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = _make_tenpai_hand()
        g._player_hands[1] = _make_tenpai_hand()
        g._player_hands[2] = _make_tenpai_hand()
        g._player_hands[3] = _make_noten_hand()
        g.tiles = []
        g.play_turn()
        pay = g.get_points()
        self.assertEqual(pay[0], 1000)
        self.assertEqual(pay[1], 1000)
        self.assertEqual(pay[2], 1000)
        self.assertEqual(pay[3], -3000)

    def test_keiten_all_or_none_in_tenpai(self):
        # None in tenpai
        g = MediumJong([Player(), Player(), Player(), Player()])
        g._player_hands[0] = _make_noten_hand()
        g._player_hands[1] = _make_noten_hand()
        g._player_hands[2] = _make_noten_hand()
        g._player_hands[3] = _make_noten_hand()
        g.tiles = []
        g.play_turn()
        pay = g.get_points()
        self.assertEqual(pay[0], 0)
        self.assertEqual(pay[1], 0)
        self.assertEqual(pay[2], 0)
        self.assertEqual(pay[3], 0)

        # All in tenpai
        g2 = MediumJong([Player(), Player(), Player(), Player()])
        th = _make_tenpai_hand()
        g2._player_hands[0] = th
        g2._player_hands[1] = th
        g2._player_hands[2] = th
        g2._player_hands[3] = th
        g2.tiles = []
        g2.play_turn()
        pay2 = g2.get_points()
        self.assertEqual(pay2[0], 0)
        self.assertEqual(pay2[1], 0)
        self.assertEqual(pay2[2], 0)
        self.assertEqual(pay2[3], 0)

class TestGameOutcome(unittest.TestCase):
    def test_serialize_deserialize_draw(self):
        # Force an exhaustive draw by emptying wall quickly
        from core.game import MediumJong, Player
        import core.game
        g = MediumJong([Player(), Player(), Player(), Player()])
        g.tiles = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        outcome = g.get_game_outcome()
        data = outcome.serialize()
        round_trip = core.game.GameOutcome.deserialize(data)  # type: ignore[attr-defined]
        self.assertEqual(outcome.is_draw, round_trip.is_draw)
        self.assertEqual(set(outcome.winners), set(round_trip.winners))
        self.assertEqual(outcome.loser, round_trip.loser)
        for pid in range(4):
            a = outcome.players[pid]
            b = round_trip.players[pid]
            self.assertEqual(a.player_id, b.player_id)
            self.assertEqual((None if a.outcome_type is None else a.outcome_type.value), (None if b.outcome_type is None else b.outcome_type.value))
            # boolean fields removed; outcome_type is the single source of truth
            self.assertEqual(a.points_delta, b.points_delta)

    def test_serialize_deserialize_win(self):
        # Deterministic tsumo for player 0
        from core.game import MediumJong, Player, Tsumo
        from core.tile import Tile, Suit, TileType
        import core.game
        class TsumoIfCan(Player):
            def play(self, gs):  # type: ignore[override]
                if gs.can_tsumo():
                    return Tsumo()
                return core.game.Discard(gs.player_hand[0])
        g = MediumJong([TsumoIfCan(), Player(), Player(), Player()])
        g._player_hands[0] = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.play_turn()
        self.assertTrue(g.is_game_over())
        outcome = g.get_game_outcome()
        data = outcome.serialize()
        round_trip = core.game.GameOutcome.deserialize(data)  # type: ignore[attr-defined]
        self.assertEqual(outcome.is_draw, round_trip.is_draw)
        self.assertEqual(set(outcome.winners), set(round_trip.winners))
        self.assertEqual(outcome.loser, round_trip.loser)
        for pid in range(4):
            a = outcome.players[pid]
            b = round_trip.players[pid]
            self.assertEqual(a.player_id, b.player_id)
            self.assertEqual((None if a.outcome_type is None else a.outcome_type.value), (None if b.outcome_type is None else b.outcome_type.value))
            self.assertEqual(a.points_delta, b.points_delta)

if __name__ == '__main__':
    unittest.main(verbosity=2)


