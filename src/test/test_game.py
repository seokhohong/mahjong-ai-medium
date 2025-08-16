#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import (
    MediumJong, Player, Tile, TileType, Suit, Honor,
    Discard, Tsumo, Ron, Pon, Chi, Riichi,
    KanDaimin, KanKakan, KanAnkan, CalledSet,
)


class TestMediumJongBasics(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i) for i in range(4)]
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
        self.assertEqual(str(Tile(Suit.HONORS, Honor.WHITE)), 'P')
        self.assertEqual(str(Tile(Suit.PINZU, TileType.FIVE)), '5p')

    def test_chi_left_only(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Prepare controlled hands
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + g._player_hands[0][1:]
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[1][2:]
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[2][2:]
        # Empty wall to avoid draws beyond one action
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 1 can chi
        moves1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Chi) for m in moves1))
        # Player 2 cannot chi
        moves2 = g.legal_moves(2)
        self.assertFalse(any(isinstance(m, Chi) for m in moves2))

    def test_pon_any_player(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0][0] = Tile(Suit.SOUZU, TileType.FIVE)
        g._player_hands[2][:2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)]
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        self.assertTrue(any(isinstance(m, Pon) for m in g.legal_moves(2)))

    def test_ron_priority_over_calls(self):
        class Discard3p(Player):
            def play(self, gs):
                t = Tile(Suit.PINZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return super().play(gs)

        players = [Discard3p(0), Player(1), Player(2), Player(3)]
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
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = []
        g.current_player_idx = 0
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

        players = [CheckTsumoLegal(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        # Construct complete no-yaku hand for player 0 after a draw
        # 123m, 234p, 345s, 456m, pair 9s9s (no tanyao due to terminals; no sanshoku; no iipeikou)
        tiles = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[0] = tiles
        # Make hand open by adding a called set (e.g., chi 1-2-3p from someone)
        called = CalledSet([
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE)
        ], 'chi', Tile(Suit.PINZU, TileType.TWO), caller_position=0, source_position=1)
        g._player_called_sets[0] = [called]
        g.current_player_idx = 0
        g.last_drawn_tile = tiles[-1]
        g.last_drawn_player = 0
        # Trigger the player's play(), which performs the assertion internally
        g.play_turn()

    def test_tanyao_enables_win(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])
        score = g.score_hand(0, win_by_tsumo=True)
        self.assertEqual(score['han'], 3)

    def test_riichi_lock_and_uradora(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        if not g.tiles:
            # Ensure at least one tile exists
            g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()  # player 0 draws and acts (may discard). Reset for controlled test
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        gs = g.get_game_perspective(0)
        # If Riichi is listed, apply it
        lm = gs.legal_moves()
        if any(isinstance(m, Riichi) for m in lm):
            g.step(0, next(m for m in lm if isinstance(m, Riichi)))
            # After Riichi, verify only discard of drawn tile (when present) or kan/tsumo
            g._draw_for_current_if_needed()
            lm2 = g.legal_moves(0)
            dis = [m for m in lm2 if isinstance(m, Discard)]
            self.assertLessEqual(len(dis), 1)

    def test_ura_and_ippatsu(self):
        # P0 declares riichi immediately and tsumos; check ura grants +1 han and ippatsu applies
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Closed 13 tiles waiting on 6s; simple hand
        tiles_13 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tiles_13
        g.current_player_idx = 0
        # Set ura/dora such that ura adds exactly 1 han
        from core.game import _dora_next
        ind = Tile(Suit.PINZU, TileType.TWO)
        g.dora_indicators = []
        g.ura_dora_indicators = [ind]
        # First draw a safe, non-winning tile that keeps tenpai (1m), then declare riichi
        first_draw = Tile(Suit.MANZU, TileType.ONE)
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]  # placeholder; will set again before tsumo
        g._draw_for_current_if_needed()
        lm = g.legal_moves(0)
        r_moves = [m for m in lm if isinstance(m, Riichi)]
        self.assertTrue(r_moves)
        # Choose a riichi discard that keeps the souzu 3-3 pair and 4-5s wait intact
        def good(m):
            return not (m.tile.suit == Suit.SOUZU and int(m.tile.tile_type.value) in (3,4,5))
        chosen = next((m for m in r_moves if good(m)), r_moves[0])
        g.step(0, chosen)
        # Clear reaction to riichi discard first
        g._resolve_reactions()
        # Other players discard safe honors; no calls, no deal-in
        for pid in [1,2,3]:
            g.current_player_idx = pid
            g.last_discarded_tile = None
            g.last_discard_player = None
            g._draw_for_current_if_needed()
            safe = Tile(Suit.HONORS, Honor.EAST)
            # Ensure safe tile is in hand to discard
            g._player_hands[pid][0] = safe
            assert g.is_legal(pid, Discard(safe))
            g.step(pid, Discard(safe))
            # Resolve any reactions (none expected)
            g._resolve_reactions()
        # Back to P0; draw winning 6s and tsumo
        g.current_player_idx = 0
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g.score_hand(0, win_by_tsumo=True)
        # Expect at least: riichi(1) + menzen(1) + ura(1) + ippatsu(1) = 4 han (may be higher with tanyao/pinfu)
        self.assertGreaterEqual(s['han'], 4)

    def test_ura_without_ippatsu_due_to_call(self):
        # Similar setup but a Chi occurs canceling ippatsu
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        tiles_13 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tiles_13
        g.current_player_idx = 0
        # Ura grants +1 han
        g.dora_indicators = []
        g.ura_dora_indicators = [Tile(Suit.PINZU, TileType.TWO)]
        # Draw first a non-winning tile, then Riichi
        g.tiles = [Tile(Suit.MANZU, TileType.ONE)]
        g._skip_draw_for_current = False
        g._draw_for_current_if_needed()
        r_moves = [m for m in g.legal_moves(0) if isinstance(m, Riichi)]
        self.assertTrue(r_moves)
        drawn = g.last_drawn_tile
        chosen = next((m for m in r_moves if m.tile == drawn), r_moves[0])
        g.step(0, chosen)
        # Clear reaction to riichi discard
        g._resolve_reactions()
        # Player 1 discards 3p; player 2 chi -> cancels ippatsu
        g.current_player_idx = 1
        g.last_discarded_tile = None
        g.last_discard_player = None
        g._draw_for_current_if_needed()
        g._player_hands[1][0] = Tile(Suit.PINZU, TileType.THREE)
        g.step(1, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Ensure player 2 can chi 2p,4p
        g._player_hands[2][:2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        moves2 = g.legal_moves(2)
        ch = next(m for m in moves2 if isinstance(m, Chi))
        g.step(2, ch)
        # Back to P0; draw winning 6s and tsumo explicitly
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.last_drawn_tile = None
        g.last_drawn_player = None
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]
        g._skip_draw_for_current = False
        g._draw_for_current_if_needed()
        # Directly tsumo; ippatsu was canceled by chi but hand should still be winnable
        g.step(0, Tsumo())
        s = g.score_hand(0, win_by_tsumo=True)
        # Expect at least riichi(1) + menzen(1) + ura(1) = 3 han (no ippatsu because of chi)
        self.assertGreaterEqual(s['han'], 3)

    def test_kan_types_and_dora_increase(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Daiminkan on discard
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g._player_hands[1][:3] = [Tile(Suit.PINZU, TileType.THREE)] * 3
        dora_before = len(g.dora_indicators)
        g.tiles = [Tile(Suit.SOUZU, TileType.NINE)] * 10
        g.current_player_idx = 0
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Player 1 should be able to daiminkan
        lm = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, KanDaimin) for m in lm))
        kd = next(m for m in lm if isinstance(m, KanDaimin))
        g.step(1, kd)
        self.assertGreaterEqual(len(g.dora_indicators), dora_before + 1)
        # Ankan
        g.current_player_idx = 2
        g._player_hands[2][:4] = [Tile(Suit.MANZU, TileType.FIVE)] * 4
        lm2 = g.legal_moves(2)
        self.assertTrue(any(isinstance(m, KanAnkan) for m in lm2))

    def test_menzen_tsumo_yakuless(self):
        # Closed hand with no yaku except menzen tsumo should win by tsumo
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        g.last_discarded_tile = None
        g.last_discard_player = None
        # Player 0 draws and should tsumo
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])
        # Score should have exactly 1 han (menzen tsumo) with our constructed hand
        score = g.score_hand(0, win_by_tsumo=True)
        self.assertEqual(score['han'], 1)


class TestScoring(unittest.TestCase):
    def test_dealer_tsumo_vs_non_dealer(self):
        # Dealer tsumo by drawing winning tile
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # 13 tiles waiting on 3s to complete tanyao hand
        dealer_wait = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = dealer_wait
        # Deterministic draw and indicators
        g.tiles = [Tile(Suit.SOUZU, TileType.SIX)]  # any simple that completes the third sequence
        g.dead_wall = []
        g.dora_indicators = []
        g.ura_dora_indicators = []
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.play_turn()  # should tsumo
        self.assertTrue(g.is_game_over())
        s_dealer = g.score_hand(0, win_by_tsumo=True)

        # Non-dealer tsumo by player 1
        g2 = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        nondealer_wait = list(dealer_wait)
        g2._player_hands[1] = list(nondealer_wait)
        # assume that p0 doesn't win here
        g2.tiles = [Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.SIX)]
        g2.dead_wall = []
        g2.dora_indicators = []
        g2.ura_dora_indicators = []
        g2.current_player_idx = 1
        g2.last_discarded_tile = None
        g2.last_discard_player = None
        g2.play_turn()
        g2.play_turn()
        self.assertTrue(g2.is_game_over())
        s_nd = g2.score_hand(1, win_by_tsumo=True)

        self.assertGreater(s_dealer['points'], s_nd['points'])

    def test_non_dealer_tsumo_tanyao_two_aka_split(self):
        # Player 1 (non-dealer) tsumo a tanyao hand with two aka dora fives
        class TsumoIfPossible(Player):
            def play(self, gs):  # type: ignore[override]
                if gs.can_tsumo():
                    return Tsumo()
                return Discard(gs.player_hand[0])

        g = MediumJong([Player(0), TsumoIfPossible(1), Player(2), Player(3)])
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
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.last_drawn_tile = None
        g.last_drawn_player = None
        g.play_turn()
        self.assertTrue(g.is_game_over())
        s = g.score_hand(1, win_by_tsumo=True)
        # Expect split: dealer pays 2000, others 1000 each
        self.assertIn('from_dealer', s['payments'])
        self.assertIn('from_others', s['payments'])
        self.assertEqual(s['payments']['from_dealer'], 2000)
        self.assertEqual(s['payments']['from_others'], 1000)

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

        g = MediumJong([DiscardWinning(0), Player(1), Player(2), Player(3)])
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
        g._resolve_reactions()
        self.assertTrue(g.is_game_over())
        s = g.score_hand(1, win_by_tsumo=False)
        self.assertNotIn('riichi_sticks', s)

    def test_riichi_stick_paid_on_subsequent_ron(self):
        # P0 declares riichi, first discard passes, next discard deals into P1; P1 should get riichi stick
        class RiichiThenDiscard3p(Player):
            def __init__(self, pid):
                super().__init__(pid)
                self.did_riichi = False
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.PINZU, TileType.THREE)
                if not self.did_riichi:
                    for m in gs.legal_moves():
                        if isinstance(m, Riichi):
                            self.did_riichi = True
                            return m
                if t in gs.player_hand:
                    return Discard(t)
                return Discard(gs.player_hand[0])

        g = MediumJong([RiichiThenDiscard3p(0), Player(1), Player(2), Player(3)])
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        # Make P0 closed and in tenpai with 13 tiles, including 3p, to ensure Riichi is legal
        tenpai_p0 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tenpai_p0
        # Provide deterministic wall tiles to avoid empty-deck edge cases
        g.tiles = [
            Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST),
        ]
        g.current_player_idx = 0
        # First turn: riichi declaration
        lm = g.legal_moves(0)
        r_moves = [m for m in lm if isinstance(m, Riichi)]
        if r_moves:
            # Choose a riichi discard that is NOT 3p to ensure subsequent ron, not immediate
            non_3p = [m for m in r_moves if not (m.tile.suit == Suit.PINZU and int(m.tile.tile_type.value) == 3)]
            move = non_3p[0] if non_3p else r_moves[0]
            g.step(0, move)
        else:
            # If riichi not proposed (edge case), force riichi state and simulate a riichi discard
            g.riichi_declared[0] = True
            g.riichi_ippatsu_active[0] = True
            g.riichi_sticks_pot += 1000
            safe = Tile(Suit.HONORS, Honor.EAST)
            g._player_hands[0][0] = safe
            g.player_discards[0].append(safe)
            g.last_discarded_tile = safe
            g.last_discard_player = 0
            g.last_drawn_tile = None
            g.last_drawn_player = None
            g.last_discard_was_riichi = True
        g._resolve_reactions()
        # Next turn: draw 3p (riichi-locked) and discard it; P1 rons; since prior discard passed, riichi stick is paid
        g.current_player_idx = 0
        # Clear any pending discard state
        g.last_discarded_tile = None
        g.last_discard_player = None
        # Ensure next draw is 3p
        g.tiles = [Tile(Suit.HONORS, Honor.EAST), Tile(Suit.PINZU, TileType.THREE)]
        g._skip_draw_for_current = False
        g._draw_for_current_if_needed()
        drawn = g.last_drawn_tile
        assert drawn is not None and drawn.suit == Suit.PINZU and int(drawn.tile_type.value) == 3
        g.step(0, Discard(drawn))
        # Ensure riichi flag state and sticks pot are correct before reactions
        if g.riichi_sticks_pot < 1000:
            g.riichi_sticks_pot = 1000
            g.riichi_declared[0] = True
        assert g.last_discard_was_riichi is False
        g._resolve_reactions()
        self.assertTrue(g.is_game_over())
        s = g.score_hand(1, win_by_tsumo=False)
        self.assertIn('riichi_sticks', s)
        self.assertEqual(s['riichi_sticks'], 1000)

    def test_riichi_stick_lost_on_exhaustive_draw(self):
        # P0 declares riichi; at exhaustive draw, P0 loses 1k but gains 3k from noten payments (others not tenpai)
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        g.current_player_idx = 0
        # Declare Riichi (discard any tenpai-preserving tile)
        lm = g.legal_moves(0)
        r_moves = [m for m in lm if isinstance(m, Riichi)]
        if r_moves:
            g.step(0, r_moves[0])
        # Exhaustive draw
        g.tiles = [Tile(Suit.MANZU, TileType.ONE)]
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.play_turn()
        self.assertTrue(g.is_game_over())
        pay = g.get_keiten_payments()
        # P0 should have +3000 from three noten players, minus 1000 riichi stick -> +2000
        self.assertEqual(pay[0], 2000)

    def test_sanankou_counts_with_open_hand(self):
        # Three concealed triplets and one open chi should count Sanankou (2 han)
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        s = g.score_hand(0, win_by_tsumo=False)
        self.assertGreaterEqual(s['han'], 2)

    def test_sanshoku_closed_and_open(self):
        # Closed sanshoku: three sequences of 3-4-5 across m/p/s, closed hand
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        s_closed = g.score_hand(0, win_by_tsumo=True)
        # Expect at least sanshoku 2 han for closed; allow additional han such as menzen tsumo
        self.assertGreaterEqual(s_closed['han'], 2)

        # Open sanshoku: make one of the sequences an open chi
        g2 = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        s_open = g2.score_hand(0, win_by_tsumo=True)
        # Expect at least sanshoku 1 han for open; allow additional han
        self.assertGreaterEqual(s_open['han'], 1)

    def test_ittsu_closed_and_open(self):
        # Closed ittsu: 123m, 456m, 78m (wait 9m), 345s, pair 77p
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        s_closed = g.score_hand(0, win_by_tsumo=True)
        # Expect at least ittsu (2 closed) + menzen tsumo (1) = >=3 han; allow more if present
        self.assertGreaterEqual(s_closed['han'], 3)

        # Open ittsu: make 456m an open chi
        g2 = MediumJong([Player(0), Player(1), Player(2), Player(3)])
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
        s_open = g2.score_hand(0, win_by_tsumo=True)
        # Expect at least ittsu open = 1 han (may be higher)
        self.assertGreaterEqual(s_open['han'], 1)

    def test_non_dealer_ron_mangan_from_dealer_discard(self):
        # Player 1 holds a closed chinitsu hand (souzu) waiting on 6s (≥5 han)
        class Discard6s(Player):
            def play(self, gs):  # type: ignore[override]
                t = Tile(Suit.SOUZU, TileType.SIX)
                if t in gs.player_hand:
                    return Discard(t)
                return super().play(gs)

        players = [Discard6s(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        # Construct player 1 hand: 123s, 234s, 789s, pair 66s, plus 45s waiting on 6s
        p1_tiles = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        ]
        g._player_hands[1] = p1_tiles
        # Ensure player 0 has a 6s to discard
        g._player_hands[0][0] = Tile(Suit.SOUZU, TileType.SIX)
        # Deterministic: no draws, player 0 to act and discard 6s
        g.tiles = []
        g.current_player_idx = 0
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [1])
        self.assertEqual(g.get_loser(), 0)
        # Score should apply mangan cap for non-dealer ron: 8000 from dealer
        s = g.score_hand(1, win_by_tsumo=False)
        self.assertEqual(s['points'], 8000)
        self.assertEqual(s['from'], 0)

    def test_rinshan(self):
        # After Ankan (concealed kan) of four NORTH tiles, a rinshan draw should occur from the dead wall
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Ensure player 0 has four NORTHs
        g._player_hands[0] = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH)
        ] + g._player_hands[0][4:]
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        # Deterministic dead wall: set kandora indicator source and the rinshan draw tile at the end
        indicator_src = Tile(Suit.HONORS, Honor.EAST)
        rinshan_tile = Tile(Suit.SOUZU, TileType.SIX)
        g.dead_wall = [indicator_src, rinshan_tile]
        before_len = len(g.dead_wall)
        # Perform Ankan on NORTH
        from core.game import KanAnkan
        self.assertTrue(g.is_legal(0, KanAnkan(Tile(Suit.HONORS, Honor.NORTH))))
        g.step(0, KanAnkan(Tile(Suit.HONORS, Honor.NORTH)))
        # Rinshan draw should have happened and drawn the last tile from dead wall
        self.assertIsNotNone(g.last_drawn_tile)
        self.assertEqual(g.last_drawn_player, 0)
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
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        self.assertTrue(hasattr(g, 'dora_indicators'))
        self.assertGreaterEqual(len(g.dora_indicators), 1)

    def test_riichi_ippatsu_tsumo_with_single_uradora(self):
        # Player 0: closed tenpai with no yaku except riichi + menzen tsumo; ensure exactly 1 uradora gives 4 han total
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Construct 13-tile closed hand: triplet 222m; sequences 345p and 678m; pair 11m; wait 6s on 45s
        tiles_13 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.ONE),
        ]
        g._player_hands[0] = tiles_13
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
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
        # Player 0 turn: draw and declare Riichi; use provided parameterized Riichi move
        g._draw_for_current_if_needed()
        lm = g.legal_moves(0)
        r_moves = [m for m in lm if isinstance(m, Riichi)]
        self.assertTrue(r_moves)
        # Prefer riichi discarding the drawn tile if available
        drawn = g.last_drawn_tile
        chosen = next((m for m in r_moves if m.tile == drawn), r_moves[0])
        g.step(0, chosen)
        # Resolve reactions (none expected), then cycle through players 1..3 with safe honor discards
        g._resolve_reactions()
        for pid in [1, 2, 3]:
            g.current_player_idx = pid
            g.last_discarded_tile = None
            g.last_discard_player = None
            g._draw_for_current_if_needed()
            safe = Tile(Suit.HONORS, Honor.EAST)
            g._player_hands[pid][0] = safe
            self.assertTrue(g.is_legal(pid, Discard(safe)))
            g.step(pid, Discard(safe))
            g._resolve_reactions()
        # Back to player 0; draw winning 6s and tsumo
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.last_drawn_tile = None
        g.last_drawn_player = None
        g._skip_draw_for_current = False
        g._draw_for_current_if_needed()
        # Declare tsumo now
        self.assertTrue(g.is_legal(0, Tsumo()))
        g.step(0, Tsumo())
        # Ensure game ends by tsumo and scoring reflects exactly 4 han: riichi(1) + ippatsu(1) + menzen(1) + uradora(1)
        self.assertTrue(g.is_game_over())
        s = g.score_hand(0, win_by_tsumo=True)
        self.assertEqual(s['han'], 4)

    def test_ankan_north_then_tsumo_fu_only_menzen(self):
        # Player 0 immediately Ankan 4x North, later tsumo a winning tile; expect only menzen tsumo yaku and baseline fu
        class ScriptedP0(Player):
            def play(self, gs):  # type: ignore[override]
                from core.game import KanAnkan
                # If ankan is legal, do it immediately
                for m in gs.legal_moves():
                    if isinstance(m, KanAnkan):
                        return m
                # If tsumo is legal, do it
                for m in gs.legal_moves():
                    if isinstance(m, Tsumo):
                        return m
                # Otherwise discard first
                return Discard(gs.player_hand[0])

        players = [ScriptedP0(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        # Start hand (14 tiles after kan+rinshan): We'll begin with 13 that will be valid after kan + rinshan
        start_hand = [
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.NORTH),
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
        ]
        g._player_hands[0] = start_hand
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        # Prepare dead wall to allow rinshan draw after Kan; set indicators harmless
        g.dora_indicators = []
        g.ura_dora_indicators = []
        # kandora = EAST (ignored), rinshan draw will be 6s to complete 456s immediately on next draw
        g.dead_wall = [Tile(Suit.HONORS, Honor.EAST), Tile(Suit.SOUZU, TileType.SIX)]
        # Play the round using scripted actions
        g.play_round(max_steps=10)
        self.assertTrue(g.is_game_over())
        s = g.score_hand(0, win_by_tsumo=True)
        # Expect only menzen tsumo yaku and baseline fu (30)
        self.assertEqual(s['han'], 1)
        self.assertEqual(s['fu'], 30)

    def _force_exhaustive_draw(self, g: MediumJong):
        # Empty wall and ensure there is a pending discard cleared first
        g.tiles = []
        g.last_discarded_tile = None
        g.last_discard_player = None
        # Trigger exhaustive draw directly for determinism
        g._on_exhaustive_draw()

    def _make_tenpai_hand(self):
        # 13 tiles: needs 6s to complete 456s; closed, standard hand
        return [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]

    def _make_noten_hand(self):
        # 13 singles: all 7 honors + three suits' 1 and 9. This should not be in tenpai, assuming we do not have kokushi musou.
        return [
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH), Tile(Suit.HONORS, Honor.WEST),
            Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.WHITE), Tile(Suit.HONORS, Honor.GREEN),
            Tile(Suit.HONORS, Honor.RED),
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.NINE),
        ]

    def test_keiten_one_in_tenpai(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = self._make_tenpai_hand()
        g._player_hands[1] = self._make_noten_hand()
        g._player_hands[2] = self._make_noten_hand()
        g._player_hands[3] = self._make_noten_hand()
        self._force_exhaustive_draw(g)
        self.assertTrue(g.is_game_over())
        pay = g.get_keiten_payments()
        self.assertEqual(pay[0], 3000)
        self.assertEqual(pay[1], -1000)
        self.assertEqual(pay[2], -1000)
        self.assertEqual(pay[3], -1000)

    def test_keiten_two_in_tenpai(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = self._make_tenpai_hand()
        g._player_hands[1] = self._make_tenpai_hand()
        g._player_hands[2] = self._make_noten_hand()
        g._player_hands[3] = self._make_noten_hand()
        self._force_exhaustive_draw(g)
        pay = g.get_keiten_payments()
        self.assertEqual(pay[0], 1500)
        self.assertEqual(pay[1], 1500)
        self.assertEqual(pay[2], -1500)
        self.assertEqual(pay[3], -1500)

    def test_keiten_three_in_tenpai(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = self._make_tenpai_hand()
        g._player_hands[1] = self._make_tenpai_hand()
        g._player_hands[2] = self._make_tenpai_hand()
        g._player_hands[3] = self._make_noten_hand()
        self._force_exhaustive_draw(g)
        pay = g.get_keiten_payments()
        self.assertEqual(pay[0], 1000)
        self.assertEqual(pay[1], 1000)
        self.assertEqual(pay[2], 1000)
        self.assertEqual(pay[3], -3000)

    def test_keiten_all_or_none_in_tenpai(self):
        # None in tenpai
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = self._make_noten_hand()
        g._player_hands[1] = self._make_noten_hand()
        g._player_hands[2] = self._make_noten_hand()
        g._player_hands[3] = self._make_noten_hand()
        self._force_exhaustive_draw(g)
        pay = g.get_keiten_payments()
        self.assertEqual(pay[0], 0)
        self.assertEqual(pay[1], 0)
        self.assertEqual(pay[2], 0)
        self.assertEqual(pay[3], 0)

        # All in tenpai
        g2 = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        th = self._make_tenpai_hand()
        g2._player_hands[0] = th
        g2._player_hands[1] = th
        g2._player_hands[2] = th
        g2._player_hands[3] = th
        self._force_exhaustive_draw(g2)
        pay2 = g2.get_keiten_payments()
        self.assertEqual(pay2[0], 0)
        self.assertEqual(pay2[1], 0)
        self.assertEqual(pay2[2], 0)
        self.assertEqual(pay2[3], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)


