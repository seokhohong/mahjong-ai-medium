#!/usr/bin/env python3
import unittest
import random

from core.game import MediumJong, Player, OutcomeType
from core.tile import Tile, TileType, Suit
from core.tenpai import waits_optimized as _waits_opt


class TestWaitsTracking(unittest.TestCase):
    def test_waits_matches_optimized_on_draws_over_multiple_games(self):
        # Compare MediumJong.get_waits() vs tenpai.waits_optimized() after each draw
        seeds = [11, 123, 2025]
        for seed in seeds:
            random.seed(seed)
            g = MediumJong([Player(), Player(), Player(), Player()])

            # Perform a single controlled draw per game and compare waits
            if g.tiles:
                g._draw_tile()
                pid = g.current_player_idx

                hand = list(g._player_hands[pid])
                called_sets = list(g._player_called_sets[pid])
                expected = sorted(_waits_opt(hand, called_sets), key=lambda t: (t.suit.value, int(t.tile_type.value), getattr(t, 'aka', False)))
                actual = g.get_oracle().get_waits(pid)

                self.assertEqual(
                    [str(t) for t in actual],
                    [str(t) for t in expected],
                    f"seed={seed} pid={pid} mismatch waits: actual={actual} expected={expected}"
                )

    def test_waits_contains_ron_tile_in_basic_scenario(self):
        # Controlled scenario: player 1 waits on 3p (ryanmen 2-3-4p shape). After P0 discards 3p, P1 should be able to ron.
        from mj_test.test_core.test_utils import ForceDiscardPlayer, NoReactionPlayer

        g = MediumJong([
            ForceDiscardPlayer(Tile(Suit.PINZU, TileType.THREE)),
            Player(),
            NoReactionPlayer(),
            NoReactionPlayer(),
        ])
        # Construct P1 hand that is waiting on 3p
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]

        # Sanity: waits_optimized should include 3p for P1
        expected = _waits_opt(list(g._player_hands[1]), list(g._player_called_sets[1]))
        self.assertIn(Tile(Suit.PINZU, TileType.THREE), expected)

        # The game's waits getter should also include 3p once it's P1's draw
        # Directly set it to P1's turn and draw a harmless tile to trigger waits update
        g.current_player_idx = 1
        g._reactable_tile = None
        g._owner_of_reactable_tile = None
        g.tiles = [Tile(Suit.MANZU, TileType.TWO)] + g.tiles
        if g.tiles:
            g._draw_tile()
        waits_p1 = [str(t) for t in g.get_oracle().get_waits(1)]
        self.assertIn(str(Tile(Suit.PINZU, TileType.THREE)), waits_p1)

    def test_deals_in_matches_ron_tile_over_5_games(self):
        random.seed(12345)
        for _ in range(5):
            g = MediumJong([Player(), Player(), Player(), Player()])
            g.play_round()
            outcome = g.get_game_outcome()
            # Only check Ron cases
            if outcome.is_draw:
                continue
            loser = g.get_loser()
            winners = g.get_winners()
            # If tsumo, skip; only validate ron
            if any(outcome.outcome_type(w) != OutcomeType.RON for w in winners):
                continue
            self.assertIsNotNone(loser)
            self.assertTrue(len(winners) >= 1)
            # Recompute waits at terminal state to be robust
            oracle = g.get_oracle()
            for pid in range(4):
                oracle.update_waits_for(g, pid)
            # Ron tile is the last reactable tile from the loser
            ron_tile = g.player_discards[loser][-1]
            self.assertIsNotNone(ron_tile)
            # deals_in for the loser should include the ron tile (aka-insensitive for rule logic)
            di = oracle.deal_in_tiles(loser)  # type: ignore[arg-type]
            self.assertTrue(
                any(t.functionally_equal(ron_tile) for t in di),
                msg=f"loser={loser} winners={winners} ron_tile={ron_tile} deals_in={[str(t) for t in di]}\nOutcome={outcome}"
            )


if __name__ == '__main__':
    unittest.main()
