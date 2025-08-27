#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.tile import Tile, TileType, Suit, Honor
from core.action import Discard, Action
from core.game import GamePerspective, CalledSet
from core.learn.recording_ac_player import RecordingACPlayer, TENPAI_REWARD, YAKUHAI_REWARD
from core.learn.policy_utils import encode_two_head_action


class _ForcedMovePlayer(RecordingACPlayer):
    """RecordingACPlayer that always returns a pre-set move from compute_play."""
    def __init__(self, forced_move):
        class _DummyNet:
            def evaluate(self, gs):
                # Not used
                return [0.0]*10, [0.0]*40, 0.0
        super().__init__(network=_DummyNet(), temperature=1.0)
        self._forced_move = forced_move

    def compute_play(self, gs):  # type: ignore[override]
        ai, ti = encode_two_head_action(self._forced_move)
        return self._forced_move, 0.0, int(ai), int(ti), 0.0


class TestRecordingPlayerTenpaiReward(unittest.TestCase):
    def _base_gp(self, hand, called_sets_by_player, newly_drawn_tile=None):
        return GamePerspective(player_hand=hand, remaining_tiles=30, reactable_tile=, owner_of_reactable_tile=None,
                               called_sets=called_sets_by_player, player_discards={0: [], 1: [], 2: [], 3: []},
                               called_discards={0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=newly_drawn_tile,
                               seat_winds={0: Honor.EAST, 1: Honor.SOUTH, 2: Honor.WEST, 3: Honor.NORTH},
                               round_wind=Honor.EAST, dora_indicators=[],
                               riichi_declaration_tile={0: -1, 1: -1, 2: -1, 3: -1})

    def test_reward_given_when_discard_puts_into_tenpai_closed(self):
        # Known 13-tile tenpai; add extra junk then discard it
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        tenpai13 = base_s + [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
                              Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        extra = Tile(Suit.HONORS, Honor.WHITE)
        hand14 = list(tenpai13) + [extra]
        # Set newly_drawn_tile to a tile from the 13 so prior 13 become noten
        newly_drawn = Tile(Suit.PINZU, TileType.FOUR)
        gp = self._base_gp(hand14, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=newly_drawn)

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        # Last recorded reward should be TENPAI_REWARD
        self.assertAlmostEqual(player.experience.rewards[-1], TENPAI_REWARD)

    def test_no_reward_when_discard_not_puts_into_tenpai_closed(self):
        # Noten 13; add arbitrary extra and discard inside-suit that keeps noten
        noten13 = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST), Tile(Suit.HONORS, Honor.NORTH),
        ]
        extra = Tile(Suit.PINZU, TileType.TWO)
        hand14 = list(noten13) + [extra]
        gp = self._base_gp(hand14, {0: [], 1: [], 2: [], 3: []})

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        self.assertAlmostEqual(player.experience.rewards[-1], 0.0)

    def test_reward_given_when_discard_puts_into_tenpai_open(self):
        # Open hand scenario mirroring test in test_tenpai
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
        extra = Tile(Suit.HONORS, Honor.WHITE)
        # IMPORTANT: player's concealed hand should NOT include called set tiles.
        # Build 8 concealed tiles (7 + extra) and provide called sets separately.
        hand14 = list(concealed7) + [extra]
        # Choose newly drawn tile from concealed7 so prior 13 (concealed7-td + extra) is noten
        newly_drawn = Tile(Suit.PINZU, TileType.FOUR)
        gp = self._base_gp(hand14, {0: [chi, pon], 1: [], 2: [], 3: []}, newly_drawn_tile=newly_drawn)

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        self.assertAlmostEqual(player.experience.rewards[-1], TENPAI_REWARD)

    def test_reward_given_after_call_then_discard_into_tenpai_open(self):
        # Same open-hand as above, but simulate post-call discard (no newly_drawn_tile)
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
        extra = Tile(Suit.HONORS, Honor.WHITE)
        hand14 = list(concealed7) + [extra]
        # newly_drawn_tile=None simulates that we just called and now must discard
        gp = self._base_gp(hand14, {0: [chi, pon], 1: [], 2: [], 3: []}, newly_drawn_tile=None)

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        self.assertAlmostEqual(player.experience.rewards[-1], TENPAI_REWARD)


    def test_yakuhai_reward_on_draw_closed(self):
        # Two WHITE in hand; draw third WHITE -> should award yakuhai reward
        white = Tile(Suit.HONORS, Honor.WHITE)
        noten13 = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH),
            white, white,
        ]
        extra = Tile(Suit.MANZU, TileType.TWO)
        hand14 = list(noten13) + [Tile(Suit.HONORS, Honor.WHITE)]
        gp = self._base_gp(hand14, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=Tile(Suit.HONORS, Honor.WHITE))

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        self.assertAlmostEqual(player.experience.rewards[-1], YAKUHAI_REWARD)

    def test_yakuhai_reward_on_pon_open(self):
        # Pon EAST triplet should award yakuhai reward
        from core.action import Pon
        east = Tile(Suit.HONORS, Honor.EAST)
        hand = [Tile(Suit.MANZU, TileType.ONE)] * 14
        gp = self._base_gp(hand, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=None)
        move = Pon(tiles=[east, east, east])
        player = _ForcedMovePlayer(move)
        player.choose_reaction(gp, options=[move])
        self.assertAlmostEqual(player.experience.rewards[-1], YAKUHAI_REWARD)

    def test_no_yakuhai_reward_on_draw_non_yakuhai_honor(self):
        # Draw third WEST when seat wind EAST and round wind EAST -> WEST is not yakuhai => no reward
        west = Tile(Suit.HONORS, Honor.WEST)
        noten13 = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH),
            west, west,
        ]
        extra = Tile(Suit.MANZU, TileType.TWO)
        hand14 = list(noten13) + [Tile(Suit.HONORS, Honor.WEST)]
        gp = self._base_gp(hand14, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=Tile(Suit.HONORS, Honor.WEST))

        move = Discard(tile=extra)
        player = _ForcedMovePlayer(move)
        player.play(gp)
        # Could be tenpai reward in some crafted hands; assert reward is not YAKUHAI_REWARD by expecting 0 or TENPAI_REWARD
        self.assertIn(player.experience.rewards[-1], (0.0, TENPAI_REWARD))

    def test_no_yakuhai_reward_on_chi(self):
        # Chi should never yield yakuhai reward
        from core.action import Chi
        tiles = [Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE)]
        move = Chi(tiles=tiles, chi_variant_index=0)
        hand = [Tile(Suit.MANZU, TileType.ONE)] * 14
        gp = self._base_gp(hand, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=None)
        player = _ForcedMovePlayer(move)
        player.choose_reaction(gp, options=[move])
        self.assertAlmostEqual(player.experience.rewards[-1], 0.0)

    def test_no_yakuhai_reward_on_pon_non_yakuhai(self):
        # Pon WEST when not seat/round wind -> no yakuhai reward
        from core.action import Pon
        west = Tile(Suit.HONORS, Honor.WEST)
        move = Pon(tiles=[west, west, west])
        hand = [Tile(Suit.MANZU, TileType.ONE)] * 14
        gp = self._base_gp(hand, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=None)
        player = _ForcedMovePlayer(move)
        player.choose_reaction(gp, options=[move])
        self.assertAlmostEqual(player.experience.rewards[-1], 0.0)

    def test_finalize_rescales_positive_intermediates(self):
        # Build three steps: two positive intermediates (yakuhai, tenpai) and a neutral final step.
        # Finalize with a terminal value smaller than their sum and verify proportional scaling.
        # Step 1: yakuhai draw (positive)
        white = Tile(Suit.HONORS, Honor.WHITE)
        noten13 = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH),
            white, white,
        ]
        extra = Tile(Suit.MANZU, TileType.TWO)
        hand14_y = list(noten13) + [Tile(Suit.HONORS, Honor.WHITE)]
        gp_y = self._base_gp(hand14_y, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=Tile(Suit.HONORS, Honor.WHITE))
        pmove1 = Discard(tile=extra)
        player = _ForcedMovePlayer(pmove1)
        player.play(gp_y)

        # Step 2: tenpai transition (positive)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        tenpai13 = base_s + [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN),
                              Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        extra2 = Tile(Suit.HONORS, Honor.WHITE)
        hand14_t = list(tenpai13) + [extra2]
        newly_drawn2 = Tile(Suit.PINZU, TileType.FOUR)
        gp_t = self._base_gp(hand14_t, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=newly_drawn2)
        pmove2 = Discard(tile=extra2)
        player._forced_move = pmove2
        player.play(gp_t)

        # Step 3: neutral (no yakuhai, no tenpai change)
        neutral_hand = [Tile(Suit.MANZU, TileType.ONE)] * 14
        gp_n = self._base_gp(neutral_hand, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=Tile(Suit.MANZU, TileType.TWO))
        pmove3 = Discard(tile=Tile(Suit.MANZU, TileType.THREE))
        player._forced_move = pmove3
        player.play(gp_n)

        # Now finalize with smaller terminal cap than sum of positives (0.05 + 0.1 = 0.15)
        terminal = 0.08
        player.finalize_episode(terminal)

        # After finalize: first two (intermediate) positives scaled to sum to terminal, last equals terminal
        self.assertEqual(len(player.experience.rewards), 3)
        pos_sum = player.experience.rewards[0] + player.experience.rewards[1]
        self.assertAlmostEqual(pos_sum, terminal)
        self.assertAlmostEqual(player.experience.rewards[-1], terminal)

    def test_finalize_zeroes_positive_when_terminal_nonpositive(self):
        # Create a single positive intermediate, then finalize with negative terminal
        white = Tile(Suit.HONORS, Honor.WHITE)
        hand14 = [white, white] + [Tile(Suit.MANZU, TileType.ONE)]*11 + [white]
        gp = self._base_gp(hand14, {0: [], 1: [], 2: [], 3: []}, newly_drawn_tile=white)
        move = Discard(tile=Tile(Suit.MANZU, TileType.TWO))
        player = _ForcedMovePlayer(move)
        player.play(gp)
        # Add a neutral second step so we have a terminal slot
        player._forced_move = Discard(tile=Tile(Suit.MANZU, TileType.THREE))
        player.play(self._base_gp([Tile(Suit.MANZU, TileType.ONE)]*14, {0: [], 1: [], 2: [], 3: []}))

        player.finalize_episode(-1.0)
        # First reward should be clamped to 0, last reward is -1.0
        self.assertAlmostEqual(player.experience.rewards[0], 0.0)
        self.assertAlmostEqual(player.experience.rewards[-1], -1.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
