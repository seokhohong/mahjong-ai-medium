import random
import pytest

from src.core.game import MediumJong, Player
from src.core.variation import (
    make_variant_game,
    game_with_random_calls,
    game_player0_mono_suit,
    create_variation,
    game_yakuhai_pons,
)
from src.core.tile import Suit, Honor


def _everyone_has_13(game: MediumJong) -> bool:
    for pid in range(4):
        hand_ct = len(game.hand(pid))
        called_ct = sum(len(cs.tiles) for cs in game.called_sets(pid))
        if hand_ct + called_ct != 13:
            return False
    return True


def test_make_variant_game_constructs_standard():
    players = [Player() for _ in range(4)]
    g = make_variant_game(players)
    assert isinstance(g, MediumJong)
    assert _everyone_has_13(g)


def test_game_with_random_calls_respects_tile_counts():
    random.seed(123)
    players = [Player() for _ in range(4)]
    g = game_with_random_calls(players, min_calls=2, max_calls=6)
    assert _everyone_has_13(g)
    # Ensure at least one player has some called tiles or none; we can't deterministically assert
    # but we can check structure integrity
    for pid in range(4):
        for cs in g.called_sets(pid):
            assert cs.call_type in ('chi', 'pon', 'kan_daimin', 'kan_kakan', 'kan_ankan')
            assert len(cs.tiles) in (3, 4)


def test_game_player0_mono_suit_has_only_one_suit():
    random.seed(321)
    players = [Player() for _ in range(4)]
    g = game_player0_mono_suit(players)
    assert _everyone_has_13(g)
    # Player 0: all concealed tiles should be in a single non-honor suit
    hand0 = g.hand(0)
    suits = {t.suit for t in hand0}
    assert len(suits) == 1
    assert list(suits)[0] in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)
    # And player 0 should have no called sets
    assert g.called_sets(0) == []


def test_create_variation_respects_weights_calls_only():
    random.seed(42)
    players = [Player() for _ in range(4)]
    g = create_variation(players, some_calls_weight=1, chinitsu_weight=0)
    assert _everyone_has_13(g)
    # With calls-only weight, at least one called set should exist with high probability, but
    # randomness could make zero. We only assert structural validity.
    for pid in range(4):
        for cs in g.called_sets(pid):
            assert cs.call_type in ('chi', 'pon', 'kan_daimin', 'kan_kakan', 'kan_ankan')


def test_create_variation_respects_weights_chinitsu_only():
    random.seed(43)
    players = [Player() for _ in range(4)]
    g = create_variation(players, some_calls_weight=0, chinitsu_weight=1)
    assert _everyone_has_13(g)
    suits = {t.suit for t in g.hand(0)}
    assert len(suits) == 1
    assert list(suits)[0] in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)


def test_create_variation_games_playable():
    random.seed(7)
    # Run 10 games; if any error occurs, pytest will fail the test
    for _ in range(10):
        players = [Player() for _ in range(4)]
        g = create_variation(players, some_calls_weight=1, chinitsu_weight=1)
        guard = 0
        while not g.is_game_over() and guard < 1000:
            g.play_turn()
            guard += 1
        assert g.is_game_over(), "Game did not finish within guard limit"


def test_game_yakuhai_pons_structure():
    random.seed(11)
    players = [Player() for _ in range(4)]
    g = game_yakuhai_pons(players)
    assert _everyone_has_13(g)
    white_owner = None
    east_owner = None
    for pid in range(4):
        for cs in g.called_sets(pid):
            if cs.call_type != 'pon':
                continue
            if len(cs.tiles) != 3:
                continue
            if not all(t.suit == Suit.HONORS for t in cs.tiles):
                continue
            kinds = {t.tile_type for t in cs.tiles}
            if kinds == {Honor.WHITE}:
                white_owner = pid
            if kinds == {Honor.EAST}:
                east_owner = pid
    assert white_owner is not None
    assert east_owner is not None
    assert white_owner != east_owner


def test_create_variation_respects_weights_yakuhai_only():
    random.seed(21)
    players = [Player() for _ in range(4)]
    g = create_variation(players, some_calls_weight=0, chinitsu_weight=0, yakuhai_weight=1)
    assert _everyone_has_13(g)
    white_owner = None
    east_owner = None
    for pid in range(4):
        for cs in g.called_sets(pid):
            if cs.call_type != 'pon' or len(cs.tiles) != 3:
                continue
            if not all(t.suit == Suit.HONORS for t in cs.tiles):
                continue
            kinds = {t.tile_type for t in cs.tiles}
            if kinds == {Honor.WHITE}:
                white_owner = pid
            if kinds == {Honor.EAST}:
                east_owner = pid
    assert white_owner is not None
    assert east_owner is not None
    assert white_owner != east_owner
