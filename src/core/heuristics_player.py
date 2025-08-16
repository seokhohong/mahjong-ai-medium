from __future__ import annotations
from typing import Dict, List

from .game import (
    Player,
    GamePerspective,
    Discard,
    Tsumo,
    Riichi,
    Ron,
    PassCall,
    Chi,
    Pon,
    KanDaimin,
    Tile,
    Suit,
    TileType,
    Honor,
)


def _count_in_hand(hand: List[Tile]) -> Dict[tuple, int]:
    cnt: Dict[tuple, int] = {}
    for t in hand:
        key = (t.suit, t.tile_type)
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def _is_tanyao_hand(hand: List[Tile]) -> bool:
    for t in hand:
        if t.suit == Suit.HONORS:
            return False
        v = int(t.tile_type.value)
        if v == 1 or v == 9:
            return False
    return True


def _has_yakuhai_anko(hand: List[Tile], seat_wind: Honor, round_wind: Honor) -> bool:
    cnt = _count_in_hand(hand)
    # Dragons
    for h in (Honor.WHITE, Honor.GREEN, Honor.RED):
        if cnt.get((Suit.HONORS, h), 0) >= 3:
            return True
    if cnt.get((Suit.HONORS, seat_wind), 0) >= 3:
        return True
    if cnt.get((Suit.HONORS, round_wind), 0) >= 3:
        return True
    return False


class MediumHeuristicsPlayer(Player):
    """Heuristic player for MediumJong.

    Priorities:
    - Tsumo if possible
    - Declare Riichi whenever possible (parameterized by discard tile)
    - Discard policy:
      1) Lone honors first (honors with count == 1)
      2) Otherwise discard tiles with fewest neighbors (same-suit +/-1)
    - Reactions:
      - Ron if possible
      - Decline Chi/Pon unless either there is a yakuhai ankō in hand or the hand is pure tanyao (all 2-8)
    """

    def play(self, game_state: GamePerspective):  # type: ignore[override]
        # Win immediately if possible
        for m in game_state.legal_moves():
            if isinstance(m, Tsumo):
                return m
        # Declare Riichi if available
        for m in game_state.legal_moves():
            if isinstance(m, Riichi):
                return m
        # Choose a discard per heuristic
        discards = [m for m in game_state.legal_moves() if isinstance(m, Discard)]
        if discards:
            hand = list(game_state.player_hand)
            counts = _count_in_hand(hand)

            def neighbor_count(tile: Tile) -> int:
                if tile.suit == Suit.HONORS:
                    return 0
                v = int(tile.tile_type.value)
                s = tile.suit
                # neighbors: same suit v-1 and v+1 in hand
                c = 0
                if v - 1 >= 1:
                    c += counts.get((s, TileType(v - 1)), 0)
                if v + 1 <= 9:
                    c += counts.get((s, TileType(v + 1)), 0)
                # duplicates slightly increase connections
                c += counts.get((s, tile.tile_type), 0) - 1
                return c

            def is_lone_honor(tile: Tile) -> bool:
                if tile.suit != Suit.HONORS:
                    return False
                return counts.get((tile.suit, tile.tile_type), 0) == 1

            def discard_key(d: Discard):
                t = d.tile
                return (
                    0 if is_lone_honor(t) else 1,
                    neighbor_count(t),
                    1 if t.suit == Suit.HONORS else 0,
                    int(t.tile_type.value) if t.suit != Suit.HONORS else int(t.tile_type.value),
                )

            discards.sort(key=discard_key)
            return discards[0]

        # Fallback: first legal move
        return game_state.legal_moves()[0]

    def choose_reaction(self, game_state: GamePerspective, options):  # type: ignore[override]
        # Ron if possible
        if game_state.can_ron():
            return Ron()
        # Decide if we allow calling chi/pon
        hand = list(game_state.player_hand)
        allow_calls = _has_yakuhai_anko(hand, game_state.seat_winds[game_state.player_id], game_state.round_wind) or _is_tanyao_hand(hand)
        if not allow_calls:
            return PassCall()
        # Prefer KanDaimin over Pon over Chi if allowed
        if options and options.get('kan_daimin'):
            return KanDaimin(options['kan_daimin'][0])
        if options and options.get('pon'):
            return Pon(options['pon'][0])
        if options and options.get('chi'):
            return Chi(options['chi'][0])
        return PassCall()


