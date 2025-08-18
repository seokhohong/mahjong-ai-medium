from __future__ import annotations

from typing import Any, Dict, List

from ..game import (
    GamePerspective, Tile, TileType, Suit, Honor,
    Discard, Chi, Pon, Ron, Tsumo, PassCall,
    KanDaimin, KanKakan, KanAnkan, Riichi,
)
from .ac_constants import chi_variant_index
from .feature_engineering import tile_to_index
import numpy as np


def flat_index_for_action(gs: GamePerspective, action: Any) -> int:
    """Map an action in a given GamePerspective to the flat policy index (length 152).

    Layout:
      0..34: discard
      35..69: riichi (by discard tile)
      70: tsumo
      71: ron
      72..77: chi no-aka (0..2), chi with-aka (3..5)
      78..79: pon [no-aka, with-aka]
      80: daiminkan
      81..115: kakan (by tile)
      116..150: ankan (by tile)
      151: pass
    """
    def tile_flat_index(t: Tile) -> int:
        # Mirror the feature_engineering mapping (0/9/18 are aka fives; honors 28..34)
        return int(tile_to_index(t))

    if isinstance(action, Discard):
        return tile_flat_index(action.tile)
    if isinstance(action, Riichi):
        return 35 + tile_flat_index(action.tile)
    if isinstance(action, Tsumo):
        return 70
    if isinstance(action, Ron):
        return 71
    if isinstance(action, Chi):
        last = gs.last_discarded_tile
        v = chi_variant_index(last, action.tiles)
        if v is None or v < 0:
            return -1
        aka = any(t.aka for t in (action.tiles + ([last] if last else [])))
        return 72 + v + (3 if aka else 0)
    if isinstance(action, Pon):
        last = gs.last_discarded_tile
        aka = any(t.aka for t in (action.tiles + ([last] if last else [])))
        return 79 if aka else 78
    if isinstance(action, KanDaimin):
        return 80
    if isinstance(action, KanKakan):
        return 81 + tile_flat_index(action.tile)
    if isinstance(action, KanAnkan):
        return 116 + tile_flat_index(action.tile)
    if isinstance(action, PassCall):
        return 151
    return -1


def build_move_from_flat(gs: GamePerspective, choice: int):
    """Construct a concrete move from a flat policy index using the perspective."""
    # Discard
    if 0 <= choice <= 34:
        target = choice
        for t in gs.player_hand:
            if int(tile_to_index(t)) == target:
                return Discard(t)
        return None
    # Riichi
    if 35 <= choice <= 69:
        idx = choice - 35
        for t in gs.player_hand:
            if int(tile_to_index(t)) == idx:
                move = Riichi(t)
                if gs.is_legal(move):
                    return move
        return None
    # Tsumo / Ron
    if choice == 70:
        return Tsumo()
    if choice == 71:
        return Ron()
    # Chi
    if 72 <= choice <= 77:
        base = choice - 72
        with_aka = base >= 3
        variant = base - 3 if with_aka else base
        opts = gs.get_call_options().get('chi', [])
        last = gs.last_discarded_tile
        for pair in opts:
            v = chi_variant_index(last, pair)
            if v == variant:
                aka = any(t.aka for t in (pair + ([last] if last else [])))
                if aka == with_aka:
                    move = Chi(pair)
                    if gs.is_legal(move):
                        return move
        return None
    # Pon
    if 78 <= choice <= 79:
        with_aka = (choice == 79)
        opts = gs.get_call_options().get('pon', [])
        last = gs.last_discarded_tile
        for pair in opts:
            aka = any(t.aka for t in (pair + ([last] if last else [])))
            if aka == with_aka:
                move = Pon(pair)
                if gs.is_legal(move):
                    return move
        return None
    # Daiminkan
    if choice == 80:
        opts = gs.get_call_options().get('kan_daimin', [])
        if opts:
            move = KanDaimin(opts[0])
            if gs.is_legal(move):
                return move
        return None
    # Kakan
    if 81 <= choice <= 115:
        idx = choice - 81
        for t in gs.player_hand:
            if int(tile_to_index(t)) == idx:
                move = KanKakan(t)
                if gs.is_legal(move):
                    return move
        return None
    # Ankan
    if 116 <= choice <= 150:
        idx = choice - 116
        for t in gs.player_hand:
            if int(tile_to_index(t)) == idx:
                move = KanAnkan(t)
                if gs.is_legal(move):
                    return move
        return None
    # Pass
    if choice == 151:
        return PassCall()
    return None


def _tile_from_str(s: str) -> Tile:
    # Handle honor single-letter forms
    if len(s) == 1:
        honor_map = {
            'E': Honor.EAST, 'S': Honor.SOUTH, 'W': Honor.WEST, 'N': Honor.NORTH,
            'P': Honor.WHITE, 'G': Honor.GREEN, 'R': Honor.RED,
        }
        if s in honor_map:
            return Tile(Suit.HONORS, honor_map[s])
    # Suited forms like '5p', with aka as '0m/0p/0s'
    rank_str, suit_char = str(s)[:-1], str(s)[-1]
    suit = Suit(suit_char)
    if rank_str == '0':
        return Tile(suit, TileType.FIVE, aka=True)
    v = int(rank_str)
    return Tile(suit, TileType(v))


def serialize_action(action: Any) -> Dict[str, Any]:
    """Centralized minimal action serialization for logging or datasets."""
    if isinstance(action, Discard):
        return {'type': 'discard', 'tile': str(action.tile)}
    if isinstance(action, Riichi):
        return {'type': 'riichi', 'tile': str(action.tile)}
    if isinstance(action, Chi):
        return {'type': 'chi', 'tiles': [str(action.tiles[0]), str(action.tiles[1])]}
    if isinstance(action, Pon):
        return {'type': 'pon'}
    if isinstance(action, KanDaimin):
        return {'type': 'kan_daimin'}
    if isinstance(action, KanKakan):
        return {'type': 'kan_kakan', 'tile': str(action.tile)}
    if isinstance(action, KanAnkan):
        return {'type': 'kan_ankan', 'tile': str(action.tile)}
    if isinstance(action, Ron):
        return {'type': 'ron'}
    if isinstance(action, Tsumo):
        return {'type': 'tsumo'}
    return {'type': 'pass'}




