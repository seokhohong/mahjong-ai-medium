from __future__ import annotations

from typing import Any, Dict, List


from ..tile import Tile, TileType, Suit, Honor

from ..action import (
    Discard, Chi, Pon, Ron, Tsumo, PassCall,
    KanDaimin, KanKakan, KanAnkan, Riichi,
)
from .ac_constants import (
    chi_variant_index,
    ACTION_HEAD_ORDER,
    ACTION_HEAD_INDEX,
    TILE_HEAD_NOOP,
)


def _flat_tile_index(tile: Tile) -> int:
    """Map a Tile to the 0..36 flat index used by action masks (37 slots).

    Per-suit blocks:
    - Manzu: 0..9  (0m for aka five, 1m..9m -> 1..9)
    - Pinzu: 10..19 (0p -> 10, 1p..9p -> 11..19)
    - Souzu: 20..29 (0s -> 20, 1s..9s -> 21..29)
    - Honors: 30..36
    """
    if tile.suit == Suit.MANZU:
        if tile.tile_type == TileType.FIVE and getattr(tile, 'aka', False):
            return 0
        return int(tile.tile_type.value)
    if tile.suit == Suit.PINZU:
        if tile.tile_type == TileType.FIVE and getattr(tile, 'aka', False):
            return 10
        return 10 + int(tile.tile_type.value)
    if tile.suit == Suit.SOUZU:
        if tile.tile_type == TileType.FIVE and getattr(tile, 'aka', False):
            return 20
        return 20 + int(tile.tile_type.value)
    return 29 + int(tile.tile_type.value)


# =====================
# Two-head encode/decode
# =====================

def encode_two_head_action(action: Any) -> tuple[int, int]:
    """Encode a concrete action into (action_idx, tile_idx) under two-head policy.

    - action_idx indexes ACTION_HEAD_ORDER
    - tile_idx: 0..36 for tiles (direct flat index), TILE_HEAD_NOOP (37) for no-op
    """
    def to_tile_head_idx(t: Tile) -> int:
        return int(_flat_tile_index(t))

    # Discard / Riichi (tile-parameterized)
    if isinstance(action, Discard):
        return ACTION_HEAD_INDEX['discard'], to_tile_head_idx(action.tile)
    if isinstance(action, Riichi):
        return ACTION_HEAD_INDEX['riichi'], to_tile_head_idx(action.tile)
    # Singleton actions
    if isinstance(action, Tsumo):
        return ACTION_HEAD_INDEX['tsumo'], TILE_HEAD_NOOP
    if isinstance(action, Ron):
        return ACTION_HEAD_INDEX['ron'], TILE_HEAD_NOOP
    if isinstance(action, PassCall):
        return ACTION_HEAD_INDEX['pass'], TILE_HEAD_NOOP
    if isinstance(action, KanDaimin):
        return ACTION_HEAD_INDEX['kan_daimin'], TILE_HEAD_NOOP
    # Tile-parameterized Kans
    if isinstance(action, KanKakan):
        return ACTION_HEAD_INDEX['kan_kakan'], to_tile_head_idx(action.tile)
    if isinstance(action, KanAnkan):
        return ACTION_HEAD_INDEX['kan_ankan'], to_tile_head_idx(action.tile)
    # Chi (fully enumerated in action head)
    if isinstance(action, Chi):
        has_aka = False
        for tile in action.tiles:
            if tile.aka:
                has_aka = True

        if action.chi_variant_index == 0:
            return ACTION_HEAD_INDEX['chi_low_aka'] if has_aka else ACTION_HEAD_INDEX['chi_low_noaka'], TILE_HEAD_NOOP
        if action.chi_variant_index == 1:
            return ACTION_HEAD_INDEX['chi_mid_aka'] if has_aka else ACTION_HEAD_INDEX['chi_mid_noaka'], TILE_HEAD_NOOP
        if action.chi_variant_index == 2:
            return ACTION_HEAD_INDEX['chi_high_aka'] if has_aka else ACTION_HEAD_INDEX['chi_high_noaka'], TILE_HEAD_NOOP

    # Pon (fully enumerated in action head)
    if isinstance(action, Pon):
        for tile in action.tiles:
            if tile.aka:
                return ACTION_HEAD_INDEX['pon_aka'], TILE_HEAD_NOOP
        return ACTION_HEAD_INDEX['pon_noaka'], TILE_HEAD_NOOP

    return -1, -1


def build_move_from_two_head(gs, action_idx: int, tile_idx: int):
    """Construct an action from two-head indices.

    tile_idx: 0..36 => flat tile index; TILE_HEAD_NOOP (37) => no-op
    """
    if not (0 <= action_idx < len(ACTION_HEAD_ORDER)):
        return None
    name = ACTION_HEAD_ORDER[action_idx]

    def from_tile_head_idx(idx: int) -> int:
        # Accept direct 0..36; returns -1 on invalid or no-op
        if idx == TILE_HEAD_NOOP:
            return -1
        if 0 <= idx <= 36:
            return idx
        return -1

    # Helper: pick a hand tile matching a flat 0..36 index
    def find_hand_tile_by_flat(flat_idx: int) -> Tile | None:
        for t in gs.player_hand:
            if int(_flat_tile_index(t)) == flat_idx:
                return t
        return None

    # Non-parameterized
    if name == 'tsumo':
        return Tsumo()
    if name == 'ron':
        return Ron()
    if name == 'pass':
        return PassCall()
    if name == 'kan_daimin':
        # Select first available daiminkan from flat reactions (only one should be available)
        for r in gs.get_call_options():
            if isinstance(r, KanDaimin):
                return r if gs.is_legal(r) else None
        return None

    # Tile-parameterized via tile head
    if name in ('discard', 'riichi', 'kan_kakan', 'kan_ankan'):
        flat_idx = from_tile_head_idx(tile_idx)
        if flat_idx < 0:
            return None
        t = find_hand_tile_by_flat(flat_idx)
        if t is None:
            return None
        if name == 'discard':
            m = Discard(t)
        elif name == 'riichi':
            m = Riichi(t)
        elif name == 'kan_kakan':
            m = KanKakan(t)
        else:
            m = KanAnkan(t)
        return m if gs.is_legal(m) else None

    # Chi variants
    if name.startswith('chi_'):
        with_aka = name.endswith('_aka')
        base = name.split('_')[1]  # low/mid/high
        variant_map = {'low': 0, 'mid': 1, 'high': 2}
        variant = variant_map.get(base, -1)
        if variant < 0:
            return None
        last = gs._reactable_tile
        for r in gs.get_call_options():
            if isinstance(r, Chi) and getattr(r, 'chi_variant_index', -1) == variant:
                aka = False
                if last is not None:
                    aka = any(getattr(t, 'aka', False) for t in (r.tiles + [last]))
                if aka == with_aka:
                    return r if gs.is_legal(r) else None
        return None

    # Pon variants
    if name.startswith('pon_'):
        with_aka = name.endswith('_aka')
        last = gs._reactable_tile
        for r in gs.get_call_options():
            if isinstance(r, Pon):
                aka = False
                if last is not None:
                    aka = any(getattr(t, 'aka', False) for t in (r.tiles + [last]))
                if aka == with_aka:
                    return r if gs.is_legal(r) else None
        return None

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




