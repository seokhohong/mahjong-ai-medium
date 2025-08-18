from __future__ import annotations

# Canonical order for AC main policy head
# Indices correspond to logits/probabilities emitted by ACNetwork.head_action
MAIN_HEAD_ORDER = ['chi', 'pon', 'ron', 'tsumo', 'discard', 'pass']

# Mapping from action type string to main head index
MAIN_HEAD_INDEX = {name: idx for idx, name in enumerate(MAIN_HEAD_ORDER)}

# Generic MediumJong constants for AC components
NUM_PLAYERS: int = 4
# Flat policy space sizing matches policy_utils/build_move_from_flat and GamePerspective.legal_flat_mask
# Layout (length 152):
# 0..34:  Discard by tile index
# 35..69: Riichi by discard tile
# 70:     Tsumo
# 71:     Ron
# 72..77: Chi variants (no-aka 0..2, with-aka 3..5)
# 78..79: Pon (no-aka, with-aka)
# 80:     Kan Daimin
# 81..115: Kan Kakan by tile index
# 116..150: Kan Ankan by tile index
# 151:    Pass
FLAT_POLICY_SIZE: int = 152
# Tile index space (0 is padding; 1..37 represent suited ranks/suits per encoding)
TILE_INDEX_PAD: int = 0
TILE_INDEX_SIZE: int = 38

# Training-time caps
AC_MAX_CALLED_TILES_PER_PLAYER: int = 12
AC_MAX_CALLS: int = 4
AC_MAX_CALLED_SET_SIZE: int = 4
# Called-sets serialized default shape: [player, set_index, tiles_in_set]
CALLED_SETS_DEFAULT_SHAPE = (AC_MAX_CALLED_SET_SIZE, AC_MAX_CALLS, AC_MAX_CALLED_SET_SIZE)

AC_MAX_DISCARDS_PER_PLAYER: int = 21
# Game-state vector length used by serialization pipeline (fallback default)
GAME_STATE_VEC_LEN: int = 32
# Default maximum turns for episode rollout
MAX_TURNS: int = 200


def action_type_to_main_index(action_type: str) -> int:
    """Map a serialized action type to the main head index.

    Falls back to 'pass' on unknown types.
    """
    at = (action_type or 'pass').lower()
    return MAIN_HEAD_INDEX.get(at, MAIN_HEAD_INDEX['pass'])


def chi_variant_index(last_discarded_tile: 'Tile', tiles: list['Tile']) -> int:
    """Return chi variant index relative to the last discarded tile.

    0: left (low)  -> tiles ranks == [d-2, d-1]
    1: mid         -> tiles ranks == [d-1, d+1]
    2: right (high)-> tiles ranks == [d+1, d+2]

    Returns -1 if not a recognizable chi pair relative to last.
    """
    # Local import to avoid circulars
    try:
        from ..game import Tile  # type: ignore
    except Exception:
        Tile = object  # type: ignore
    if last_discarded_tile is None or len(tiles) < 2:
        return -1
    d = int(getattr(last_discarded_tile.tile_type, 'value', 0))
    try:
        ranks = sorted(int(getattr(t.tile_type, 'value', 0)) for t in tiles)
    except Exception:
        return -1
    if ranks == [d - 2, d - 1]:
        return 0
    if ranks == [d - 1, d + 1]:
        return 1
    if ranks == [d + 1, d + 2]:
        return 2
    return -1


