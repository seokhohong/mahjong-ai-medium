from __future__ import annotations

# Two-head policy specification
# Action head enumerates core actions, including fully enumerated Chi (6) and Pon (2) variants.
# Tile head enumerates tiles plus a no-op index.

# Action head order (size = 14)
# - Singletons (not parameterized by tile head): tsumo, ron, pass
# - Tile-parameterized (use tile head): discard, riichi, kan
# - Chi variants (6): low/mid/high x no-aka/aka
# - Pon variants (2): no-aka, aka
ACTION_HEAD_ORDER: list[str] = [
    'discard',       # uses tile head
    'riichi',        # uses tile head
    'tsumo',
    'ron',
    'pass',
    'kan',           # uses tile head (kakan/ankan pass tile; daiminkan uses no-op)
    'chi_low_noaka',
    'chi_mid_noaka',
    'chi_high_noaka',
    'chi_low_aka',
    'chi_mid_aka',
    'chi_high_aka',
    'pon_noaka',
    'pon_aka',
]
ACTION_HEAD_INDEX = {name: i for i, name in enumerate(ACTION_HEAD_ORDER)}
ACTION_HEAD_SIZE: int = len(ACTION_HEAD_ORDER)

# Tile head indexing (size = 38):
# - 0..36 = flat tile index (direct mapping)
# - 37    = no-op (TILE_HEAD_NOOP)
TILE_HEAD_SIZE: int = 38
TILE_HEAD_NOOP: int = TILE_HEAD_SIZE - 1

# Legacy flat policy (kept for reference/back-compat reading of old datasets/models only)
FLAT_POLICY_SIZE: int = 160
# Tile index space used by feature embeddings (unchanged)
TILE_INDEX_PAD: int = 0
TILE_INDEX_SIZE: int = 38

from ..constants import (
    MAX_CALLS,
    MAX_CALLED_SET_SIZE,
    MAX_CALLED_TILES_PER_PLAYER,
    CALLED_SETS_DEFAULT_SHAPE,
    MAX_DISCARDS_PER_PLAYER,
)
# Game-state vector length used by serialization pipeline (fallback default)
GAME_STATE_VEC_LEN: int = 32
# Default maximum turns for episode rollout
MAX_TURNS: int = 256


def action_type_to_main_index(action_type: str) -> int:
    """Legacy helper; for two-head setup prefer ACTION_HEAD_* constants."""
    at = (action_type or 'pass').lower()
    # Best-effort mapping
    return ACTION_HEAD_INDEX.get(at, ACTION_HEAD_INDEX['pass'])


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
    raise IllegalChiException()

class IllegalChiException(Exception):
    pass

