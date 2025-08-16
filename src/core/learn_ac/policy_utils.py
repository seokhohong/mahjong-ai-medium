from __future__ import annotations

from typing import Any, Dict, List

from ..game import GamePerspective, Tile, TileType, Suit, Discard, Chi, Pon, Ron, Tsumo, PassCall
from ..encoding import tile_to_index
from .ac_constants import chi_variant_index
from ..learn.pure_policy_dataset import serialize_state, serialize_action  # type: ignore
import numpy as np


def flat_index_for_action(state_dict: Dict[str, Any], action_dict: Dict[str, Any]) -> int:
    """Map a serialized (state, action) pair to a single flat policy index (0..24).

    0..17: discard tiles; 18..20: chi low/mid/high; 21: pon; 22: ron; 23: tsumo; 24: pass
    """
    atype = action_dict.get('type', 'pass')
    if atype == 'discard':
        tile = action_dict.get('tile')
        return int(tile_to_index(_tile_from_str(tile))) if tile is not None else -1
    if atype == 'chi':
        tiles = action_dict.get('tiles', []) or []
        last = state_dict.get('last_discarded_tile')
        last_tile = _tile_from_str(last) if last else None
        pair: List[Tile] = [_tile_from_str(tiles[0]), _tile_from_str(tiles[1])] if len(tiles) >= 2 else []
        chi_idx = chi_variant_index(last_tile, pair)
        return 18 + int(chi_idx) if chi_idx is not None and chi_idx >= 0 else -1
    if atype == 'pon':
        return 21
    if atype == 'ron':
        return 22
    if atype == 'tsumo':
        return 23
    # pass/unknown
    return 24


def build_move_from_flat(gs: GamePerspective, choice: int):
    """Construct a concrete move from a flat policy index using game perspective."""
    if 0 <= choice <= 17:
        for t in gs.player_hand:
            if int(tile_to_index(t)) == int(choice):
                return Discard(t)
        return None
    if 18 <= choice <= 20:
        opts = gs.get_call_options().get('chi', [])
        last = getattr(gs, 'last_discarded_tile', None)
        for pair in opts:
            v = chi_variant_index(last, pair)
            if v == (choice - 18):
                return Chi(pair)
        return None
    if choice == 21:
        opts = gs.get_call_options().get('pon', [])
        return Pon(opts[0]) if opts else None
    if choice == 22:
        return Ron()
    if choice == 23:
        return Tsumo()
    if choice == 24:
        return PassCall()
    return None


def _tile_from_str(s: str) -> Tile:
    v = int(str(s)[:-1])
    suit = Suit(str(s)[-1])
    return Tile(suit, TileType(v))


def legal_flat_mask(gs: GamePerspective) -> np.ndarray:
    """Compute a 25-length 0/1 mask of legal flat actions for a perspective using the flat-index mapping.

    Leverages gs.legal_moves() and the same flat mapping used elsewhere to ensure consistency.
    """
    mask = np.zeros(25, dtype=np.float64)
    sd = serialize_state(gs)
    for move in gs.legal_moves():
        ad = serialize_action(move)
        idx = flat_index_for_action(sd, ad)
        if 0 <= idx < 25:
            mask[int(idx)] = 1.0
    return mask


