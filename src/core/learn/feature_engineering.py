from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np

from core import game
from core.constants import (
    MAX_CALLS as MAX_CALLED_SETS,
    MAX_CALLED_SET_SIZE as MAX_TILES_PER_CALLED_SET,
    MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
)
from core.game import GamePerspective, CalledSet
from core.tile import Tile, Suit, TileType, Honor, tile_flat_index, tile_from_flat_index, UNIQUE_TILE_COUNT


# Fixed sizes for vectorization (aligned with game rules)
# A player's hand should be 13 tiles after actions are resolved.
# During the acting player's turn (pre-discard), the transient hand may be 14.
MAX_HAND_LEN: int = 13
PAD_IDX: int = -1  # padding index; 0.. are valid tile indices (including aka slots)

def tile_is_aka(tile: Tile) -> int:
    # Deprecated: aka encoded in index now
    return 1 if (tile.suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU) and tile.tile_type == TileType.FIVE and tile.aka) else 0


def encode_game_perspective(gp: GamePerspective) -> Dict[str, Any]:
    """Encode a GamePerspective into fixed-size features.

    Breaking change: we now return explicit fields instead of a packed 'game_state'.
    Returns a dict with keys at least:
    - hand_idx: List[int] of length MAX_HAND_LEN + 1, includes the newly drawn tile for any action
    - called_idx: int array shape (4, MAX_CALLED_SETS, MAX_TILES_PER_CALLED_SET) (pads with PAD_IDX)
    - disc_idx: List[List[int]] shape (4, MAX_DISCARDS_PER_PLAYER)
    - called_discards: np.ndarray shape like disc_idx mask for calls
    - dora_indicator_tiles: np.ndarray of length 4 with tile indices (or -1 for pad)
    - round_wind: int (Honor.value)
    - seat_winds: List[int] length 4 (Honor.value per player, relative to perspective)
    - legal_action_mask: List[int] length ACTION_HEAD_SIZE
    - riichi_declarations: List[int] length 4 (discard indices, meaning which index in disc_idx riichi was declared, or -1)
    - remaining_tiles: int
    - owner_of_reactable_tile: int in {-1,0,1,2,3}
    - reactable_tile: int tile index or -1
    - newly_drawn_tile: int tile index or -1
    """
    # Hand
    hand_tiles = list(gp.player_hand)
    # Determine if we're in the acting state (GamePerspective.state is a class from core.game)
    state_is_action = (getattr(gp, 'state', None) is not None) and (
        gp.state is game.Action or getattr(gp.state, '__name__', '') == 'Action'
    )
    hand_vals: List[int] = [int(tile_flat_index(t)) for t in hand_tiles]

    # Always output fixed-size arrays; it is possible to have a 14-tile hand in the action state
    pad_len = (MAX_HAND_LEN + 1) - len(hand_vals)
    if pad_len < 0:
        hand_vals = hand_vals[:MAX_HAND_LEN + 1]
        pad_len = 0
    hand_idx = np.concatenate([np.asarray(hand_vals, dtype=np.int32), np.full((pad_len,), PAD_IDX, dtype=np.int32)])

    # Called tiles per player structured as (sets x tiles-per-set)
    called_rows: List[np.ndarray] = []
    for pid in range(NUM_PLAYERS):
        sets = gp.called_sets.get(pid, [])
        set_mats: List[np.ndarray] = []
        for si in range(MAX_CALLED_SETS):
            if si < len(sets):
                cs = sets[si]
                vals = [int(tile_flat_index(t)) for t in cs.tiles[:MAX_TILES_PER_CALLED_SET]]
                if len(vals) < MAX_TILES_PER_CALLED_SET:
                    vals.extend([PAD_IDX] * (MAX_TILES_PER_CALLED_SET - len(vals)))
            else:
                vals = [PAD_IDX] * MAX_TILES_PER_CALLED_SET
            set_mats.append(np.asarray(vals, dtype=np.int32))
        called_rows.append(np.stack(set_mats, axis=0))
    called_idx = np.stack(called_rows, axis=0) if called_rows else np.zeros((4, MAX_CALLED_SETS, MAX_TILES_PER_CALLED_SET), dtype=np.int32)

    # Discards per player
    disc_rows: List[np.ndarray] = []
    for pid in range(NUM_PLAYERS):
        tiles = gp.player_discards.get(pid, [])[:MAX_DISCARDS_PER_PLAYER]
        vals = [int(tile_flat_index(t)) for t in tiles]
        if len(vals) < MAX_DISCARDS_PER_PLAYER:
            pad = MAX_DISCARDS_PER_PLAYER - len(vals)
            vals.extend([PAD_IDX] * pad)
        elif len(vals) > MAX_DISCARDS_PER_PLAYER:
            vals = vals[:MAX_DISCARDS_PER_PLAYER]
        disc_rows.append(np.asarray(vals, dtype=np.int32))
    disc_idx = np.stack(disc_rows, axis=0) if disc_rows else np.zeros((4, MAX_DISCARDS_PER_PLAYER), dtype=np.int32)

    # Called discards mask per player aligned to disc_idx length
    called_discards_rows: List[np.ndarray] = []
    for pid in range(NUM_PLAYERS):
        mask = np.zeros((MAX_DISCARDS_PER_PLAYER,), dtype=np.int32)
        idxs = gp.called_discards.get(pid, []) if hasattr(gp, 'called_discards') else []
        for j in idxs:
            if 0 <= j < MAX_DISCARDS_PER_PLAYER:
                mask[j] = 1
        called_discards_rows.append(mask)
    called_discards = np.stack(called_discards_rows, axis=0)

    round_wind_val = int(gp.round_wind.value)
    seat_winds_vals = [int(gp.seat_winds[i].value) for i in range(4)]
    # Riichi declaration discard-order indices per player (-1 if not declared)
    riichi_decl_idxs = [int(gp.riichi_declaration_tile.get(i, -1)) for i in range(4)]

    # Dora list for indicators -> indices up to 4
    dora_list = getattr(gp, 'dora_indicators', []) or []
    dora_vals = [int(tile_flat_index(t)) for t in dora_list[:4]]

    # Reactable tile and dora indicators (explicit fields)
    reactable_idx = int(-1)
    if hasattr(gp, '_reactable_tile') and getattr(gp, '_reactable_tile') is not None:
        try:
            reactable_idx = int(tile_flat_index(getattr(gp, '_reactable_tile')))
        except Exception:
            reactable_idx = -1
    # Newly drawn tile index (explicit)
    newly_idx = int(-1)
    if hasattr(gp, 'newly_drawn_tile') and getattr(gp, 'newly_drawn_tile') is not None:
        try:
            newly_idx = int(tile_flat_index(getattr(gp, 'newly_drawn_tile')))
        except Exception:
            newly_idx = -1
    while len(dora_vals) < 4:
        dora_vals.append(-1)
    dora_indicator_tiles = np.asarray(dora_vals[:4], dtype=np.int32)

    # Explicit meta fields to return in place of packed game_state
    lam = gp.legal_action_mask()
    owner_idx = -1 if gp._owner_of_reactable_tile is None else int(gp._owner_of_reactable_tile)
    # Note: store current_player_idx is not directly available from perspective; infer via is_current_turn + players' turns is outside scope.
    out = {
        'hand_idx': hand_idx,
        'called_idx': called_idx,
        'disc_idx': disc_idx,
        'called_discards': called_discards,
        'round_wind': int(gp.round_wind.value),
        'seat_winds': seat_winds_vals, # technically we don't need all seat winds, just one, since they're always in order
        'legal_action_mask': lam,
        'riichi_declarations': riichi_decl_idxs,
        'remaining_tiles': int(gp.remaining_tiles),
        'owner_of_reactable_tile': int(owner_idx),
        'reactable_tile': int(reactable_idx),
        'newly_drawn_tile': int(newly_idx),
        'dora_indicator_tiles': dora_indicator_tiles,
        'deal_in_tiles': gp.deal_in_tiles,
        'wall_count': gp.wall_count
    }
    return out


def decode_game_perspective(features: Dict[str, Any]) -> GamePerspective:
    """Reconstruct a GamePerspective from encoded features.
    This consumes the new explicit fields written by encode_game_perspective() and intentionally
    drops legacy compatibility with packed 'game_state' and 'game_tile_indicators'.
    Note: Called sets are reconstructed per set slot; call_type is set to 'chi' as a placeholder.
    """
    # Fetch explicit arrays
    hand_idx_arr = np.asarray(features['hand_idx'], dtype=np.int32)
    called_idx_arr = np.asarray(features['called_idx'], dtype=np.int32)
    disc_idx_arr = np.asarray(features['disc_idx'], dtype=np.int32)
    called_discards_arr = np.asarray(features['called_discards'], dtype=np.int32)
    round_wind_val = int(features['round_wind'])
    seat_winds_vals = [int(v) for v in np.asarray(features['seat_winds'], dtype=np.int32).tolist()]
    riichi_decl_idxs_vals = [int(v) for v in np.asarray(features['riichi_declarations'], dtype=np.int32).tolist()]
    remaining_tiles = int(features['remaining_tiles'])
    owner_val = int(features.get('owner_of_reactable_tile', -1))
    reactable_idx = int(features.get('reactable_tile', -1))
    newly_idx = int(features.get('newly_drawn_tile', -1))
    dora_idxs = [int(v) for v in np.asarray(features.get('dora_indicator_tiles', []), dtype=np.int32).tolist()]
    wall_count = np.asarray(features.get('wall_count', []), dtype=np.int8)
    # Optional: deal-in tiles mask back to list of Tiles for GamePerspective
    deal_in_tiles = features.get('deal_in_tiles', [])

    # Build basic fields from indices
    player_hand: List[Tile] = []
    for i in hand_idx_arr.tolist():
        if i < 0:
            continue
        t = tile_from_flat_index(int(i))
        player_hand.append(t)
    called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}
    for pid in range(4):
        sets: List[CalledSet] = []
        if called_idx_arr.ndim == 3:
            # Shape expected: (4, MAX_CALLED_SETS, MAX_TILES_PER_CALLED_SET)
            for si in range(min(MAX_CALLED_SETS, called_idx_arr.shape[1])):
                tiles: List[Tile] = []
                for i in called_idx_arr[pid, si].tolist():
                    if int(i) < 0:
                        continue
                    tiles.append(tile_from_flat_index(int(i)))
                if tiles:
                    sets.append(CalledSet(tiles=tiles, call_type='chi', called_tile=None, caller_position=pid, source_position=None))
        else:
            # Fallback: flatten per player if not 3D
            tiles: List[Tile] = []
            for i in called_idx_arr[pid].tolist():
                if int(i) < 0:
                    continue
                tiles.append(tile_from_flat_index(int(i)))
            if tiles:
                sets = [CalledSet(tiles=tiles, call_type='chi', called_tile=None, caller_position=pid, source_position=None)]
        called_sets[pid] = sets
    player_discards: Dict[int, List[Tile]] = {}
    for pid in range(4):
        tiles: List[Tile] = []
        for i in disc_idx_arr[pid].tolist():
            if i < 0:
                continue
            t = tile_from_flat_index(int(i))
            tiles.append(t)
        player_discards[pid] = tiles

    # Reactable tile, newly drawn tile, and owner
    last_discarded_tile = tile_from_flat_index(int(reactable_idx)) if reactable_idx >= 0 else None
    last_discard_player_val = None if owner_val < 0 else int(owner_val)
    newly_drawn_tile = tile_from_flat_index(int(newly_idx)) if newly_idx >= 0 else None

    # Winds
    round_wind = Honor(round_wind_val)
    seat_winds: Dict[int, Honor] = {i: Honor(seat_winds_vals[i]) for i in range(4)}

    # Riichi declaration discard-order index per player (-1 if not declared)
    riichi_declaration_tile: Dict[int, int] = {i: int(riichi_decl_idxs_vals[i]) for i in range(4)}

    # Reconstruct called_discards as dict of lists, bounding indices to available discards length
    called_discards: Dict[int, List[int]] = {}
    for pid in range(4):
        mask = called_discards_arr[pid]
        idxs = [int(i) for i, v in enumerate(mask.tolist()) if int(v) == 1 and i < len(player_discards.get(pid, []))]
        called_discards[pid] = idxs

    # Reconstruct dora indicators from indices if available
    dora_indicators = [tile_from_flat_index(int(i)) for i in dora_idxs if int(i) >= 0]

    gp = GamePerspective(player_hand=player_hand, remaining_tiles=int(remaining_tiles), reactable_tile=last_discarded_tile,
                         owner_of_reactable_tile=last_discard_player_val, called_sets=called_sets,
                         player_discards=player_discards, called_discards=called_discards,
                         newly_drawn_tile=newly_drawn_tile, seat_winds=seat_winds, round_wind=round_wind,
                         dora_indicators=dora_indicators, riichi_declaration_tile=riichi_declaration_tile,
                         wall_count=wall_count, deal_in_tiles=deal_in_tiles)
    return gp

