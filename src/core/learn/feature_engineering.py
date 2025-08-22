from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np

from core import game
from core.constants import (
    MAX_CALLS as MAX_CALLED_SETS,
    MAX_CALLED_SET_SIZE as MAX_TILES_PER_CALLED_SET,
    MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
)
from core.game import GamePerspective, Tile, Suit, TileType, Honor, CalledSet


# Fixed sizes for vectorization (aligned with game rules)
# A player's hand should be 13 tiles after actions are resolved.
# During the acting player's turn (pre-discard), the transient hand may be 14.
MAX_HAND_LEN: int = 13
PAD_IDX: int = -1  # padding index; 0.. are valid tile indices (including aka slots)


def tile_to_index(tile: Tile) -> int:
    """Map a Tile to a unique integer index in [1..37].

    Layout (compatible with TILE_INDEX_SIZE=38):
    - 1..9   : Manzu 1..9
    - 10..18 : Pinzu 1..9
    - 19..27 : Souzu 1..9
    - 28..34 : Honors EAST(1)..RED(7)
    - 35     : Manzu aka 5
    - 36     : Pinzu aka 5
    - 37     : Souzu aka 5

    Padding uses PAD_IDX (-1) and is handled by callers.
    """
    if tile.suit == Suit.MANZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 35
        return int(tile.tile_type.value)
    if tile.suit == Suit.PINZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 36
        return 9 + int(tile.tile_type.value)
    if tile.suit == Suit.SOUZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 37
        return 18 + int(tile.tile_type.value)
    # Honors EAST(1)..RED(7) -> 28..34
    return 27 + int(tile.tile_type.value)


def index_to_tile(idx: int) -> Tile:
    """Inverse of tile_to_index.

    For idx <= 0, returns a placeholder tile (ignored by callers in padding/None cases).
    """
    if idx <= 0:
        return Tile(Suit.MANZU, TileType.ONE)
    # Aka slots
    if idx == 35:
        return Tile(Suit.MANZU, TileType.FIVE, aka=True)
    if idx == 36:
        return Tile(Suit.PINZU, TileType.FIVE, aka=True)
    if idx == 37:
        return Tile(Suit.SOUZU, TileType.FIVE, aka=True)
    # Normal suited/honors
    if 1 <= idx <= 9:
        return Tile(Suit.MANZU, TileType(idx))
    if 10 <= idx <= 18:
        return Tile(Suit.PINZU, TileType(idx - 9))
    if 19 <= idx <= 27:
        return Tile(Suit.SOUZU, TileType(idx - 18))
    # 28..34
    return Tile(Suit.HONORS, Honor(idx - 27))


def tile_is_aka(tile: Tile) -> int:
    # Deprecated: aka encoded in index now
    return 1 if (tile.suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU) and tile.tile_type == TileType.FIVE and tile.aka) else 0


def encode_game_perspective(gp: GamePerspective) -> Dict[str, Any]:
    """Encode a GamePerspective into fixed-size features.

    Returns a dict with keys: hand_idx, called_idx, disc_idx, called_discards, game_state.
    - hand_idx: List[int] of length MAX_HAND_LEN + 1, includes the newly drawn tile for any action
    - called_idx: int array shape (4, MAX_CALLED_SETS, MAX_TILES_PER_CALLED_SET)
      (pads unused set slots and tiles with PAD_IDX)
    - disc_idx: List[List[int]] shape (4, MAX_DISCARDS_PER_PLAYER)
    - game_state: flat list of scalars encoding meta state
    - newly_drawn_tile: int index of the newly drawn tile, for any action
    """
    # Hand
    hand_tiles = list(gp.player_hand)
    # Determine if we're in the acting state (GamePerspective.state is a class from core.game)
    state_is_action = (getattr(gp, 'state', None) is not None) and (
        gp.state is game.Action or getattr(gp.state, '__name__', '') == 'Action'
    )
    hand_vals: List[int] = [tile_to_index(t) for t in hand_tiles]

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
                vals = [tile_to_index(t) for t in cs.tiles[:MAX_TILES_PER_CALLED_SET]]
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
        vals = [tile_to_index(t) for t in tiles]
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

    # Game state vector (compact, fixed-length):
    # [remaining_tiles, last_discard_idx, last_discard_player,
    #  is_action_state(0/1), can_call(0/1), last_drawn_idx,
    #  round_wind(1..4), seat_winds[0..3] (1..4), riichi_flags[0..3] (0/1),
    #  can_ron(0/1), can_tsumo(0/1)]
    def tile_idx_or_zero(t):
        return 0 if t is None else tile_to_index(t)

    round_wind_val = int(gp.round_wind.value)
    seat_winds_vals = [int(gp.seat_winds[i].value) for i in range(4)]
    riichi_flags = [1 if gp.riichi_declared.get(i, False) else 0 for i in range(4)]

    # direct featurization risks overfitting on player indices, but we're going to keep it for now to avoid risk of bugs
    can_ron_flag = 1 if gp.can_ron() else 0
    can_tsumo_flag = 1 if gp.can_tsumo() else 0
    game_state_list: List[int] = [
        int(gp.remaining_tiles),
        int(tile_idx_or_zero(gp.last_discarded_tile)),
        -1 if gp.last_discard_player is None else int(gp.last_discard_player),
        int(1 if gp.state is type(gp).Action else 0) if hasattr(gp, 'Action') else int(1 if gp.state.__name__ == 'Action' else 0),
        int(gp.can_call),
        int(tile_idx_or_zero(gp.newly_drawn_tile)),
        round_wind_val,
        *seat_winds_vals,
        *riichi_flags,
        can_ron_flag,
        can_tsumo_flag,
    ]
    game_state = np.asarray(game_state_list, dtype=np.int32)
    # Note: store current_player_idx is not directly available from perspective; infer via is_current_turn + players' turns is outside scope.
    return {
        'hand_idx': hand_idx,
        'called_idx': called_idx,
        'disc_idx': disc_idx,
        'game_state': game_state,
        'called_discards': called_discards
    }


def decode_game_perspective(features: Dict[str, Any]) -> GamePerspective:
    """Reconstruct a GamePerspective from encoded features.
    Note: Called sets are reconstructed as a single flattened CalledSet per player when present.
    """
    hand_idx_arr = np.asarray(features['hand_idx'], dtype=np.int32)
    called_idx_arr = np.asarray(features['called_idx'], dtype=np.int32)
    disc_idx_arr = np.asarray(features['disc_idx'], dtype=np.int32)
    gs_arr = np.asarray(features['game_state'], dtype=np.int32)
    called_discards_arr = np.asarray(features['called_discards'], dtype=np.int32)

    # Unpack game state
    remaining_tiles = int(gs_arr[0])
    last_discard_idx = int(gs_arr[1])
    last_discard_player = int(gs_arr[2])
    state_is_action = bool(int(gs_arr[3]))
    can_call = bool(int(gs_arr[4]))
    last_drawn_idx = int(gs_arr[5])
    round_wind_val = int(gs_arr[6])
    seat_winds_vals = [int(v) for v in gs_arr[7:11].tolist()]
    riichi_flags_vals = [int(v) for v in gs_arr[11:15].tolist()]

    # Build basic fields
    player_hand: List[Tile] = []
    for i in hand_idx_arr.tolist():
        if i < 0:
            continue
        t = index_to_tile(i)
        player_hand.append(t)
    called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}
    for pid in range(4):
        sets: List[CalledSet] = []
        # Support both legacy (4,12) and new (4,4,4) shapes
        if called_idx_arr.ndim == 2:
            tiles: List[Tile] = []
            for i in called_idx_arr[pid].tolist():
                if i < 0:
                    continue
                tiles.append(index_to_tile(int(i)))
            if tiles:
                sets = [CalledSet(tiles=tiles, call_type='chi', called_tile=None, caller_position=pid, source_position=None)]
        else:
            for si in range(min(MAX_CALLED_SETS, called_idx_arr.shape[1])):
                tiles: List[Tile] = []
                for i in called_idx_arr[pid, si].tolist():
                    if int(i) < 0:
                        continue
                    tiles.append(index_to_tile(int(i)))
                if tiles:
                    sets.append(CalledSet(tiles=tiles, call_type='chi', called_tile=None, caller_position=pid, source_position=None))
        called_sets[pid] = sets
    player_discards: Dict[int, List[Tile]] = {}
    for pid in range(4):
        tiles: List[Tile] = []
        for i in disc_idx_arr[pid].tolist():
            if i < 0:
                continue
            t = index_to_tile(i)
            tiles.append(t)
        player_discards[pid] = tiles

    last_discarded_tile = index_to_tile(last_discard_idx) if last_discard_idx > 0 else None
    last_discard_player_val = None if last_discard_player < 0 else int(last_discard_player)
    newly_drawn_tile = index_to_tile(last_drawn_idx) if last_drawn_idx > 0 else None

    # Winds
    round_wind = Honor(round_wind_val)
    seat_winds: Dict[int, Honor] = {i: Honor(seat_winds_vals[i]) for i in range(4)}

    # State types: ensure identity matches core.game classes for `is` checks
    StateType = game.Action if state_is_action else game.Reaction

    # Riichi flags
    riichi_declared: Dict[int, bool] = {i: bool(riichi_flags_vals[i]) for i in range(4)}

    # Reconstruct called_discards as dict of lists, bounding indices to available discards length
    called_discards: Dict[int, List[int]] = {}
    for pid in range(4):
        mask = called_discards_arr[pid]
        idxs = [int(i) for i, v in enumerate(mask.tolist()) if int(v) == 1 and i < len(player_discards.get(pid, []))]
        called_discards[pid] = idxs

    gp = GamePerspective(
        player_hand=player_hand,
        remaining_tiles=int(remaining_tiles),
        last_discarded_tile=last_discarded_tile,
        last_discard_player=last_discard_player_val,
        called_sets=called_sets,
        player_discards=player_discards,
        called_discards=called_discards,
        state=StateType,
        newly_drawn_tile=newly_drawn_tile,
        can_call=can_call,
        seat_winds=seat_winds,
        round_wind=round_wind,
        riichi_declared=riichi_declared,
    )
    return gp


