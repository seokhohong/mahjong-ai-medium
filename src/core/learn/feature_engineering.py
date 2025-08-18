from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np

from core import game
from core.game import GamePerspective, Tile, Suit, TileType, Honor, CalledSet


# Fixed sizes for vectorization
MAX_HAND_LEN: int = 14
MAX_CALLED_TILES_PER_PLAYER: int = 12  # up to four melds => 12 tiles
MAX_DISCARDS_PER_PLAYER: int = 30
PAD_IDX: int = -1  # padding index; 0.. are valid tile indices (including aka slots)


def tile_to_index(tile: Tile) -> int:
    """Map a Tile to a 1..34 index; 0 used for padding.
    Aka 5s map to suit-specific "0" slots per riichi notation: 0m, 0p, 0s.
    Padding uses PAD_IDX (-1).
    """
    if tile.suit == Suit.MANZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 0  # 0m
        return int(tile.tile_type.value)
    if tile.suit == Suit.PINZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 9  # 0p
        return 9 + int(tile.tile_type.value)
    if tile.suit == Suit.SOUZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 18  # 0s
        return 18 + int(tile.tile_type.value)
    # Honors EAST(1)..RED(7) -> 28..34
    return 27 + int(tile.tile_type.value)


def index_to_tile(idx: int) -> Tile:
    """Inverse of tile_to_index (0 returns a placeholder 1m which should be ignored by caller)."""
    if idx < 0:
        return Tile(Suit.MANZU, TileType.ONE)
    # Aka slots
    if idx == 0:
        return Tile(Suit.MANZU, TileType.FIVE, aka=True)
    if idx == 9:
        return Tile(Suit.PINZU, TileType.FIVE, aka=True)
    if idx == 18:
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
    - hand_idx: List[int] of length MAX_HAND_LEN
    - called_idx: List[List[int]] shape (4, MAX_CALLED_TILES_PER_PLAYER)
    - disc_idx: List[List[int]] shape (4, MAX_DISCARDS_PER_PLAYER)
    - game_state: flat list of scalars encoding meta state
    """
    # Hand
    hand_tiles = list(gp.player_hand)
    hand_vals: List[int] = [tile_to_index(t) for t in hand_tiles[:MAX_HAND_LEN]]
    # Always output fixed-size arrays; pad with PAD_IDX when shorter and trim when longer
    if len(hand_vals) < MAX_HAND_LEN:
        pad = MAX_HAND_LEN - len(hand_vals)
        hand_vals.extend([PAD_IDX] * pad)
    elif len(hand_vals) > MAX_HAND_LEN:
        hand_vals = hand_vals[:MAX_HAND_LEN]
    hand_idx = np.asarray(hand_vals, dtype=np.int32)

    # Called tiles per player (flatten tiles only)
    called_rows: List[np.ndarray] = []
    for pid in range(4):
        tiles: List[Tile] = []
        for cs in gp.called_sets.get(pid, []):
            tiles.extend(cs.tiles)
        vals = [tile_to_index(t) for t in tiles[:MAX_CALLED_TILES_PER_PLAYER]]
        if len(vals) < MAX_CALLED_TILES_PER_PLAYER:
            pad = MAX_CALLED_TILES_PER_PLAYER - len(vals)
            vals.extend([PAD_IDX] * pad)
        elif len(vals) > MAX_CALLED_TILES_PER_PLAYER:
            vals = vals[:MAX_CALLED_TILES_PER_PLAYER]
        called_rows.append(np.asarray(vals, dtype=np.int32))
    called_idx = np.stack(called_rows, axis=0) if called_rows else np.zeros((4, MAX_CALLED_TILES_PER_PLAYER), dtype=np.int32)

    # Discards per player
    disc_rows: List[np.ndarray] = []
    for pid in range(4):
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
    for pid in range(4):
        mask = np.zeros((MAX_DISCARDS_PER_PLAYER,), dtype=np.int32)
        idxs = gp.called_discards.get(pid, []) if hasattr(gp, 'called_discards') else []
        for j in idxs:
            if 0 <= j < MAX_DISCARDS_PER_PLAYER:
                mask[j] = 1
        called_discards_rows.append(mask)
    called_discards = np.stack(called_discards_rows, axis=0)

    # Game state vector (compact, fixed-length):
    # [player_id, remaining_tiles, last_discard_idx, last_discard_player,
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
        int(gp.player_id),
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
        'called_discards': called_discards,
    }


def decode_game_perspective(features: Dict[str, Any]) -> GamePerspective:
    """Reconstruct a GamePerspective from encoded features.
    Note: Called sets are reconstructed as a single flattened CalledSet per player when present.
    """
    hand_idx_arr = np.asarray(features['hand_idx'], dtype=np.int32)
    called_idx_arr = np.asarray(features['called_idx'], dtype=np.int32)
    disc_idx_arr = np.asarray(features['disc_idx'], dtype=np.int32)
    gs_arr = np.asarray(features['game_state'], dtype=np.int32)
    called_discards_arr = np.asarray(features.get('called_discards', np.zeros((4, MAX_DISCARDS_PER_PLAYER), dtype=np.int32)), dtype=np.int32)

    # Unpack game state
    player_id = int(gs_arr[0])
    remaining_tiles = int(gs_arr[1])
    last_discard_idx = int(gs_arr[2])
    last_discard_player = int(gs_arr[3])
    state_is_action = bool(int(gs_arr[4]))
    can_call = bool(int(gs_arr[5]))
    last_drawn_idx = int(gs_arr[6])
    round_wind_val = int(gs_arr[7])
    seat_winds_vals = [int(v) for v in gs_arr[8:12].tolist()]
    riichi_flags_vals = [int(v) for v in gs_arr[12:16].tolist()]

    # Build basic fields
    player_hand: List[Tile] = []
    for i in hand_idx_arr.tolist():
        if i < 0:
            continue
        t = index_to_tile(i)
        player_hand.append(t)
    called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}
    for pid in range(4):
        tiles: List[Tile] = []
        for i in called_idx_arr[pid].tolist():
            if i < 0:
                continue
            t = index_to_tile(i)
            tiles.append(t)
        if tiles:
            # Represent as a single synthetic called set (type 'chi') for reconstruction purposes
            called_sets[pid] = [CalledSet(tiles=tiles, call_type='chi', called_tile=None, caller_position=pid, source_position=None)]
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

    # State types
    StateType = game.Action if state_is_action else game.Reaction if hasattr(GamePerspective, 'Reaction') else type('Reaction', (), {})

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
        player_id=int(player_id),
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


