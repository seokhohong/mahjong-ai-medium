from __future__ import annotations

# Players / seating
NUM_PLAYERS: int = 4
DEALER_ID_START: int = 0

# Tiles
TILE_COPIES_DEFAULT: int = 4
STANDARD_HAND_TILE_COUNT: int = 13
DEAD_WALL_TILES: int = 14  # Riichi dead wall contains 14 tiles
TOTAL_TILES: int = 52 + 70

# Game state vector length used by feature engineering and AC network inputs.
# Structure from feature engineering:
# - Base scalar fields: player_id, remaining_tiles, last_discard_idx, last_discard_player,
#   is_action_state, can_call, last_drawn_idx, round_wind -> 8
# - Seat winds per player: NUM_PLAYERS
# - Riichi flags per player: NUM_PLAYERS
# - Extra legality flags: can_ron, can_tsumo -> 2
GAME_STATE_BASE_FIELDS: int = 8
GAME_STATE_SEAT_WIND_COUNT: int = NUM_PLAYERS
GAME_STATE_RIICHI_FLAGS_COUNT: int = NUM_PLAYERS
GAME_STATE_EXTRA_FLAGS: int = 2
GAME_STATE_VEC_LEN: int = (
    GAME_STATE_BASE_FIELDS
    + GAME_STATE_SEAT_WIND_COUNT
    + GAME_STATE_RIICHI_FLAGS_COUNT
    + GAME_STATE_EXTRA_FLAGS
)

# Dora/Uradora
INITIAL_DORA_INDICATORS: int = 1
INITIAL_URADORA_INDICATORS: int = 1

# Sorting
SUIT_ORDER = {
    'm': 0,  # Manzu
    'p': 1,  # Pinzu
    's': 2,  # Souzu
    'z': 3,  # Honors
}

# Scoring constants (simplified riichi-like)
FU_CHIITOI: int = 25
FU_BASELINE: int = 30
POINTS_ROUNDING: int = 100

# Yaku han values (simplified)
CHANTA_OPEN_HAN: int = 1
CHANTA_CLOSED_HAN: int = 2
JUNCHAN_OPEN_HAN: int = 2
JUNCHAN_CLOSED_HAN: int = 3
SANANKOU_HAN: int = 2
SANSOKU_OPEN_HAN: int = 1  # Sanshoku doujun
SANSOKU_CLOSED_HAN: int = 2
IIPEIKOU_HAN: int = 1      # Closed-only
ITTSU_OPEN_HAN: int = 1    # Pure straight (ittsu)
ITTSU_CLOSED_HAN: int = 2