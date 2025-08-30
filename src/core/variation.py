import random
from typing import List, Dict, Tuple, Optional

from .game import MediumJong, CalledSet
from .tile import Tile, Suit, TileType, Honor, make_tile


# -----------------------------
# Weights for random variation selection
# -----------------------------

# Probability weights for variation selection in create_variation()
SOME_CALLS_WEIGHT: int = 3
CHINITSU_WEIGHT: int = 1
YAKUHAI_WEIGHT: int = 5
DEFENSE_WEIGHT: int = 1


# -----------------------------
# Helpers to assemble custom games
# -----------------------------

class VariationCreationError(Exception):
    """Raised when a requested variation cannot be created (e.g., tile pool exhaustion)."""

def _build_full_pool(tile_copies: int) -> List[Tile]:
    pool: List[Tile] = []
    # Suited tiles with one aka 5 each
    for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
        for v in range(1, 10):
            if v == 5:
                # (tile_copies - 1) normal + 1 aka
                for _ in range(max(tile_copies - 1, 0)):
                    pool.append(Tile(suit, TileType(v)))
                if tile_copies > 0:
                    pool.append(Tile(suit, TileType(v), aka=True))
            else:
                for _ in range(tile_copies):
                    pool.append(Tile(suit, TileType(v)))
    # Honors
    for h in Honor:
        for _ in range(tile_copies):
            pool.append(Tile(Suit.HONORS, h))
    random.shuffle(pool)
    return pool


def _remove_exact_or_functional(pool: List[Tile], target: Tile) -> Tile:
    """Remove a tile from pool matching `target`.
    Prefer exact aka match; otherwise remove any functional equal tile.
    Returns the removed tile (concrete).
    """
    # First try exact
    for i, t in enumerate(pool):
        if t.exactly_equal(target):
            return pool.pop(i)
    # Then functional
    for i, t in enumerate(pool):
        if t.functionally_equal(target):
            return pool.pop(i)
    raise ValueError(f"Tile not available in pool: {target}")


def _allocate_called_set(pool: List[Tile], call_type: str, base_suit: Suit, base_val: int, variant: int = 0,
                         caller: int = 0, source: Optional[int] = None) -> CalledSet:
    """Allocate tiles for a Chi or Pon (or simple Daiminkan) from the pool and
    return a `CalledSet` with fields filled.

    For 'chi': variant 0 means [v, v+1, v+2] sequence starting at base_val.
    For 'pon': three of base tile.
    For 'kan_daimin': four of base tile.
    """
    tiles: List[Tile] = []
    called_tile: Optional[Tile] = None
    if call_type == 'chi':
        if base_suit == Suit.HONORS:
            raise ValueError('Chi cannot be formed with honors')
        seq = [base_val, base_val + 1, base_val + 2]
        if not (1 <= base_val <= 7):
            raise ValueError('Chi base value must be 1..7')
        # Choose the called tile as the middle of the sequence for simplicity
        chosen_called_val = seq[1]
        # Allocate concrete tiles
        for v in seq:
            t = make_tile(base_suit, v)
            tiles.append(_remove_exact_or_functional(pool, t))
        # Mark called tile among allocated tiles
        for t in tiles:
            if t.suit == base_suit and int(t.tile_type.value) == chosen_called_val:
                called_tile = t
                break
        return CalledSet(tiles=tiles, call_type='chi', called_tile=called_tile, caller_position=caller, source_position=source)
    elif call_type == 'pon':
        for _ in range(3):
            t = make_tile(base_suit, base_val)
            tiles.append(_remove_exact_or_functional(pool, t))
        # Choose called tile as one of them
        called_tile = tiles[0]
        return CalledSet(tiles=tiles, call_type='pon', called_tile=called_tile, caller_position=caller, source_position=source)
    elif call_type == 'kan_daimin':
        for _ in range(4):
            t = make_tile(base_suit, base_val)
            tiles.append(_remove_exact_or_functional(pool, t))
        called_tile = tiles[0]
        return CalledSet(tiles=tiles, call_type='kan_daimin', called_tile=called_tile, caller_position=caller, source_position=source)
    else:
        raise ValueError(f"Unsupported call_type: {call_type}")


def _finalize_game_from_pool(game: MediumJong, pool: List[Tile], hands: Dict[int, List[Tile]], called: Dict[int, List[CalledSet]]) -> MediumJong:
    """Given a fresh MediumJong and the remaining pool, assign hands/calls and rebuild walls/dora.
    Ensures each player has 13 tiles in possession (hand + called tiles).
    """
    # Assign called sets first
    for pid in range(4):
        game._player_called_sets[pid] = list(called.get(pid, []))
    # Assign concealed hands, ensuring counts
    for pid in range(4):
        game._player_hands[pid] = list(hands.get(pid, []))
    # Fill up remaining concealed tiles so that hand + called tiles == 13
    for pid in range(4):
        need = 13 - sum(len(cs.tiles) for cs in game._player_called_sets[pid]) - len(game._player_hands[pid])
        if need < 0:
            raise ValueError(f"Player {pid} overfilled: possession exceeds 13 tiles")
        for _ in range(need):
            game._player_hands[pid].append(pool.pop())
    # Rebuild dead wall (14 tiles by default if enough remaining)
    game.dead_wall = []
    dead_n = min(len(pool), 14)
    for _ in range(dead_n):
        game.dead_wall.append(pool.pop())
    # Place dora/ura indicators according to game rules: top (-1) and second (-2)
    game.dora_indicators = []
    game.ura_dora_indicators = []
    if game.dead_wall:
        game.dora_indicators.append(game.dead_wall[-1])
    if len(game.dead_wall) >= 2:
        game.ura_dora_indicators.append(game.dead_wall[-2])
    # Remaining pool becomes the live wall
    game.tiles = pool
    # Reset runtime state
    game.player_discards = {i: [] for i in range(4)}
    game.called_discards = {i: [] for i in range(4)}
    game._reactable_tile = None
    game._owner_of_reactable_tile = None
    game.last_drawn_tile = None
    game._next_move_is_action = True
    game.current_player_idx = 0
    game.game_over = False
    game.winners = []
    game.loser = None
    game.last_discard_was_riichi = False
    game.riichi_declaration_tile = {i: -1 for i in range(4)}
    game.riichi_ippatsu_active = {i: False for i in range(4)}
    game.riichi_sticks_pot = 0
    game.keiten_payments = None
    game.points = None
    # Update oracle waits
    for pid in range(4):
        game.oracle.update_waits_for(game, pid)
    return game


# -----------------------------
# Public API
# -----------------------------

def make_variant_game(players: List) -> MediumJong:
    """Return a fresh standard MediumJong game (baseline variant).
    This is a convenience wrapper in the variations module.
    """
    # Type: accepts list of Player or objects compatible with Player constructor used elsewhere
    return MediumJong(players)


def game_with_random_calls(players: List, min_calls: int = 2, max_calls: int = 6) -> MediumJong:
    """Initialize a game where a random number of open calls (chi/pon) have already occurred
    before the first turn. Total calls distributed across players between min_calls..max_calls.

    Ensures each player has exactly 13 tiles in possession (concealed hand + called tiles).
    """
    game = MediumJong(players)
    # Rebuild from scratch using a fresh pool to avoid coupling with constructor's deal
    pool = _build_full_pool(game.tile_copies)

    # Prepare called sets distribution
    total_calls = random.randint(min_calls, max_calls)
    called_by: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}

    for _ in range(total_calls):
        caller = random.randrange(4)
        # Prefer chi half the time; ensure viable base
        if random.random() < 0.5:
            suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU])
            base = random.randint(1, 7)
            # Allocate chi; source must be left player for a chi, but we only need a plausible id
            source = (caller - 1) % 4
            try:
                cs = _allocate_called_set(pool, 'chi', suit, base, caller=caller, source=source)
            except Exception:
                # Fallback to pon on error (e.g., out of tiles)
                suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU, Suit.HONORS])
                base = random.randint(1, 9 if suit != Suit.HONORS else 7)
                cs = _allocate_called_set(pool, 'pon', suit, base, caller=caller, source=random.randrange(4))
        else:
            suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU, Suit.HONORS])
            base = random.randint(1, 9 if suit != Suit.HONORS else 7)
            cs = _allocate_called_set(pool, 'pon', suit, base, caller=caller, source=random.randrange(4))
        called_by[caller].append(cs)

    # Hands start empty; finalize
    hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
    return _finalize_game_from_pool(game, pool, hands, called_by)


def game_defense_two_tenpai_good_waits(players: List) -> MediumJong:
    """Initialize a game where two distinct players start in tenpai with good (ryanmen) waits.

    Additionally, pre-populate each player's discard pile with 5 random tiles from the live wall.

    Construction strategy for each selected player:
    - 3 complete sequences (9 tiles)
    - 1 pair (2 tiles)
    - 1 open-ended two-tile sequence [v, v+1] with v in 2..8 (ryanmen wait on v-1 or v+2)
    -> total 13 tiles (tenpai) with an open-ended wait.
    """
    game = MediumJong(players)
    pool = _build_full_pool(game.tile_copies)

    # Choose two distinct players to be tenpai
    pA, pB = random.sample(range(4), 2)

    hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
    called: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}

    def allocate_sequence(pool_ref: List[Tile], suit: Suit, start: int, hand_out: List[Tile]) -> None:
        if suit == Suit.HONORS or not (1 <= start <= 7):
            raise ValueError("Invalid sequence parameters")
        for v in (start, start + 1, start + 2):
            hand_out.append(_remove_exact_or_functional(pool_ref, make_tile(suit, v)))

    def allocate_pair(pool_ref: List[Tile], suit: Suit, val: int, hand_out: List[Tile]) -> None:
        t = make_tile(suit, val)
        hand_out.append(_remove_exact_or_functional(pool_ref, t))
        hand_out.append(_remove_exact_or_functional(pool_ref, t))

    def build_tenpai_ryanmen(pid: int) -> None:
        nonlocal pool, hands
        target_suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU])
        base = random.randint(2, 7)  # ryanmen partial [base, base+1], open on (base-1) or (base+2)

        # 3 sequences
        seqs_built = 0
        attempts = 0
        while seqs_built < 3 and attempts < 40:
            s = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU])
            start = random.randint(1, 7)
            try:
                allocate_sequence(pool, s, start, hands[pid])
                seqs_built += 1
            except ValueError:
                attempts += 1
                continue
        if seqs_built < 3:
            raise VariationCreationError("Unable to allocate three sequences for tenpai hand")

        # Pair
        pair_attempts = 0
        while True:
            if pair_attempts > 40:
                raise VariationCreationError("Unable to allocate pair for tenpai hand")
            pair_attempts += 1
            pair_suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU, Suit.HONORS])
            pair_val = random.randint(1, 9 if pair_suit != Suit.HONORS else 7)
            try:
                allocate_pair(pool, pair_suit, pair_val, hands[pid])
                break
            except ValueError:
                continue

        # Partial open-ended sequence [base, base+1] in target_suit
        try:
            hands[pid].append(_remove_exact_or_functional(pool, make_tile(target_suit, base)))
            hands[pid].append(_remove_exact_or_functional(pool, make_tile(target_suit, base + 1)))
        except ValueError as e:
            raise VariationCreationError(f"Unable to allocate ryanmen partial sequence: {e}")

        # Sanity: must be exactly 13 tiles
        if len(hands[pid]) != 13:
            raise VariationCreationError(f"Constructed hand has {len(hands[pid])} tiles, expected 13")

    # Build for both players
    build_tenpai_ryanmen(pA)
    build_tenpai_ryanmen(pB)

    # Finalize game state with current hands (others will be filled to 13)
    game = _finalize_game_from_pool(game, pool, hands, called)

    # Pre-populate 5 random discards for each player by taking from live wall
    for pid in range(4):
        for _ in range(5):
            if not game.tiles:
                raise VariationCreationError("Ran out of tiles while assigning initial discards")
            game.player_discards[pid].append(game.tiles.pop())
        # Update waits cache (optional; discards do not affect waits, but keep fresh)
        game.oracle.update_waits_for(game, pid)

    return game


def game_player0_mono_suit(players: List) -> MediumJong:
    """Initialize a game where player 0 holds only tiles of a single random suit (no honors).
    Player 0's total possession is 13 tiles (all in one suit, possibly including an aka 5).
    Other players are dealt randomly from the remaining pool.
    """
    game = MediumJong(players)
    pool = _build_full_pool(game.tile_copies)

    # Choose a suit for player 0 (exclude honors)
    suit = random.choice([Suit.MANZU, Suit.PINZU, Suit.SOUZU])

    # Reserve 13 tiles from the chosen suit for player 0
    p0_tiles: List[Tile] = []
    # Build a list of candidate tiles (functionally) to request
    # Prefer non-aka fives first to keep aka distribution natural
    values = [1,2,3,4,5,6,7,8,9]
    random.shuffle(values)

    # Keep drawing suited tiles until 13 are reserved
    while len(p0_tiles) < 13:
        v = random.choice(values)
        t = make_tile(suit, v)
        try:
            p0_tiles.append(_remove_exact_or_functional(pool, t))
        except ValueError:
            # If exhausted for some value, try others; break if not enough overall remain
            # As long as tile_copies >= 4, there will be plenty of suited tiles to fill 13
            # If we truly can't fill, fall back to another value iteration
            remaining_same_suit = any(x.suit == suit for x in pool)
            if not remaining_same_suit:
                raise
            continue

    hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
    hands[0] = p0_tiles
    called: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}

    return _finalize_game_from_pool(game, pool, hands, called)


def create_variation(
    players: List,
    some_calls_weight: int = SOME_CALLS_WEIGHT,
    chinitsu_weight: int = CHINITSU_WEIGHT,
    yakuhai_weight: int = YAKUHAI_WEIGHT,
    defense_weight: int = DEFENSE_WEIGHT,
    min_calls: int = 2,
    max_calls: int = 6,
) -> MediumJong:
    """Create a custom game variation using weighted selection.

    When `some_calls_weight` is 1 and `chinitsu_weight` is 2, about 66% of games
    will be the mono-suit (chinitsu-like) setup for player 0.

    Args:
        players: List of players for `MediumJong`.
        some_calls_weight: Weight for selecting the "random calls already made" variant.
        chinitsu_weight: Weight for selecting the "player 0 mono-suit" variant.
        min_calls: Minimum number of called sets across players when selecting the calls variant.
        max_calls: Maximum number of called sets across players when selecting the calls variant.
    """
    weights = [
        max(0, int(some_calls_weight)),
        max(0, int(chinitsu_weight)),
        max(0, int(yakuhai_weight)),
        max(0, int(defense_weight)),
    ]
    if sum(weights) == 0:
        # Fallback to equal likelihood if all zero
        weights = [1, 1, 1, 1]
    # Retry up to 5 times if a variation fails to construct (e.g., tile exhaustion)
    last_err: Optional[Exception] = None
    for _ in range(5):
        choice = random.choices(population=["calls", "chinitsu", "yakuhai", "defense"], weights=weights, k=1)[0]
        try:
            if choice == "calls":
                return game_with_random_calls(players, min_calls=min_calls, max_calls=max_calls)
            if choice == "chinitsu":
                return game_player0_mono_suit(players)
            # yakuhai
            if choice == "yakuhai":
                return game_yakuhai_pons(players)
            # defense
            return game_defense_two_tenpai_good_waits(players)
        except (VariationCreationError, ValueError) as e:
            # Try again with a fresh selection
            last_err = e
            continue
    # Exhausted retries
    raise VariationCreationError(f"Failed to create variation after 5 attempts: {last_err}")


def game_yakuhai_pons(players: List) -> MediumJong:
    """Initialize a game with yakuhai-open setups:
    - One randomly chosen player starts with a Pon of White (Haku).
    - Another distinct player starts with a Pon of East (Ton).

    Remaining tiles are dealt normally to reach 13 tiles of possession per player.
    """
    game = MediumJong(players)
    pool = _build_full_pool(game.tile_copies)

    # Pick two distinct players
    p_white, p_east = random.sample(range(4), 2)

    called_by: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}

    # Allocate Pon of White (Honor.WHITE value is 5 per enum in tile.py ordering)
    try:
        cs_white = _allocate_called_set(
            pool,
            'pon',
            Suit.HONORS,
            int(Honor.WHITE.value),
            caller=p_white,
            source=random.randrange(4),
        )
    except ValueError as e:
        raise VariationCreationError(f"Unable to allocate Pon of White: {e}")
    called_by[p_white].append(cs_white)

    # Allocate Pon of East
    try:
        cs_east = _allocate_called_set(
            pool,
            'pon',
            Suit.HONORS,
            int(Honor.EAST.value),
            caller=p_east,
            source=random.randrange(4),
        )
    except ValueError as e:
        raise VariationCreationError(f"Unable to allocate Pon of East: {e}")
    called_by[p_east].append(cs_east)

    # Finalize with empty explicit hands
    hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
    return _finalize_game_from_pool(game, pool, hands, called_by)
