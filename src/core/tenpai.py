from typing import List, Dict, Tuple, Set, Any
from .constants import SUIT_ORDER
from .game import Suit, TileType, Honor, Tile, CalledSet, Riichi

# Cache for meld checking results
_meld_cache: Dict[Tuple[Tuple[int, ...], int], bool] = {}
_standard_hand_cache: Dict[Tuple[int, ...], bool] = {}

# Pre-computed lookup for tile to integer conversion
_tile_to_int_cache: Dict[Tuple[str, int], int] = {}

def _init_tile_cache():
    """Initialize the tile to integer cache"""
    if not _tile_to_int_cache:
        # Use suit symbols to match Suit.value ('m','p','s','z')
        # Manzu 0..8
        for v in range(1, 10):
            _tile_to_int_cache[('m', v)] = 0 + (v - 1)
        # Pinzu 9..17
        for v in range(1, 10):
            _tile_to_int_cache[('p', v)] = 9 + (v - 1)
        # Souzu 18..26
        for v in range(1, 10):
            _tile_to_int_cache[('s', v)] = 18 + (v - 1)
        # Honors 27..33
        for v in range(1, 8):
            _tile_to_int_cache[('z', v)] = 27 + (v - 1)

_init_tile_cache()

def _tile_to_int_fast(tile: Tile) -> int:
    """Convert tile to integer 0-33 for faster operations"""
    suit_val = tile.suit.value  # 'm','p','s','z'
    type_val = int(tile.tile_type.value)
    return _tile_to_int_cache[(suit_val, type_val)]

def _tiles_to_counts_array(tiles) -> List[int]:
    """Convert tiles to count array (34 elements) - fastest version"""
    counts = [0] * 34
    for tile in tiles:
        counts[_tile_to_int_fast(tile)] += 1
    return counts

def _index_to_tile(idx: int) -> Tile:
    """Convert 0..33 index back to Tile."""
    if idx >= 27:
        honor_val = idx - 27 + 1
        return Tile(Suit.HONORS, Honor(honor_val))
    suit_idx = idx // 9
    tile_val = (idx % 9) + 1
    suit = [Suit.MANZU, Suit.PINZU, Suit.SOUZU][suit_idx]
    return Tile(suit, TileType(tile_val))

def _tile_sort_key(t):
    """Keep for compatibility"""
    return (SUIT_ORDER[t.suit.value], int(t.tile_type.value))

def _count_tiles(tiles):
    """Keep for compatibility - returns same format as original"""
    cnt: Dict[Tuple[Suit, int], int] = {}
    for t in tiles:
        key = (t.suit, int(t.tile_type.value))
        cnt[key] = cnt.get(key, 0) + 1
    return cnt

def _quick_reject_tenpai(counts: List[int], target_tiles: int = 14) -> bool:
    """Quick rejection tests for tenpai checking"""
    total = sum(counts)
    
    # Wrong number of tiles
    if total != target_tiles and total != target_tiles - 1:
        return True
    
    # Check for invalid counts
    for c in counts:
        if c > 4:
            return True
    
    # Count pairs and isolated tiles
    pairs = 0
    isolated_honors = 0
    
    # Check honors (can only form pairs/triplets)
    for i in range(27, 34):
        if counts[i] == 1:
            isolated_honors += 1
        elif counts[i] == 2:
            pairs += 1
        elif counts[i] == 4:
            pairs += 2  # Can form 2 pairs
    
    # Too many isolated honors
    if isolated_honors > 2:
        return True
    
    # No pairs at all (need at least one for standard hand)
    if total == target_tiles and pairs == 0:
        # Check if any numbered tiles can form pairs
        has_pair_potential = any(counts[i] >= 2 for i in range(27))
        if not has_pair_potential:
            return True
    
    return False

def _can_form_melds_optimized(counts: List[int], num_melds: int) -> bool:
    """Optimized meld checking with better pruning"""
    if num_melds == 0:
        return all(c == 0 for c in counts)
    
    # Check cache
    cache_key = (tuple(counts), num_melds)
    if cache_key in _meld_cache:
        return _meld_cache[cache_key]
    
    counts = list(counts)  # Make a copy for modification
    
    # Process honors first - they can only form triplets
    for i in range(27, 34):
        if counts[i] % 3 != 0:
            _meld_cache[cache_key] = False
            return False
        num_melds -= counts[i] // 3
        counts[i] = 0
    
    if num_melds <= 0:
        result = num_melds == 0 and all(c == 0 for c in counts[:27])
        _meld_cache[cache_key] = result
        return result
    
    # Try to form melds with number tiles
    def solve_suit(start_idx: int, end_idx: int, remaining_melds: int) -> bool:
        if remaining_melds == 0:
            return all(counts[i] == 0 for i in range(start_idx, end_idx))
        
        for i in range(start_idx, end_idx):
            if counts[i] == 0:
                continue
                
            # Try triplet
            if counts[i] >= 3:
                counts[i] -= 3
                if solve_suit(start_idx, end_idx, remaining_melds - 1):
                    counts[i] += 3
                    return True
                counts[i] += 3
            
            # Try sequence (only valid for tiles 1-7 in suit)
            if i % 9 <= 6 and i + 2 < end_idx and counts[i] > 0 and counts[i+1] > 0 and counts[i+2] > 0:
                counts[i] -= 1
                counts[i+1] -= 1
                counts[i+2] -= 1
                if solve_suit(start_idx, end_idx, remaining_melds - 1):
                    counts[i] += 1
                    counts[i+1] += 1
                    counts[i+2] += 1
                    return True
                counts[i] += 1
                counts[i+1] += 1
                counts[i+2] += 1
            
            # Can't use this tile
            return False
        
        return remaining_melds == 0
    
    # Process each suit independently
    for suit_start in [0, 9, 18]:
        suit_tiles = sum(counts[suit_start:suit_start+9])
        if suit_tiles % 3 != 0:
            _meld_cache[cache_key] = False
            return False
        
        suit_melds = suit_tiles // 3
        if suit_melds > 0:
            if not solve_suit(suit_start, suit_start + 9, suit_melds):
                _meld_cache[cache_key] = False
                return False
            
            num_melds -= suit_melds
            for i in range(suit_start, suit_start + 9):
                counts[i] = 0
    
    result = num_melds == 0
    
    # Cache the result (limit cache size)
    if len(_meld_cache) < 50000:
        _meld_cache[cache_key] = result
    
    return result

def _can_form_melds_concealed(tiles, num_melds: int) -> bool:
    """Compatibility wrapper using optimized version"""
    if num_melds == 0:
        return len(tiles) == 0
    if len(tiles) != 3 * num_melds:
        return False
    
    counts = _tiles_to_counts_array(tiles)
    return _can_form_melds_optimized(counts, num_melds)

def _can_form_standard_hand_counts(counts: List[int]) -> bool:
    """Check if counts form a standard hand (cached)"""
    cache_key = tuple(counts)
    if cache_key in _standard_hand_cache:
        return _standard_hand_cache[cache_key]
    
    # Quick rejection
    if _quick_reject_tenpai(counts, 14):
        result = False
    else:
        result = False
        # Try each possible pair
        for i in range(34):
            if counts[i] >= 2:
                counts[i] -= 2
                if _can_form_melds_optimized(counts, 4):
                    counts[i] += 2
                    result = True
                    break
                counts[i] += 2
    
    # Cache result
    if len(_standard_hand_cache) < 10000:
        _standard_hand_cache[cache_key] = result
    
    return result

def _can_form_standard_hand(tiles) -> bool:
    """Compatibility wrapper using optimized version"""
    if len(tiles) != 14:
        return False
    
    counts = _tiles_to_counts_array(tiles)
    return _can_form_standard_hand_counts(counts)

def hand_is_tenpai_for_tiles(tiles) -> bool:
    """Optimized tenpai checking for exactly 13 tiles"""
    from .game import Suit, TileType, Tile, Honor
    
    if len(tiles) != 13:
        return False
    
    # Convert to counts once
    base_counts = _tiles_to_counts_array(tiles)
    
    # Quick reject
    if _quick_reject_tenpai(base_counts, 13):
        # Check for seven pairs tenpai (6 pairs + 1 single)
        pairs = 0
        single_idx = -1
        for i in range(34):
            if base_counts[i] == 2:
                pairs += 1
            elif base_counts[i] == 1:
                if single_idx != -1:  # More than one single
                    return False
                single_idx = i
            elif base_counts[i] != 0 and base_counts[i] != 3:
                return False
        
        if pairs == 6 and single_idx != -1:
            return True
        
        # Not seven pairs and failed quick reject
        if pairs < 3:  # Need at least some pairs for standard hand
            return False
    
    # Smart candidate generation - only check tiles that could complete melds
    candidates_set: Set[int] = set()
    
    # For each tile we have, check what could complete it
    for i in range(34):
        if base_counts[i] == 0:
            continue
            
        # Can complete a pair
        if base_counts[i] == 1:
            candidates_set.add(i)
        
        # Can complete a triplet
        if base_counts[i] == 2:
            candidates_set.add(i)
        
        # For numbered tiles, check sequences
        if i < 27:
            tile_num = i % 9
            # Patterns using local neighbors
            # If we have consecutive tiles, endpoints can wait
            if tile_num >= 1 and base_counts[i-1] > 0:
                # Having i-1 and i -> candidate i-2 (if valid) and i+1
                if tile_num >= 2:
                    candidates_set.add(i-2)
                if tile_num <= 7:
                    candidates_set.add(i+1)
            if tile_num <= 7 and base_counts[i+1] > 0:
                # Having i and i+1 -> candidate i-1 and i+2
                if tile_num >= 1:
                    candidates_set.add(i-1)
                if tile_num <= 6:
                    candidates_set.add(i+2)
            # If we have a gap of 2 (like 2 and 4), the middle tile completes a sequence
            if tile_num <= 6 and base_counts[i+2] > 0:
                candidates_set.add(i+1)
            if tile_num >= 2 and base_counts[i-2] > 0:
                candidates_set.add(i-1)
    
    # Test each candidate
    test_counts = list(base_counts)
    for idx in candidates_set:
        if test_counts[idx] >= 4:  # Can't add fifth tile
            continue
        
        test_counts[idx] += 1
        if _can_form_standard_hand_counts(test_counts):
            return True
        test_counts[idx] -= 1
    
    # Fallback: chiitoi (seven pairs) tenpai check: 6 pairs + 1 single
    pairs = 0
    singles = 0
    for i in range(34):
        if base_counts[i] == 2:
            pairs += 1
        elif base_counts[i] == 1:
            singles += 1
        elif base_counts[i] not in (0, 3, 4):
            # Having e.g. a count of 5 is invalid and already filtered earlier, but be strict
            return False
    if pairs == 6 and singles == 1:
        return True
    return False

def hand_is_tenpai(hand, called_sets: List[CalledSet] = None) -> bool:
    """Check if a hand is tenpai (optimized). If called_sets provided, consider open hand structure.

    For open hands, chiitoi is invalid; we only check standard hand with calls.
    """
    from .game import Suit, TileType, Tile, Honor
    
    if called_sets:
        # Effective total should be 14 after adding one tile (3 per called meld)
        base_counts = _tiles_to_counts_array(hand)
        # Try each possible addable tile
        for i in range(34):
            if base_counts[i] >= 4:
                continue
            t = _index_to_tile(i)
            if can_complete_standard_with_calls(hand + [t], called_sets):
                return True
        return False
    
    # Closed hand logic
    if len(hand) % 3 != 1:
        return False
    if len(hand) == 13:
        return hand_is_tenpai_for_tiles(hand)
    # For other hand sizes, check all possible tiles
    base_counts = _tiles_to_counts_array(hand)
    # Quick rejection
    if _quick_reject_tenpai(base_counts, len(hand) + 1):
        return False
    # Try each possible tile
    for i in range(34):
        if base_counts[i] >= 4:
            continue
        base_counts[i] += 1
        if _can_form_standard_hand_counts(base_counts):
            base_counts[i] -= 1
            return True
        base_counts[i] -= 1
    return False

def hand_is_tenpai_with_calls(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
    """Explicit API to check tenpai state when there are called sets."""
    return hand_is_tenpai(concealed_tiles, called_sets)

def waits_for_tiles(tiles):
    """Find all tiles that complete the hand (optimized)"""
    from .game import Suit, TileType, Tile, Honor
    
    waits: List[Tile] = []
    if len(tiles) != 13:
        return waits
    
    base_counts = _tiles_to_counts_array(tiles)
    
    # Check for seven pairs wait
    pairs = 0
    single_idx = -1
    for i in range(34):
        if base_counts[i] == 2:
            pairs += 1
        elif base_counts[i] == 1:
            if single_idx == -1:
                single_idx = i
            else:
                single_idx = -2  # Multiple singles
    
    seven_pairs_wait = None
    if pairs == 6 and single_idx >= 0:
        # Seven pairs tenpai - waiting on the single
        seven_pairs_wait = single_idx
    
    # Test each possible tile
    test_counts = list(base_counts)
    for i in range(34):
        if test_counts[i] >= 4:  # Can't add fifth
            continue
        
        test_counts[i] += 1
        is_complete = _can_form_standard_hand_counts(test_counts)
        test_counts[i] -= 1
        
        if is_complete or i == seven_pairs_wait:
            # Convert index back to Tile
            if i >= 27:  # Honor
                honor_val = i - 27 + 1
                waits.append(Tile(Suit.HONORS, Honor(honor_val)))
            else:  # Number tile
                suit_idx = i // 9
                tile_val = (i % 9) + 1
                suit = [Suit.MANZU, Suit.PINZU, Suit.SOUZU][suit_idx]
                waits.append(Tile(suit, TileType(tile_val)))
    
    return waits

# Public function to clear caches if needed
def clear_hand_caches():
    """Clear all caches (useful between games or if memory is a concern)"""
    global _meld_cache, _standard_hand_cache
    _meld_cache.clear()
    _standard_hand_cache.clear()


# Additional helpers ported from game.py to centralize completion logic
def is_chiitoi(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
	# Chiitoitsu only with fully concealed 14 tiles and no calls
	if called_sets:
		return False
	if len(concealed_tiles) != 14:
		return False
	counts = _tiles_to_counts_array(concealed_tiles)
	# Exactly seven pairs
	pair_count = sum(1 for c in counts if c == 2)
	return pair_count == 7


def can_complete_standard_with_calls(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
	"""Return True if concealed + called sets can complete a standard hand.

	Called sets count as completed melds. After removing one pair from concealed tiles,
	the remainder must decompose into the remaining number of melds.
	"""
	called_melds = len(called_sets)
	if called_melds > 4:
		return False
	needed_melds_from_concealed = 4 - called_melds
	# Total effective tiles must be 14
	total_effective = len(concealed_tiles) + 3 * called_melds
	if total_effective != 14:
		return False
	# If no melds needed from concealed, just need a pair in concealed
	counts = _tiles_to_counts_array(concealed_tiles)
	# Try every possible pair position (34 kinds)
	for i in range(34):
		if counts[i] >= 2:
			counts[i] -= 2
			remaining_tiles = sum(counts)
			if remaining_tiles == 3 * needed_melds_from_concealed and _can_form_melds_optimized(counts, needed_melds_from_concealed):
				counts[i] += 2
				return True
			counts[i] += 2
	return False


def legal_riichi_moves(riichi_declared, called_sets, player_hand) -> List[Riichi]:
    """Return all legal Riichi moves by checking tenpai-after-discard directly.

    Requirements per is_legal(Riichi):
    - Action state (assumed by caller)
    - Not already in riichi
    - Hand is closed (no called sets)
    - Discarding the specified tile keeps the hand in tenpai (13 tiles check)
    """
    # Preconditions
    if riichi_declared.get(0, False):
        return []
    if called_sets.get(0, []):
        return []
    # Deduplicate by tile identity and test tenpai-after-discard once per kind
    from .tenpai import hand_is_tenpai_for_tiles as _tenpai_tiles
    hand = list(player_hand)
    results: List[Riichi] = []
    # Build an index of first occurrence per key for efficient removal
    first_idx: Dict[Tuple[Any, Any], int] = {}
    for i, t in enumerate(hand):
        key = (t.suit, t.tile_type)
        if key not in first_idx:
            first_idx[key] = i
    for key, idx in first_idx.items():
        # Remove one tile of this key and test tenpai
        hand_after = hand[:]  # 14 -> 13
        hand_after.pop(idx)
        if _tenpai_tiles(hand_after):
            # Use representative object; aka flag doesn't affect riichi legality
            t_rep = hand[idx]
            results.append(Riichi(t_rep))
    return results