import random
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from .constants import (
    NUM_PLAYERS as MC_NUM_PLAYERS,
    DEALER_ID_START,
    TILE_COPIES_DEFAULT,
    STANDARD_HAND_TILE_COUNT,
    DEAD_WALL_TILES,
    INITIAL_DORA_INDICATORS,
    INITIAL_URADORA_INDICATORS,
    SUIT_ORDER,
    FU_CHIITOI,
    FU_BASELINE,
    POINTS_ROUNDING,
    CHANTA_OPEN_HAN,
    CHANTA_CLOSED_HAN,
    JUNCHAN_OPEN_HAN,
    JUNCHAN_CLOSED_HAN,
    SANANKOU_HAN,
    SANSOKU_OPEN_HAN,
    SANSOKU_CLOSED_HAN,
    IIPEIKOU_HAN,
    ITTSU_OPEN_HAN,
    ITTSU_CLOSED_HAN, STANDARD_HAND_TILE_COUNT,
)


# MediumJong: Expanded Riichi-like implementation
# - Suits: Manzu (m), Pinzu (p), Souzu (s) and Honors (winds/dragons)
# - Calls: Chi, Pon, Kan (daiminkan, kakan, ankan)
# - Yaku requirement to win; rudimentary scoring (fu/han) with dora/uradora
# - Round/seat winds; player 0 is dealer (East) and East round
# - Riichi declaration; after riichi, only Win/Kan allowed; uradora on riichi win


class Suit(Enum):
    MANZU = 'm'
    PINZU = 'p'
    SOUZU = 's'
    HONORS = 'z'


class TileType(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


class Honor(Enum):
    EAST = 1
    SOUTH = 2
    WEST = 3
    NORTH = 4
    WHITE = 5  # haku
    GREEN = 6  # hatsu
    RED = 7    # chun


@dataclass
class Tile:
    suit: Suit
    tile_type: Union[TileType, Honor]
    aka: bool = False  # red-dora five indicator for suited 5s

    def __str__(self) -> str:
        if self.suit == Suit.HONORS:
            mapping = {
                Honor.EAST: 'E', Honor.SOUTH: 'S', Honor.WEST: 'W', Honor.NORTH: 'N',
                Honor.WHITE: 'P', Honor.GREEN: 'G', Honor.RED: 'R',
            }
            return mapping[self.tile_type]  # type: ignore[index]
        return f"{int(self.tile_type.value)}{self.suit.value}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tile) and self.suit == other.suit and self.tile_type == other.tile_type

    def __hash__(self) -> int:
        return hash((self.suit, self.tile_type))


# Actions and reactions
@dataclass
class Action: ...


@dataclass
class Reaction: ...


@dataclass
class Tsumo(Action): ...


@dataclass
class Ron(Reaction): ...


@dataclass
class Discard(Action):
    tile: Tile


@dataclass
class Riichi(Action):
    # Declare riichi by discarding this tile
    tile: Tile


@dataclass
class Pon(Reaction):
    tiles: List[Tile]


@dataclass
class Chi(Reaction):
    tiles: List[Tile]


@dataclass
class PassCall(Reaction): ...


@dataclass
class KanDaimin(Reaction):
    # Call Kan on a discard with three identical tiles from hand
    tiles: List[Tile]


@dataclass
class KanKakan(Action):
    # Upgrade an existing Pon to Kan using the drawn 4th tile
    tile: Tile


@dataclass
class KanAnkan(Action):
    # Concealed Kan from four tiles in hand
    tile: Tile


@dataclass
class CalledSet:
    tiles: List[Tile]
    call_type: str  # 'chi' | 'pon' | 'kan_daimin' | 'kan_kakan' | 'kan_ankan'
    called_tile: Optional[Tile]
    caller_position: int
    source_position: Optional[int]  # None for ankan/kakan


class InvalidHandStateException(Exception):
    pass


def _is_suited(t: Tile) -> bool:
    return t.suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)


def _tile_sort_key(t: Tile) -> Tuple[int, int]:
    return (SUIT_ORDER[t.suit.value], int(t.tile_type.value))

def _decompose_standard_with_pred(tiles: List[Tile], pred_meld, pred_pair) -> bool:
    """Try to decompose into 4 melds + 1 pair satisfying predicates.

    pred_meld(meld_tiles[3]) -> bool, pred_pair(tile) -> bool
    """
    if len(tiles) != 14:
        return False
    tiles = sorted(list(tiles), key=_tile_sort_key)

    # Count maps
    def build_counts(ts: List[Tile]):
        counts: Dict[Suit, List[int]] = {
            Suit.MANZU: [0] * 10,
            Suit.PINZU: [0] * 10,
            Suit.SOUZU: [0] * 10,
        }
        honors = [0] * 8
        for t in ts:
            if t.suit == Suit.HONORS:
                honors[int(t.tile_type.value)] += 1
            else:
                counts[t.suit][int(t.tile_type.value)] += 1
        return counts, honors

    def make_tile(suit: Suit, val: int) -> Tile:
        return Tile(suit, TileType(val)) if suit != Suit.HONORS else Tile(Suit.HONORS, Honor(val))

    for i in range(len(tiles) - 1):
        a, b = tiles[i], tiles[i + 1]
        if a.suit == b.suit and a.tile_type == b.tile_type and pred_pair(a):
            remaining = tiles[:i] + tiles[i+2:]
            counts, honors = build_counts(remaining)

            def dfs() -> bool:
                # Process suited tiles first
                for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
                    c = counts[suit]
                    for v in range(1, 10):
                        if c[v] > 0:
                            # Triplet
                            if c[v] >= 3:
                                meld = [make_tile(suit, v)] * 3
                                if pred_meld(meld):
                                    c[v] -= 3
                                    if dfs():
                                        return True
                                    c[v] += 3
                            # Sequence
                            if v <= 7 and c[v+1] > 0 and c[v+2] > 0:
                                meld = [make_tile(suit, v), make_tile(suit, v+1), make_tile(suit, v+2)]
                                if pred_meld(meld):
                                    c[v] -= 1; c[v+1] -= 1; c[v+2] -= 1
                                    if dfs():
                                        return True
                                    c[v] += 1; c[v+1] += 1; c[v+2] += 1
                            return False
                # Honors must form triplets and satisfy pred
                for hv in range(1, 8):
                    if honors[hv] > 0:
                        if honors[hv] >= 3:
                            meld = [make_tile(Suit.HONORS, hv)] * 3
                            if not pred_meld(meld):
                                return False
                            honors[hv] -= 3
                            if dfs():
                                return True
                            honors[hv] += 3
                        return False
                return True

            if dfs():
                return True
    return False


def _is_chanta(all_tiles: List[Tile]) -> bool:
    # All sets (including pair) contain terminal or honor; at least one honor or terminal in each
    def pred_meld(meld: List[Tile]) -> bool:
        if any(t.suit == Suit.HONORS for t in meld):
            return True
        vals = [int(t.tile_type.value) for t in meld]
        return min(vals) == 1 or max(vals) == 9
    def pred_pair(tile: Tile) -> bool:
        return tile.suit == Suit.HONORS or int(tile.tile_type.value) in (1, 9)
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _is_junchan(all_tiles: List[Tile]) -> bool:
    # All sets (including pair) contain terminals only; no honors
    def pred_meld(meld: List[Tile]) -> bool:
        if any(t.suit == Suit.HONORS for t in meld):
            return False
        vals = [int(t.tile_type.value) for t in meld]
        return min(vals) == 1 or max(vals) == 9
    def pred_pair(tile: Tile) -> bool:
        return tile.suit != Suit.HONORS and int(tile.tile_type.value) in (1, 9)
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _count_sanankou(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> int:
    # Count concealed triplets in hand plus any concealed kans
    cnt = _count_tiles(concealed_tiles)
    triples = sum(1 for c in cnt.values() if c >= 3)
    triples += sum(1 for cs in called_sets if cs.call_type == 'kan_ankan')
    return triples


def _has_sanshoku_sequences(all_tiles: List[Tile]) -> bool:
    # Check for three identical sequences across suits (man, pin, sou)
    # Build counts per suit
    suit_vals = {
        Suit.MANZU: [0] * 10,
        Suit.PINZU: [0] * 10,
        Suit.SOUZU: [0] * 10,
    }
    for t in all_tiles:
        if t.suit in suit_vals:
            suit_vals[t.suit][int(t.tile_type.value)] += 1
    # For every start 1..7, see if each suit has at least one of v,v+1,v+2
    for v in range(1, 8):
        if all(suit_vals[s][v] >= 1 and suit_vals[s][v+1] >= 1 and suit_vals[s][v+2] >= 1 for s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)):
            return True
    return False


def _has_iipeikou(concealed_tiles: List[Tile]) -> bool:
    # Closed-only: one pair of identical sequences in the same suit
    # Count sequences in each suit; if any sequence can be formed twice disjointly (i,i+1,i+2) from counts, we consider it present
    suit_vals = {
        Suit.MANZU: [0] * 10,
        Suit.PINZU: [0] * 10,
        Suit.SOUZU: [0] * 10,
    }
    for t in concealed_tiles:
        if t.suit in suit_vals:
            suit_vals[t.suit][int(t.tile_type.value)] += 1
    for s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
        c = list(suit_vals[s])
        for v in range(1, 8):
            if c[v] >= 2 and c[v+1] >= 2 and c[v+2] >= 2:
                return True
    return False

def _has_ittsu(all_tiles: List[Tile]) -> bool:
    """Pure straight: 1-9 straight within a single suit (1-2-3, 4-5-6, 7-8-9).
    We detect presence of at least one of each segment in the same suit.
    """
    suit_vals = {
        Suit.MANZU: [0] * 10,
        Suit.PINZU: [0] * 10,
        Suit.SOUZU: [0] * 10,
    }
    for t in all_tiles:
        if t.suit in suit_vals:
            suit_vals[t.suit][int(t.tile_type.value)] += 1
    for s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
        c = suit_vals[s]
        if all(c[v] >= 1 for v in (1,2,3)) and all(c[v] >= 1 for v in (4,5,6)) and all(c[v] >= 1 for v in (7,8,9)):
            return True
    return False

def _is_chi_possible_with(hand: List[Tile], target: Tile) -> List[List[Tile]]:
    options: List[List[Tile]] = []
    if target.suit == Suit.HONORS:
        return options
    s = target.suit
    v = int(target.tile_type.value)
    def has(val: int) -> Optional[Tile]:
        for t in hand:
            if t.suit == s and int(t.tile_type.value) == val:
                return t
        return None
    # (v-2, v-1)
    if v - 2 >= 1 and v - 1 >= 1:
        a = has(v-2); b = has(v-1)
        if a and b:
            options.append([a, b])
    # (v-1, v+1)
    if v - 1 >= 1 and v + 1 <= 9:
        a = has(v-1); b = has(v+1)
        if a and b:
            options.append([a, b])
    # (v+1, v+2)
    if v + 1 <= 9 and v + 2 <= 9:
        a = has(v+1); b = has(v+2)
        if a and b:
            options.append([a, b])
    return options


def _count_tiles(tiles: List[Tile]) -> Dict[Tuple[Suit, int], int]:
    cnt: Dict[Tuple[Suit, int], int] = {}
    for t in tiles:
        key = (t.suit, int(t.tile_type.value))
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def _dora_next(tile: Tile) -> Tile:
    # Next tile cycling within suit/honors for dora mapping
    if tile.suit == Suit.HONORS:
        order = [Honor.EAST, Honor.SOUTH, Honor.WEST, Honor.NORTH, Honor.WHITE, Honor.GREEN, Honor.RED]
        idx = order.index(tile.tile_type)  # type: ignore[arg-type]
        return Tile(Suit.HONORS, order[(idx + 1) % len(order)])
    v = int(tile.tile_type.value)
    nv = 1 if v == 9 else v + 1
    return Tile(tile.suit, TileType(nv))


def hand_is_tenpai_for_tiles(tiles: List[Tile]) -> bool:
    # Use extracted implementation
    from .tenpai import hand_is_tenpai_for_tiles as _tenpai_tiles
    return _tenpai_tiles(tiles)


def hand_is_tenpai(hand: List[Tile]) -> bool:
    from .tenpai import hand_is_tenpai as _tenpai
    # Closed hand path preserved; game-level exhaustive draw checks tenpai without calls
    return _tenpai(hand)


def _calc_dora_han(hand_tiles: List[Tile], called_sets: List[CalledSet], indicators: List[Tile]) -> int:
    all_tiles: List[Tile] = []
    all_tiles.extend(hand_tiles)
    for cs in called_sets:
        all_tiles.extend(cs.tiles)
    dora_tiles = [_dora_next(ind) for ind in indicators]
    return sum(1 for t in all_tiles for d in dora_tiles if t.suit == d.suit and t.tile_type == d.tile_type)


def _count_aka_han(all_tiles: List[Tile]) -> int:
    # Aka dora: each red five is worth 1 han
    return sum(1 for t in all_tiles if t.aka)


def _is_tanyao(all_tiles: List[Tile]) -> bool:
    for t in all_tiles:
        if t.suit == Suit.HONORS:
            return False
        v = int(t.tile_type.value)
        if v == 1 or v == 9:
            return False
    return True


def _is_chiitoi(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
    from .tenpai import is_chiitoi as _chiitoi
    return _chiitoi(concealed_tiles, called_sets)


def _count_triplet_value(cnt: Dict[Tuple[Suit, int], int], suit: Suit, val: int) -> int:
    return 1 if cnt.get((suit, val), 0) >= 3 else 0


def _is_toitoi(all_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
    # All groups are triplets/kan and a pair
    # Approx: if no suited sequences in called sets and counts of each suited number are multiples of 0,2,3,4 and number of numbers used in sequences is 0.
    for cs in called_sets:
        if cs.call_type == 'chi':
            return False
    # Very rough check: if standard-formable and there is no way to take a sequence from suited counts
    # We'll simply detect presence of any three-in-a-row suited numbers as evidence against toitoi
    cnt = _count_tiles(all_tiles)
    for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
        for v in range(1, 8):
            if cnt.get((suit, v), 0) > 0 and cnt.get((suit, v+1), 0) > 0 and cnt.get((suit, v+2), 0) > 0:
                return False
    return True


def _is_honitsu(all_tiles: List[Tile]) -> bool:
    suits = {t.suit for t in all_tiles if t.suit != Suit.HONORS}
    has_honors = any(t.suit == Suit.HONORS for t in all_tiles)
    return has_honors and len(suits) == 1


def _is_chinitsu(all_tiles: List[Tile]) -> bool:
    suits = {t.suit for t in all_tiles if t.suit != Suit.HONORS}
    has_honors = any(t.suit == Suit.HONORS for t in all_tiles)
    return not has_honors and len(suits) == 1


def _yakuhai_han(concealed_tiles: List[Tile], called_sets: List[CalledSet], seat_wind: Honor, round_wind: Honor) -> int:
    # 1 han per dragon triplet; 1 han for seat wind triplet; 1 for round wind triplet
    cnt = _count_tiles(concealed_tiles + [t for cs in called_sets for t in cs.tiles])
    han = 0
    # Dragons
    for h in (Honor.WHITE, Honor.GREEN, Honor.RED):
        han += _count_triplet_value(cnt, Suit.HONORS, int(h.value))
    # Seat wind
    han += _count_triplet_value(cnt, Suit.HONORS, int(seat_wind.value))
    # Round wind
    han += _count_triplet_value(cnt, Suit.HONORS, int(round_wind.value))
    return han


def _is_open_hand(called_sets: List[CalledSet]) -> bool:
    return any(cs.call_type in ('chi', 'pon', 'kan_daimin', 'kan_kakan') for cs in called_sets)


def _is_pinfu(all_tiles: List[Tile], called_sets: List[CalledSet], seat_wind: Honor, round_wind: Honor) -> bool:
    # Closed-only; all sequences; pair not honors or seat/round
    if _is_open_hand(called_sets):
        return False
    def pred_meld(meld: List[Tile]) -> bool:
        # Reject triplets (all same value)
        a, b, c = meld
        return not (a.suit == b.suit == c.suit and int(a.tile_type.value) == int(b.tile_type.value) == int(c.tile_type.value))
    def pred_pair(tile: Tile) -> bool:
        if tile.suit == Suit.HONORS:
            return False
        # Exclude seat/round wind as pair
        return True
    # If any honors present, _decompose_standard_with_pred will fail because honors only form triplets
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _score_fu_and_han(concealed_tiles: List[Tile], called_sets: List[CalledSet],
                      winner_id: int, dealer_id: int, win_by_tsumo: bool,
                      riichi_declared: bool, seat_wind: Honor, round_wind: Honor,
                      dora_indicators: List[Tile], ura_indicators: List[Tile]) -> Tuple[int, int, int]:
    # Returns (fu, han, han_from_dora)
    all_tiles = concealed_tiles + [t for cs in called_sets for t in cs.tiles]

    # Yaku detection (subset, enough for tests)
    han = 0
    chiitoi = _is_chiitoi(concealed_tiles, called_sets)
    if chiitoi:
        han += 2
        fu = FU_CHIITOI
    else:
        # Standard hand requires 4 melds + pair; if not formable, no win
        fu = FU_BASELINE
        if _is_tanyao(all_tiles):
            han += 1
        if _is_toitoi(all_tiles, called_sets):
            han += 2
        # Honitsu/Chinitsu; open/closed handled roughly by presence of chi/pon (open)
        open_hand = _is_open_hand(called_sets)
        if _is_honitsu(all_tiles):
            han += 2 if open_hand else 3
        if _is_chinitsu(all_tiles):
            han += 5 if open_hand else 6
        # Chanta/Junchan
        if _is_chanta(concealed_tiles + [t for cs in called_sets for t in cs.tiles]):
            han += CHANTA_OPEN_HAN if open_hand else CHANTA_CLOSED_HAN
        if _is_junchan(concealed_tiles + [t for cs in called_sets for t in cs.tiles]):
            han += JUNCHAN_OPEN_HAN if open_hand else JUNCHAN_CLOSED_HAN
        # Yakuhai
        han += _yakuhai_han(concealed_tiles, called_sets, seat_wind, round_wind)
        # Sanankou
        if _count_sanankou(concealed_tiles, called_sets) >= 3:
            han += SANANKOU_HAN
        # Sanshoku doujun
        # Sanshoku doujun: counts for both closed and open hands
        if _has_sanshoku_sequences(all_tiles):
            han += SANSOKU_OPEN_HAN if open_hand else SANSOKU_CLOSED_HAN
        # Ittsu (pure straight): open/closed
        if _has_ittsu(all_tiles):
            han += ITTSU_OPEN_HAN if open_hand else ITTSU_CLOSED_HAN
        # Iipeikou (closed only)
        if not open_hand and _has_iipeikou(concealed_tiles):
            han += IIPEIKOU_HAN
        # Pinfu
        if _is_pinfu(all_tiles, called_sets, seat_wind, round_wind):
            han += 1

    # Riichi
    if riichi_declared:
        han += 1
    # Menzen (menzen tsumo): closed hand tsumo
    if win_by_tsumo and not _is_open_hand(called_sets):
        han += 1

    # Dora (including aka)
    dora_han = _calc_dora_han(concealed_tiles, called_sets, dora_indicators)
    if riichi_declared:
        dora_han += _calc_dora_han(concealed_tiles, called_sets, ura_indicators)
    # Count red-5 (aka) as dora
    han += dora_han + _count_aka_han(all_tiles)

    return fu, han, dora_han


class GamePerspective:
    def __init__(self,
                 player_hand: List[Tile],
                 player_id: int,
                 remaining_tiles: int,
                 last_discarded_tile: Optional[Tile],
                 last_discard_player: Optional[int],
                 called_sets: Dict[int, List[CalledSet]],
                 player_discards: Dict[int, List[Tile]],
                 called_discards: Dict[int, List[int]],
                 state: type,
                 newly_drawn_tile: Optional[Tile],
                 can_call: bool,
                 seat_winds: Dict[int, Honor],
                 round_wind: Honor,
                 riichi_declared: Dict[int, bool],
                 ) -> None:
        self.player_hand = sorted(list(player_hand), key=_tile_sort_key)
        self.player_id = player_id
        self.remaining_tiles = remaining_tiles
        self.last_discarded_tile = last_discarded_tile
        self.last_discard_player = last_discard_player
        self.called_sets = {pid: list(sets) for pid, sets in called_sets.items()}
        self.player_discards = {pid: list(ts) for pid, ts in player_discards.items()}
        self.called_discards = {pid: list(idxs) for pid, idxs in called_discards.items()}
        self.state = state
        self.newly_drawn_tile = newly_drawn_tile
        self.can_call = can_call
        self.seat_winds = dict(seat_winds)
        self.round_wind = round_wind
        self.riichi_declared = dict(riichi_declared)

    def _concealed_tiles(self) -> List[Tile]:
        return list(self.player_hand)

    def _has_yaku_if_complete(self) -> bool:
        # Simple heuristic: evaluate yaku on this hand if it is standard or chiitoi
        ct = self._concealed_tiles()
        cs = self.called_sets.get(self.player_id, [])
        if _is_chiitoi(ct, cs):
            return True
        from .tenpai import can_complete_standard_with_calls
        if can_complete_standard_with_calls(ct, cs):
            all_tiles = ct + [t for s in cs for t in s.tiles]
            if _is_tanyao(all_tiles):
                return True
            if _is_toitoi(all_tiles, cs):
                return True
            if _is_honitsu(all_tiles) or _is_chinitsu(all_tiles):
                return True
            if _is_chanta(all_tiles) or _is_junchan(all_tiles):
                return True
            if _yakuhai_han(ct, cs, self.seat_winds[self.player_id], self.round_wind) > 0:
                return True
        return False

    def can_tsumo(self) -> bool:
        if self.newly_drawn_tile is None:
            return False
        # Require yaku
        # Evaluate win using current hand (already includes the drawn tile)
        ct = list(self.player_hand)
        cs = self.called_sets.get(self.player_id, [])
        # Hand must be complete in standard or chiitoi form
        from .tenpai import can_complete_standard_with_calls
        complete = _is_chiitoi(ct, cs) or can_complete_standard_with_calls(ct, cs)
        if not complete:
            return False
        # Check yaku presence (menzen tsumo counts as yaku for closed hands)
        all_tiles = ct + [t for s in cs for t in s.tiles]
        if _is_tanyao(all_tiles) or _is_toitoi(all_tiles, cs) or _is_honitsu(all_tiles) or _is_chinitsu(all_tiles) or _is_chanta(all_tiles) or _is_junchan(all_tiles):
            return True
        if len(cs) == 0 and _has_iipeikou(ct):
            return True
        if _has_sanshoku_sequences(all_tiles) or _has_ittsu(all_tiles):
            return True
        if _yakuhai_han(ct, cs, self.seat_winds[self.player_id], self.round_wind) > 0:
            return True
        # Menzen tsumo yaku
        if not _is_open_hand(cs):
            return True
        return False

    def can_ron(self) -> bool:
        if self.last_discarded_tile is None or self.last_discard_player == self.player_id:
            return False
        # Furiten blocks ron
        if self._is_furiten():
            return False
        return self._win_possible(require_yaku=True, include_last_discard=True)

    def _win_possible(self, require_yaku: bool, include_last_discard: bool = False) -> bool:
        ct = list(self.player_hand)
        if include_last_discard and self.last_discarded_tile is not None:
            ct = ct + [self.last_discarded_tile]
        cs = self.called_sets.get(self.player_id, [])
        ok = False
        if _is_chiitoi(ct, cs):
            ok = True
        # For standard hand, consider called sets when checking completeness
        from .tenpai import can_complete_standard_with_calls
        if can_complete_standard_with_calls(ct, cs):
            ok = True
        # No extra fallback; standard completion already considered via ct (which may include drawn tile)
        if not ok:
            return False
        if not require_yaku:
            return True
        # Check presence of at least one yaku
        all_tiles = ct + [t for s in cs for t in s.tiles]
        if _is_tanyao(all_tiles):
            return True
        if _is_toitoi(all_tiles, cs):
            return True
        if _is_honitsu(all_tiles) or _is_chinitsu(all_tiles):
            return True
        if _is_chanta(all_tiles) or _is_junchan(all_tiles):
            return True
        # Closed-only ii-peikou as a yaku to satisfy win condition
        if len(cs) == 0 and _has_iipeikou(ct):
            return True
        # Sanshoku sequences available open/closed contribute yaku to win condition
        if _has_sanshoku_sequences(all_tiles):
            return True
        # Ittsu contributes yaku for win
        if _has_ittsu(all_tiles):
            return True
        if _yakuhai_han(ct, cs, self.seat_winds[self.player_id], self.round_wind) > 0:
            return True
        # Pinfu (closed only) counts as yaku for win condition
        if not _is_open_hand(cs) and _is_pinfu(all_tiles, cs, self.seat_winds[self.player_id], self.round_wind):
            return True
        if _is_chiitoi(ct, cs):
            return True
        # Menzen tsumo: self-draw with no open-hand calls satisfies yaku requirement
        if (not include_last_discard) and (self.newly_drawn_tile is not None) and (not _is_open_hand(cs)):
            return True
        return False

    def _waits(self) -> List[Tile]:
        # Delegate to extracted waits function for consistency and speed
        from .tenpai import waits_for_tiles
        return waits_for_tiles(list(self.player_hand))

    def _is_furiten(self) -> bool:
        # If any wait tile is in own discards, furiten applies
        waits = self._waits()
        own_discards = self.player_discards.get(self.player_id, [])
        for w in waits:
            if any(d.suit == w.suit and d.tile_type == w.tile_type for d in own_discards):
                return True
        return False

    def _tile_flat_index(self, tile: Tile) -> int:
        """Map a Tile to a compact 0..34 index for action masks.
        Suited aka 5s are mapped to 0m/0p/0s as 0, 9, 18 respectively.
        Honors are 28..34 (E,S,W,N,P,G,R).
        """
        if tile.suit == Suit.MANZU:
            if tile.tile_type == TileType.FIVE and tile.aka:
                return 0
            return int(tile.tile_type.value)
        if tile.suit == Suit.PINZU:
            if tile.tile_type == TileType.FIVE and tile.aka:
                return 9
            return 9 + int(tile.tile_type.value)
        if tile.suit == Suit.SOUZU:
            if tile.tile_type == TileType.FIVE and tile.aka:
                return 18
            return 18 + int(tile.tile_type.value)
        # Honors 28..34
        return 27 + int(tile.tile_type.value)

    def _chi_variant_index(self, last_discarded_tile: Optional[Tile], tiles: List[Tile]) -> int:
        # 0: [d-2, d-1], 1: [d-1, d+1], 2: [d+1, d+2]; -1 otherwise
        if last_discarded_tile is None or len(tiles) < 2:
            return -1
        if last_discarded_tile.suit == Suit.HONORS:
            return -1
        d = int(last_discarded_tile.tile_type.value)
        ranks = sorted(int(t.tile_type.value) for t in tiles)
        if ranks == [d - 2, d - 1]:
            return 0
        if ranks == [d - 1, d + 1]:
            return 1
        if ranks == [d + 1, d + 2]:
            return 2
        return -1

    def legal_flat_mask(self) -> List[int]:
        """Return a flat 0/1 mask indicating legal actions in a fixed action space.

        Layout (length 152):
        - 0..34: Discard by tile index (0..34)
        - 35..69: Riichi by discard tile index (0..34)
        - 70: Tsumo
        - 71: Ron
        - 72..77: Chi variants without aka [low, mid, high] then with aka [low, mid, high]
        - 78..79: Pon [no-aka, with-aka]
        - 80: Kan (daiminkan) on last discard
        - 81..115: Kan Kakan by tile index (0..34)
        - 116..150: Kan Ankan by tile index (0..34)
        - 151: Pass (only meaningful in reaction state)
        """
        TOTAL = 152
        OFF_DISCARD = 0
        OFF_RIICHI = 35
        OFF_TSUMO = 70
        OFF_RON = 71
        OFF_CHI = 72            # 0..2 no-aka, 3..5 with-aka
        OFF_PON_NOAKA = 78
        OFF_PON_AKA = 79
        OFF_KAN_DAIMIN = 80
        OFF_KAKAN = 81
        OFF_ANKAN = 116
        OFF_PASS = 151

        mask = [0] * TOTAL

        # Tsumo
        if self.is_legal(Tsumo()):
            mask[OFF_TSUMO] = 1
        # Ron
        if self.can_ron():
            mask[OFF_RON] = 1

        # Discards
        seen: set = set()
        for t in self.player_hand:
            idx = self._tile_flat_index(t)
            if idx in seen:
                continue
            seen.add(idx)
            if self.is_legal(Discard(t)):
                mask[OFF_DISCARD + idx] = 1

        # Riichi (parameterized by discard tile)
        for m in self.legal_moves():
            if isinstance(m, Riichi):
                idx = self._tile_flat_index(m.tile)
                mask[OFF_RIICHI + idx] = 1

        # Kakan/Ankan opportunities
        seen_kakan: set = set()
        seen_ankan: set = set()
        for t in self.player_hand:
            idx = self._tile_flat_index(t)
            if idx not in seen_kakan and self.is_legal(KanKakan(t)):
                mask[OFF_KAKAN + idx] = 1
                seen_kakan.add(idx)
            if idx not in seen_ankan and self.is_legal(KanAnkan(t)):
                mask[OFF_ANKAN + idx] = 1
                seen_ankan.add(idx)

        # Reaction options: Chi/Pon/KanDaimin/Pass
        if self.state is Reaction and self.last_discarded_tile is not None and self.last_discard_player != self.player_id:
            opts = self.get_call_options()
            # Chi variants
            for pair in opts.get('chi', []):
                v = self._chi_variant_index(self.last_discarded_tile, pair)
                if 0 <= v <= 2:
                    # Aka if any of the three tiles is aka
                    aka = any(t.aka for t in (pair + [self.last_discarded_tile]))
                    offset = OFF_CHI + v + (3 if aka else 0)
                    mask[offset] = 1
            # Pon
            for pon_pair in opts.get('pon', []):
                aka = any(t.aka for t in (pon_pair + [self.last_discarded_tile]))
                if aka:
                    mask[OFF_PON_AKA] = 1
                else:
                    mask[OFF_PON_NOAKA] = 1
            # Daiminkan
            if opts.get('kan_daimin'):
                mask[OFF_KAN_DAIMIN] = 1
            # Pass
            if self.is_legal(PassCall()):
                mask[OFF_PASS] = 1

        return mask

    def legal_flat_mask_np(self):
        """Return the legal actions mask as a numpy 1D array of 0/1 with length 152.

        This mirrors legal_flat_mask but returns an np.ndarray for downstream models.
        """
        import numpy as _np  # local import to avoid hard dependency at module import time
        return _np.asarray(self.legal_flat_mask(), dtype=_np.float64)

    def discard_is_called(self, player_id: int, discard_index: int) -> bool:
        return discard_index in self.called_discards.get(player_id, [])

    def get_call_options(self) -> Dict[str, List[List[Tile]]]:
        options = {'pon': [], 'chi': [], 'kan_daimin': []}  # kan_daimin: react to discard
        last = self.last_discarded_tile
        lp = self.last_discard_player
        if last is None or lp is None or lp == self.player_id:
            return options
        hand = list(self.player_hand)
        # Pon
        same = [t for t in hand if t.suit == last.suit and t.tile_type == last.tile_type]
        if len(same) >= 2:
            options['pon'].append([same[0], same[1]])
        # Chi (left player only)
        if self.player_id == (lp + 1) % 4 and last.suit != Suit.HONORS:
            for pair in _is_chi_possible_with(hand, last):
                options['chi'].append(pair)
        # Daiminkan: need three in hand
        if len(same) >= 3:
            options['kan_daimin'].append([same[0], same[1], same[2]])
        return options

    def is_legal(self, move: Union[Action, Reaction]) -> bool:
        # Riichi restriction: after declaring, only Tsumo or Kan actions are allowed on own turn
        riichi_locked = self.riichi_declared.get(self.player_id, False)
        if isinstance(move, (Tsumo, Discard, Riichi, KanKakan, KanAnkan)):
            if self.state is not Action:
                return False
            if isinstance(move, Tsumo):
                return self.can_tsumo()
            if isinstance(move, Riichi):
                # Closed hand, not already in riichi, and discarding specified tile keeps tenpai
                if self.riichi_declared.get(self.player_id, False):
                    return False
                if self.called_sets.get(self.player_id, []):
                    return False
                if move.tile not in self.player_hand:
                    return False
                # Check tenpai after discarding this tile (13 tiles state)
                from .tenpai import hand_is_tenpai_for_tiles as _tenpai_tiles
                hand_after = list(self.player_hand)
                hand_after.remove(move.tile)
                return _tenpai_tiles(hand_after)
            if isinstance(move, KanKakan):
                # Must have an existing pon of this tile
                for cs in self.called_sets.get(self.player_id, []):
                    if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].suit == move.tile.suit and cs.tiles[0].tile_type == move.tile.tile_type:
                        return move.tile in self.player_hand
                return False
            if isinstance(move, KanAnkan):
                # Need four in hand
                cnt = sum(1 for t in self.player_hand if t.suit == move.tile.suit and t.tile_type == move.tile.tile_type)
                return cnt >= 4
            if isinstance(move, Discard):
                if riichi_locked:
                    # Can only discard the newly drawn tile when riichi is locked
                    return self.newly_drawn_tile is not None and move.tile == self.newly_drawn_tile
                return move.tile in self.player_hand
            return False

        # Reactions
        if self.state is not Reaction:
            return False
        # Riichi restriction on reactions: cannot Chi/Pon/KanDaimin after declaring Riichi
        if riichi_locked and isinstance(move, (Pon, Chi, KanDaimin)):
            return False
        if isinstance(move, Ron):
            return self.can_ron()
        if isinstance(move, PassCall):
            opts = self.get_call_options()
            return self.can_ron() or bool(opts['pon'] or opts['chi'] or opts['kan_daimin'])
        if self.can_ron() and isinstance(move, (Pon, Chi, KanDaimin)):
            return False
        opts = self.get_call_options()
        if isinstance(move, Pon):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['pon'])
        if isinstance(move, Chi):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['chi'])
        if isinstance(move, KanDaimin):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['kan_daimin'])
        return False

    def legal_moves(self) -> List[Union[Action, Reaction]]:
        moves: List[Union[Action, Reaction]] = []
        riichi_locked = self.riichi_declared.get(self.player_id, False)
        if self.state is Reaction and self.last_discarded_tile is not None and self.last_discard_player is not None and self.last_discard_player != self.player_id:
            if self.can_ron():
                return [PassCall(), Ron()]
            if riichi_locked:
                # After Riichi, only Pass or Ron as reactions
                return [PassCall()]
            opts = self.get_call_options()
            any_call = False
            for ts in opts['pon']:
                moves.append(Pon(ts)); any_call = True
            for ts in opts['chi']:
                moves.append(Chi(ts)); any_call = True
            for ts in opts['kan_daimin']:
                moves.append(KanDaimin(ts)); any_call = True
            if any_call:
                moves.insert(0, PassCall())
            return moves

        if self.state is Action:
            if self.can_tsumo():
                moves.append(Tsumo())
            # Riichi declaration
            if not self.riichi_declared.get(self.player_id, False) and not self.called_sets.get(self.player_id, []):
                # Propose riichi options per discardable tile that keeps tenpai
                seen: set = set()
                for t in self.player_hand:
                    key = (t.suit.value, int(t.tile_type.value))
                    if key in seen:
                        continue
                    seen.add(key)
                    r = Riichi(t)
                    if self.is_legal(r):
                        moves.append(r)
            # Kakan opportunities
            for t in self.player_hand:
                if self.is_legal(KanKakan(t)):
                    moves.append(KanKakan(t))
            # Ankan opportunities (do not list duplicates)
            seen: set = set()
            for t in self.player_hand:
                key = (t.suit, int(t.tile_type.value))
                if key in seen:
                    continue
                seen.add(key)
                if self.is_legal(KanAnkan(t)):
                    moves.append(KanAnkan(t))
            # Discards
            if riichi_locked:
                if self.newly_drawn_tile is not None:
                    moves.append(Discard(self.newly_drawn_tile))
            else:
                for t in self.player_hand:
                    moves.append(Discard(t))
        return moves


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, game_state: GamePerspective) -> Action:
        moves = game_state.legal_moves()
        # Auto-win
        for m in moves:
            if isinstance(m, Tsumo):
                return m
        # Riichi if possible: choose one of the parameterized Riichi moves
        for m in moves:
            if isinstance(m, Riichi):
                return m
        # Discard heuristic: first discard available
        for m in moves:
            if isinstance(m, Discard):
                return m
        # Fallback
        return Discard(game_state.player_hand[0])

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
        if game_state.can_ron():
            return Ron()
        if options.get('kan_daimin'):
            return KanDaimin(options['kan_daimin'][0])
        if options.get('pon'):
            return Pon(options['pon'][0])
        if options.get('chi'):
            return Chi(options['chi'][0])
        return PassCall()


class MediumJong:
    NUM_PLAYERS = MC_NUM_PLAYERS

    def __init__(self, players: List[Player], tile_copies: int = TILE_COPIES_DEFAULT):
        if len(players) != MediumJong.NUM_PLAYERS:
            raise ValueError("MediumJong requires exactly 4 players")
        self.players = players
        self._player_hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
        self._player_called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}
        self.player_discards: Dict[int, List[Tile]] = {i: [] for i in range(4)}
        # Track which discard indices were called by other players for each player
        self.called_discards: Dict[int, List[int]] = {i: [] for i in range(4)}
        self.current_player_idx: int = 0
        self.game_over: bool = False
        self.winners: List[int] = []
        self.loser: Optional[int] = None
        # whether the next step is an action or a reaction
        self._next_move_is_action = True
        self.last_discarded_tile: Optional[Tile] = None
        self.last_discard_player: Optional[int] = None
        self.last_drawn_tile: Optional[Tile] = None
        # number of copies per tile
        self.tile_copies = tile_copies
        
        # Winds
        self.round_wind: Honor = Honor.EAST
        self.seat_winds: Dict[int, Honor] = {
            0: Honor.EAST, 1: Honor.SOUTH, 2: Honor.WEST, 3: Honor.NORTH
        }
        # Riichi flags
        self.riichi_declared: Dict[int, bool] = {i: False for i in range(4)}
        # Dora/Uradora indicators (start with 1 each hidden)
        self.dora_indicators: List[Tile] = []
        self.ura_dora_indicators: List[Tile] = []
        # Riichi Ippatsu eligibility per player (true until canceled or consumed)
        self.riichi_ippatsu_active: Dict[int, bool] = {i: False for i in range(4)}
        # Riichi sticks pot (1k per riichi declared)
        self.riichi_sticks_pot: int = 0
        # Whether the most recent discard was a Riichi declaration discard
        self.last_discard_was_riichi: bool = False
        # Keiten payments on exhaustive draw (None unless draw occurs)
        self.keiten_payments: Optional[Dict[int, int]] = None

        # Build wall and dead wall
        self.tiles: List[Tile] = []
        # Add tiles; include exactly one aka 5 per suit by replacing one copy of 5
        for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
            for v in range(1, 10):
                # Number of copies to add; if v==5, we add (tile_copies - 1) normal + 1 aka
                copies = self.tile_copies
                if v == 5 and copies > 0:
                    # Add normal fives (copies - 1)
                    for _ in range(copies - 1):
                        self.tiles.append(Tile(suit, TileType(v)))
                    # Add aka five
                    self.tiles.append(Tile(suit, TileType(v), aka=True))
                else:
                    for _ in range(copies):
                        self.tiles.append(Tile(suit, TileType(v)))
        for h in Honor:
            for _ in range(self.tile_copies):
                self.tiles.append(Tile(Suit.HONORS, h))
        random.shuffle(self.tiles)

        # Split off dead wall (drawn from the back). Dora/ura indicators taken from dead wall.
        self.dead_wall: List[Tile] = []
        for _ in range(min(DEAD_WALL_TILES, len(self.tiles))):
            self.dead_wall.append(self.tiles.pop())
        # Place initial dora/ura indicators from dead wall top
        for _ in range(INITIAL_DORA_INDICATORS):
            if self.dead_wall:
                self.dora_indicators.append(self.dead_wall[-1])
        for _ in range(INITIAL_URADORA_INDICATORS):
            if len(self.dead_wall) >= 2:
                self.ura_dora_indicators.append(self.dead_wall[-2])

        # Deal 13 tiles to each player (dealer draws first turn tile later)
        for pid in range(4):
            for _ in range(STANDARD_HAND_TILE_COUNT):
                self._player_hands[pid].append(self.tiles.pop())

    class IllegalMoveException(Exception):
        pass

    class IllegalGamePerspective(Exception):
        pass

    def hand(self, player_id: int) -> List[Tile]:
        return list(self._player_hands[player_id])

    def called_sets(self, player_id: int) -> List[CalledSet]:
        return list(self._player_called_sets[player_id])

    def get_game_perspective(self, player_id: int) -> GamePerspective:
        if self._next_move_is_action:
            if player_id != self.current_player_idx:
                raise MediumJong.IllegalGamePerspective("Instantiating Game Perspective for a player who has no legal action now")
        else:
            if player_id == self.current_player_idx:
                raise MediumJong.IllegalGamePerspective(
                    "Instantiating Game Perspective for a player who has no legal action now")
        return GamePerspective(
            player_hand=self._player_hands[player_id],
            player_id=player_id,
            remaining_tiles=len(self.tiles),
            last_discarded_tile=self.last_discarded_tile,
            last_discard_player=self.last_discard_player,
            called_sets=self._player_called_sets,
            player_discards=self.player_discards,
            called_discards=self.called_discards,
            state=Action if self._next_move_is_action else Reaction,
            newly_drawn_tile=self.last_drawn_tile,
            can_call=self.last_discarded_tile is not None and self.last_discard_player != player_id,
            seat_winds=self.seat_winds,
            round_wind=self.round_wind,
            riichi_declared=self.riichi_declared,
        )

    def is_legal(self, actor_id: int, move: Union[Action, Reaction]) -> bool:
        if self.game_over:
            return False
        return self.get_game_perspective(actor_id).is_legal(move)

    def legal_moves(self, actor_id: int) -> List[Union[Action, Reaction]]:
        if self.game_over:
            return []
        return self.get_game_perspective(actor_id).legal_moves()

    def _draw_tile(self) -> None:
        assert self.tiles
        t = self.tiles.pop()
        self._player_hands[self.current_player_idx].append(t)
        self.last_drawn_tile = t

    def _rinshan_draw(self) -> None:
        # Draw from dead wall (rinshan) after a Kan
        assert self.dead_wall
        t = self.dead_wall.pop()
        self._player_hands[self.current_player_idx].append(t)
        self.last_drawn_tile = t

    def _add_kan_dora(self) -> None:
        # Flip an additional dora indicator from the dead wall (kandora); uradora mirrors count
        if self.dead_wall:
            self.dora_indicators.append(self.dead_wall[-1])
        if len(self.dead_wall) >= 2:
            self.ura_dora_indicators.append(self.dead_wall[-2])

    def _step_action(self, actor_id, move: Action):
        if isinstance(move, Tsumo):
            self._on_win(actor_id, win_by_tsumo=True)
        if isinstance(move, Riichi):
            # Declare riichi by discarding specified tile
            self.riichi_declared[actor_id] = True
            self.riichi_ippatsu_active[actor_id] = True
            self.riichi_sticks_pot += 1000
            self._player_hands[actor_id].remove(move.tile)
            self.player_discards[actor_id].append(move.tile)
            self.last_discarded_tile = move.tile
            self.last_discard_player = actor_id
            self.last_discard_was_riichi = True
            self._next_move_is_action = False
        if isinstance(move, Discard):
            self._player_hands[actor_id].remove(move.tile)
            self.player_discards[actor_id].append(move.tile)
            self.last_discarded_tile = move.tile
            self.last_discard_player = actor_id
            # Discard after riichi cancels ippatsu for this player
            if self.riichi_declared.get(actor_id, False):
                self.riichi_ippatsu_active[actor_id] = False
            self.last_discard_was_riichi = False
            self._next_move_is_action = False
        if isinstance(move, KanKakan):
            # Upgrade an existing pon to kan
            # Remove the drawn tile from hand
            self._player_hands[actor_id].remove(move.tile)
            # Update called set to 4 tiles
            for cs in self._player_called_sets[actor_id]:
                if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].suit == move.tile.suit and cs.tiles[
                    0].tile_type == move.tile.tile_type:
                    cs.call_type = 'kan_kakan'
                    cs.tiles.append(move.tile)
                    cs.called_tile = None
                    cs.source_position = None
                    break
            # Any kan cancels ippatsu for all players
            for pid in range(4):
                self.riichi_ippatsu_active[pid] = False
            self._add_kan_dora()
            self._rinshan_draw()
            # Continue action within the same overall turn after Kan
            self._action()
        if isinstance(move, KanAnkan):
            # Remove four from hand
            rm = 0
            new_hand: List[Tile] = []
            for t in self._player_hands[actor_id]:
                if rm < 4 and t.suit == move.tile.suit and t.tile_type == move.tile.tile_type:
                    rm += 1
                else:
                    new_hand.append(t)
            self._player_hands[actor_id] = new_hand
            self._player_called_sets[actor_id].append(
                CalledSet(tiles=[Tile(move.tile.suit, move.tile.tile_type) for _ in range(4)], call_type='kan_ankan',
                          called_tile=None, caller_position=actor_id, source_position=None))
            # Any kan cancels ippatsu for all players
            for pid in range(4):
                self.riichi_ippatsu_active[pid] = False
            self._add_kan_dora()
            self._rinshan_draw()
            # Continue action within the same overall turn after Kan
            self._action()

    def _step_reactions(self, actor_id, move: Reaction):
        # Reactions to discard
        if isinstance(move, Ron):
            if actor_id not in self.winners:
                self.winners.append(actor_id)
            self.loser = self.last_discard_player
        if isinstance(move, Pon):
            # Any call cancels ippatsu
            for pid in range(4):
                self.riichi_ippatsu_active[pid] = False
            self.last_discard_was_riichi = False
            last = self.last_discarded_tile
            # Mark the discarder index as called
            discarder = self.last_discard_player
            if discarder is not None:
                called_idx = len(self.player_discards[discarder]) - 1
                if called_idx >= 0:
                    self.called_discards[discarder].append(called_idx)
            # Consume two tiles
            consumed = 0
            new_hand: List[Tile] = []
            for t in self._player_hands[actor_id]:
                if consumed < 2 and t.suit == last.suit and t.tile_type == last.tile_type:
                    consumed += 1
                else:
                    new_hand.append(t)
            self._player_hands[actor_id] = new_hand
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(3)], call_type='pon', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.current_player_idx = actor_id
            self._next_move_is_action= True
            self._action()
        if isinstance(move, Chi):
            # Any call cancels ippatsu
            for pid in range(4):
                self.riichi_ippatsu_active[pid] = False
            self.last_discard_was_riichi = False
            last = self.last_discarded_tile
            # Mark the discarder index as called
            discarder = self.last_discard_player
            if discarder is not None:
                called_idx = len(self.player_discards[discarder]) - 1
                if called_idx >= 0:
                    self.called_discards[discarder].append(called_idx)
            # Remove provided two tiles
            for t in move.tiles:
                removed = False
                new_hand: List[Tile] = []
                for h in self._player_hands[actor_id]:
                    if not removed and h.suit == t.suit and h.tile_type == t.tile_type:
                        removed = True
                        continue
                    new_hand.append(h)
                self._player_hands[actor_id] = new_hand
            seq = sorted([move.tiles[0], last, move.tiles[1]], key=lambda t: int(t.tile_type.value))
            self._player_called_sets[actor_id].append(CalledSet(tiles=seq, call_type='chi', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._next_move_is_action = True
            self._action()
        if isinstance(move, KanDaimin):
            # Any call cancels ippatsu
            for pid in range(4):
                self.riichi_ippatsu_active[pid] = False
            self.last_discard_was_riichi = False
            last = self.last_discarded_tile
            # Mark the discarder index as called
            discarder = self.last_discard_player
            if discarder is not None:
                called_idx = len(self.player_discards[discarder]) - 1
                if called_idx >= 0:
                    self.called_discards[discarder].append(called_idx)
            # Remove three from hand
            consumed = 0
            new_hand: List[Tile] = []
            for t in self._player_hands[actor_id]:
                if consumed < 3 and t.suit == last.suit and t.tile_type == last.tile_type:
                    consumed += 1
                else:
                    new_hand.append(t)
            self._player_hands[actor_id] = new_hand
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(4)], call_type='kan_daimin', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            self._add_kan_dora()

    # performs exactly one action or reaction without triggering additional ones
    # useful if we want to say, discard and then return before the other 3 players have a chance to react
    def step(self, actor_id: int, move: Union[Action, Reaction]):
        if not self.is_legal(actor_id, move):
            raise MediumJong.IllegalMoveException("Illegal move")

        # Actions by current player
        if isinstance(move, (Tsumo, Discard, Riichi, KanKakan, KanAnkan)):
            self._step_action(actor_id, move)
        else:
            self._step_reactions(actor_id, move)

    # plays a full turn, an action + gives other players a chance to react
    def play_turn(self) -> Optional[int]:
        # If we're out of tiles, we handle exhaustive draw
        if not self.tiles and not self.game_over:
            self._on_exhaustive_draw()
            return

        self._draw_tile()
        self._action()
        self.current_player_idx = (self.current_player_idx + 1) % 4

    def _action(self) -> None:
        action = self.players[self.current_player_idx].play(self.get_game_perspective(self.current_player_idx))
        self.step(self.current_player_idx, action)
        # Resolve reactions if any
        if self.last_discarded_tile is not None:
            self._resolve_reactions()
            # we already handled reactions
            self.last_discarded_tile = None
            self._next_move_is_action = True

    def _resolve_reactions(self) -> None:
        discarder = self.last_discard_player
        # Gather options/choices
        choices: Dict[int, Reaction] = {}
        can_ron: Dict[int, bool] = {}
        for pid in range(4):
            if pid == discarder:
                continue
            gs = self.get_game_perspective(pid)
            opts = gs.get_call_options()
            if gs.can_ron():
                can_ron[pid] = True
                choices[pid] = self.players[pid].choose_reaction(gs, { })  # type: ignore[arg-type]
            elif opts['pon'] or opts['chi'] or opts['kan_daimin']:
                choices[pid] = self.players[pid].choose_reaction(gs, opts)
        # Ron first
        rons = [pid for pid, ch in choices.items() if isinstance(ch, Ron) and can_ron.get(pid, False)]
        if rons:
            self.winners = rons
            self.loser = discarder
            self.game_over = True
            return
        # Priority: Pon/ Kan over Chi by seat order from left
        order = [(discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4]
        for pid in order:
            ch = choices.get(pid)
            if isinstance(ch, (Pon, KanDaimin)) and self.is_legal(pid, ch):
                self.step(pid, ch)
                return
        # Chi for immediate left
        left = (discarder + 1) % 4
        ch = choices.get(left)
        if isinstance(ch, Chi) and self.is_legal(left, ch):
            self.step(left, ch)
            return

    def _on_win(self, winner_id: int, win_by_tsumo: bool) -> None:
        self.winners = [winner_id]
        self.loser = None if win_by_tsumo else self.last_discard_player
        self.game_over = True

    def _on_exhaustive_draw(self) -> None:
        # Determine tenpai players using current hands without constructing perspectives
        tenpai_players: List[int] = []
        for pid in range(4):
            if hand_is_tenpai(self._player_hands[pid]):
                tenpai_players.append(pid)
        n_t = len(tenpai_players)
        payments: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        if n_t == 1:
            tp = tenpai_players[0]
            for pid in range(4):
                if pid != tp:
                    payments[pid] -= 1000
                    payments[tp] += 1000
        elif n_t == 2:
            tset = set(tenpai_players)
            for pid in range(4):
                if pid in tset:
                    payments[pid] += 1500
                else:
                    payments[pid] -= 1500
        elif n_t == 3:
            # One not in tenpai pays 3000, split 1000 each
            nt = next(pid for pid in range(4) if pid not in tenpai_players)
            payments[nt] -= 3000
            for pid in tenpai_players:
                payments[pid] += 1000
        else:
            # 0 or 4: no payments
            pass
        # Subtract riichi sticks from any players who declared riichi this hand (sticks go to next win, not at draw)
        for pid in range(4):
            if self.riichi_declared.get(pid, False):
                payments[pid] -= 1000
        self.keiten_payments = payments
        self.winners = []
        self.loser = None
        self.game_over = True

    def is_game_over(self) -> bool:
        return self.game_over

    def get_winners(self) -> List[int]:
        return list(self.winners)

    def get_loser(self) -> Optional[int]:
        return self.loser

    def get_keiten_payments(self) -> Optional[Dict[int, int]]:
        return None if self.keiten_payments is None else dict(self.keiten_payments)

    # Scoring API
    def score_hand(self, winner_id: int, win_by_tsumo: bool) -> Dict[str, Any]:
        concealed = list(self._player_hands[winner_id])
        cs = list(self._player_called_sets[winner_id])
        fu, han, dora_han = _score_fu_and_han(
            concealed_tiles=concealed,
            called_sets=cs,
            winner_id=winner_id,
            dealer_id=DEALER_ID_START,
            win_by_tsumo=win_by_tsumo,
            riichi_declared=self.riichi_declared[winner_id],
            seat_wind=self.seat_winds[winner_id],
            round_wind=self.round_wind,
            dora_indicators=self.dora_indicators,
            ura_indicators=self.ura_dora_indicators if self.riichi_declared[winner_id] else [],
        )
        # Ippatsu: +1 han if riichi declared, ippatsu active, and win occurs on next draw before any call or discard
        if self.riichi_declared[winner_id] and self.riichi_ippatsu_active.get(winner_id, False):
            han += 1
            # Consumed on win
            self.riichi_ippatsu_active[winner_id] = False

        # Base points
        base_points = fu * (2 ** (2 + han))
        # Apply simple mangan cap for limit hands (≥5 han)
        dealer = (winner_id == DEALER_ID_START)
        if han >= 5:
            if win_by_tsumo:
                if dealer:
                    # Dealer tsumo mangan: 2000 each from three players
                    total = 2000 * 3
                    payments = {'total_from_others': total}
                    return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': True, 'payments': payments}
                else:
                    # Non-dealer tsumo mangan: dealer 2000, others 1000 each
                    payments = {'from_dealer': 2000, 'from_others': 1000, 'total_from_all': 2000 + 2 * 1000}
                    return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': payments['total_from_all'], 'tsumo': True, 'payments': payments}
            else:
                # Ron mangan
                total = 12000 if dealer else 8000
                return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': False, 'from': self.loser}

        # Simplified rounding for non-limit hands
        def round_up_100(x: int) -> int:
            return int(math.ceil(x / float(POINTS_ROUNDING)) * POINTS_ROUNDING)

        if win_by_tsumo:
            if dealer:
                total = round_up_100(base_points * 6)
                payments = {'total_from_others': total}
                # Winner collects riichi sticks pot on win
                if self.riichi_sticks_pot > 0:
                    payments['riichi_sticks'] = self.riichi_sticks_pot
                    self.riichi_sticks_pot = 0
            else:
                # Non-dealer split: dealer pays 2x, others 1x
                dealer_pay = round_up_100(base_points * 2)
                non_dealer_pay = round_up_100(base_points)
                total = dealer_pay + 2 * non_dealer_pay
                payments = {'from_dealer': dealer_pay, 'from_others': non_dealer_pay, 'total_from_all': total}
                if self.riichi_sticks_pot > 0:
                    payments['riichi_sticks'] = self.riichi_sticks_pot
                    self.riichi_sticks_pot = 0
            return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': True, 'payments': payments}
        else:
            # Ron
            if dealer:
                total = round_up_100(base_points * 6)
            else:
                total = round_up_100(base_points * 4)
            payments: Dict[str, Any] = {'from': self.loser}
            # Riichi sticks are awarded on ron only if the winning tile is not the riichi declaration discard
            if self.riichi_sticks_pot > 0 and not self.last_discard_was_riichi:
                payments['riichi_sticks'] = self.riichi_sticks_pot
                self.riichi_sticks_pot = 0
            return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': False, **payments}


    def play_round(self, max_steps: int = 10000) -> None:
        steps = 0
        while not self.is_game_over() and steps < max_steps:
            self.play_turn()
            steps += 1
        return None


