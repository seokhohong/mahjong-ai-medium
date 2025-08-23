import random
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np

from src.core.learn.ac_constants import chi_variant_index, ACTION_HEAD_INDEX, ACTION_HEAD_ORDER, TILE_HEAD_NOOP
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
    ITTSU_CLOSED_HAN,
    BASE_POINTS_EXPONENT_OFFSET,
    RIICHI_HAN,
    MENZEN_TSUMO_HAN,
    IPPATSU_HAN,
    MANGAN_HAN_THRESHOLD,
    MANGAN_DEALER_TSUMO_PAYMENT_EACH,
    MANGAN_NON_DEALER_TSUMO_DEALER_PAYMENT,
    MANGAN_NON_DEALER_TSUMO_OTHERS_PAYMENT,
    MANGAN_DEALER_RON_POINTS,
    MANGAN_NON_DEALER_RON_POINTS,
    DEALER_TSUMO_TOTAL_MULTIPLIER,
    NON_DEALER_TSUMO_DEALER_MULTIPLIER,
    NON_DEALER_TSUMO_OTHERS_MULTIPLIER,
    DEALER_RON_MULTIPLIER,
    NON_DEALER_RON_MULTIPLIER,
)
from .learn.policy_utils import _flat_tile_index

# Tile enums, dataclass, and helpers moved to dedicated module
from .tile import (
    Suit,
    TileType,
    Honor,
    Tile,
    tile_flat_index,
    _dora_next,
    _tile_sort_key,
    make_tile,
)


# Action and reaction dataclasses moved to dedicated module
from .action import (
    Action,
    Reaction,
    Discard,
    Riichi,
    Tsumo,
    Ron,
    PassCall,
    Pon,
    Chi,
    KanDaimin,
    KanKakan,
    KanAnkan,
)


# MediumJong: Expanded Riichi-like implementation
# - Suits: Manzu (m), Pinzu (p), Souzu (s) and Honors (winds/dragons)
# - Calls: Chi, Pon, Kan (daiminkan, kakan, ankan)
# - Yaku requirement to win; rudimentary scoring (fu/han) with dora/uradora
# - Round/seat winds; player 0 is dealer (East) and East round
# - Riichi declaration; after riichi, only Win/Kan allowed; uradora on riichi win




# Structured outcome types for a completed hand
class OutcomeType(Enum):
    RON = 'Ron'
    TSUMO = 'Tsumo'
    TENPAI = 'Tenpai'
    NOTEN = 'Noten'
    DEAL_IN = 'DealIn'


@dataclass
class PlayerOutcome:
    player_id: int
    outcome_type: Optional[OutcomeType]
    won: bool
    lost: bool
    tenpai: bool
    noten: bool
    points_delta: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_id': int(self.player_id),
            'outcome_type': None if self.outcome_type is None else str(self.outcome_type.value),
            'won': bool(self.won),
            'lost': bool(self.lost),
            'tenpai': bool(self.tenpai),
            'noten': bool(self.noten),
            'points_delta': int(self.points_delta),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PlayerOutcome':
        ot_raw = data.get('outcome_type')
        ot: Optional[OutcomeType]
        if ot_raw is None:
            ot = None
        else:
            # Map string value back to enum
            mapping = {
                'Ron': OutcomeType.RON,
                'Tsumo': OutcomeType.TSUMO,
                'Tenpai': OutcomeType.TENPAI,
                'Noten': OutcomeType.NOTEN,
                'DealIn': OutcomeType.DEAL_IN,
            }
            ot = mapping.get(str(ot_raw))
        return PlayerOutcome(
            player_id=int(data['player_id']),
            outcome_type=ot,
            won=bool(data['won']),
            lost=bool(data['lost']),
            tenpai=bool(data['tenpai']),
            noten=bool(data['noten']),
            points_delta=int(data['points_delta']),
        )


@dataclass
class GameOutcome:
    # Per-player results keyed by absolute player id 0..3
    players: Dict[int, PlayerOutcome]
    winners: List[int]
    loser: Optional[int]
    is_draw: bool

    def serialize(self) -> Dict[str, Any]:
        # Serialize to a JSON/npz-friendly dict
        return {
            'is_draw': bool(self.is_draw),
            'winners': [int(w) for w in self.winners],
            'loser': None if self.loser is None else int(self.loser),
            'players': [self.players[i].to_dict() for i in range(4)],
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'GameOutcome':
        players_list = data.get('players', [])
        players: Dict[int, PlayerOutcome] = {}
        for p_dict in players_list:
            po = PlayerOutcome.from_dict(p_dict)
            players[po.player_id] = po
        # Ensure all four players exist if possible by filling defaults
        for pid in range(4):
            if pid not in players:
                players[pid] = PlayerOutcome(
                    player_id=pid,
                    outcome_type=None,
                    won=False,
                    lost=False,
                    tenpai=False,
                    noten=False,
                    points_delta=0,
                )
        return GameOutcome(
            players=players,
            winners=[int(w) for w in data.get('winners', [])],
            loser=(None if data.get('loser', None) is None else int(data['loser'])),
            is_draw=bool(data.get('is_draw', False)),
        )

    def outcome_type(self, player_id: int) -> Optional[OutcomeType]:
        po = self.players.get(player_id)
        if po is None:
            return None
        return po.outcome_type

    def __repr__(self) -> str:
        header = f"GameOutcome(draw={self.is_draw}, winners={self.winners}, loser={self.loser})"
        lines: List[str] = [header]
        for pid in range(4):
            po = self.players.get(pid)
            if po is None:
                lines.append(f"  P{pid}: <missing>")
                continue
            otype = '-' if po.outcome_type is None else po.outcome_type.value
            lines.append(
                f"  P{pid}: type={otype}, won={po.won}, lost={po.lost}, tenpai={po.tenpai}, noten={po.noten}, points={po.points_delta}"
            )
        return "\n".join(lines)

# Actions and reactions are imported from src.core.action


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

    # use imported make_tile

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

def _possible_chis(hand: List[Tile], target: Tile) -> List[Chi]:
    options: List[Chi] = []
    if target.suit == Suit.HONORS:
        return []
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
            options.append(Chi([a, b], 0)) # variant 0
    # (v-1, v+1)
    if v - 1 >= 1 and v + 1 <= 9:
        a = has(v-1); b = has(v+1)
        if a and b:
            options.append(Chi([a, b], 1))
    # (v+1, v+2)
    if v + 1 <= 9 and v + 2 <= 9:
        a = has(v+1); b = has(v+2)
        if a and b:
            options.append(Chi([a, b], 2))
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
        # Standard hand fu: start from 20 base and add components
        open_hand = _is_open_hand(called_sets)

        # Helper: classify tile as terminal/honor vs simple
        def _is_terminal_or_honor(t: Tile) -> bool:
            if t.suit == Suit.HONORS:
                return True
            v = int(t.tile_type.value)
            return v == 1 or v == 9

        # Base fu (20 for standard hands)
        fu = 20

        # 1) Meld fu from called sets
        for cs in called_sets:
            if not cs.tiles:
                continue
            t0 = cs.tiles[0]
            th = _is_terminal_or_honor(t0)
            if cs.call_type == 'chi':
                # sequences give no fu
                pass
            elif cs.call_type == 'pon':
                # Open triplet
                fu += 4 if th else 2
            elif cs.call_type in ('kan_daimin', 'kan_kakan'):
                # Open kan
                fu += 16 if th else 8
            elif cs.call_type == 'kan_ankan':
                # Closed kan
                fu += 32 if th else 16

        # 2) Meld fu from concealed triplets present in concealed tiles
        cnt = _count_tiles(concealed_tiles)
        for (suit, val), c in cnt.items():
            if c >= 3:
                # Concealed triplet (ankan counted above via called_sets)
                if suit == Suit.HONORS or val in (1, 9):
                    fu += 8
                else:
                    fu += 4

        # 3) Pair fu (yakuhai pair)
        # Detect any pair in concealed tiles that is dragon or seat/round wind
        pair_fu_added = False
        for (suit, val), c in cnt.items():
            if c >= 2 and suit == Suit.HONORS:
                try:
                    h = Honor(val)  # type: ignore[arg-type]
                except Exception:
                    continue
                if h in (Honor.WHITE, Honor.GREEN, Honor.RED) or h == seat_wind or h == round_wind:
                    fu += 2
                    pair_fu_added = True
                    break

        # 4) Wait fu (+2 for kanchan, penchan, tanki)
        # Infer winning tile by removing one tile that was likely just added and checking waits
        def _infer_win_tile_and_wait_fu() -> int:
            from .tenpai import waits_for_tiles as _waits_13
            # Build unique tile kinds from concealed hand
            kinds: List[Tile] = []
            seen = set()
            for t in concealed_tiles:
                key = (t.suit, int(t.tile_type.value))
                if key not in seen:
                    seen.add(key)
                    kinds.append(t)
            # Try each as the winning tile
            for cand in kinds:
                # Remove one cand -> 13 tiles
                removed = False
                hand13: List[Tile] = []
                for t in concealed_tiles:
                    if (not removed) and t.suit == cand.suit and t.tile_type == cand.tile_type:
                        removed = True
                        continue
                    hand13.append(t)
                if len(hand13) != len(concealed_tiles) - 1:
                    continue
                waits = _waits_13(hand13)
                # Check if cand is indeed a wait
                if any(w.suit == cand.suit and w.tile_type == cand.tile_type for w in waits):
                    # Classify wait type for suits only
                    if cand.suit == Suit.HONORS:
                        # Honors cannot form sequences; only tanki or shanpon. Tanki if hand13 had exactly one of cand.
                        if cnt.get((cand.suit, int(cand.tile_type.value)), 0) - 1 == 1:
                            return 2
                        return 0
                    v = int(cand.tile_type.value)
                    # Count of cand in 13 tiles
                    c13 = sum(1 for t in hand13 if t.suit == cand.suit and int(t.tile_type.value) == v)
                    if c13 == 1:
                        # Likely tanki (pair wait)
                        return 2
                    if c13 >= 2:
                        # Shanpon (two pairs waiting to become triplet) -> 0
                        return 0
                    # c13 == 0 -> sequence wait; distinguish kanchan/penchan/ryanmen
                    def has(val: int) -> bool:
                        return any(t.suit == cand.suit and int(t.tile_type.value) == val for t in hand13)
                    # Kanchan: v-1 and v+1 exist
                    if 2 <= v <= 8 and has(v-1) and has(v+1):
                        return 2
                    # Penchan: 1-2-[3] or [7]-8-9
                    if (v == 3 and has(1) and has(2)) or (v == 7 and has(8) and has(9)):
                        return 2
                    # Otherwise ryanmen or other -> 0
                    return 0
            return 0

        fu += _infer_win_tile_and_wait_fu()

        # 5) Winning method fu
        if win_by_tsumo:
            # +2 for tsumo; waived for pinfu tsumo below
            fu += 2
        else:
            # Closed ron +10
            if not open_hand:
                fu += 10

        # Pinfu handling:
        # - Closed pinfu tsumo is fixed 20 (no +2 tsumo)
        # - Closed pinfu ron will already be 30 (20 base + 10 menzen ron)
        if _is_pinfu(all_tiles, called_sets, seat_wind, round_wind):
            if win_by_tsumo:
                fu = 20

        # Open hand with no fu sources should be set to 30 on ron (per tests)
        if open_hand and not win_by_tsumo and fu <= 20:
            fu = 30
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
        han += RIICHI_HAN
    # Menzen (menzen tsumo): closed hand tsumo
    if win_by_tsumo and not _is_open_hand(called_sets):
        han += MENZEN_TSUMO_HAN

    # Dora (including aka) contribute to total han returned here; also return dora_han separately
    dora_han = _calc_dora_han(concealed_tiles, called_sets, dora_indicators)
    if riichi_declared:
        dora_han += _calc_dora_han(concealed_tiles, called_sets, ura_indicators)
    # Count red-5 (aka) as dora as well
    aka_han = _count_aka_han(all_tiles)
    han += dora_han + aka_han
    
    # Fu rounding (except chiitoi 25 fu already handled)
    if not chiitoi:
        # Round up to nearest 10
        if fu % 10 != 0:
            fu = fu + (10 - fu % 10)

    return fu, han, dora_han


def tile_flat_index(tile: Tile) -> int:
    """Map a Tile to a compact 0..36 index for action masks (37 slots).
    Per-suit blocks:
    - Manzu: 0..9  (0m for aka five, 1m..9m -> 1..9)
    - Pinzu: 10..19 (0p -> 10, 1p..9p -> 11..19)
    - Souzu: 20..29 (0s -> 20, 1s..9s -> 21..29)
    - Honors: 30..36 (E,S,W,N,Wh,G,R)
    """
    if tile.suit == Suit.MANZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 0
        return int(tile.tile_type.value)
    if tile.suit == Suit.PINZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 10
        return 10 + int(tile.tile_type.value)
    if tile.suit == Suit.SOUZU:
        if tile.tile_type == TileType.FIVE and tile.aka:
            return 20
        return 20 + int(tile.tile_type.value)
    # Honors 30..36
    return 29 + int(tile.tile_type.value)


def _chi_variant_index(last_discarded_tile: Optional[Tile], tiles: List[Tile]) -> int:
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


class GamePerspective:
    def __init__(self,
                 player_hand: List[Tile],
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
        self.remaining_tiles = remaining_tiles
        self._reactable_tile = last_discarded_tile
        self._owner_of_reactable_tile = last_discard_player
        self.called_sets = {pid: list(sets) for pid, sets in called_sets.items()}
        self.player_discards = {pid: list(ts) for pid, ts in player_discards.items()}
        self.called_discards = {pid: list(idxs) for pid, idxs in called_discards.items()}
        self.state = state
        self.newly_drawn_tile = newly_drawn_tile
        self.can_call = can_call
        self.seat_winds = dict(seat_winds)
        self.round_wind = round_wind
        self.riichi_declared = dict(riichi_declared)

    def __repr__(self) -> str:
        def _fmt_hand(tiles: List[Tile]) -> str:
            sorted_tiles = sorted(tiles, key=lambda t: (t.suit.value, int(t.tile_type.value) if t.suit != Suit.HONORS else int(t.tile_type.value)))
            return '[' + ', '.join(str(t) for t in sorted_tiles) + ']'

        def _fmt_called_sets(csets: List[CalledSet]) -> str:
            if not csets:
                return '[]'
            parts: List[str] = []
            for cs in csets:
                tiles = getattr(cs, 'tiles', [])
                call_type = getattr(cs, 'call_type', '?')
                sorted_tiles = sorted(tiles, key=lambda t: (t.suit.value, int(t.tile_type.value) if t.suit != Suit.HONORS else int(t.tile_type.value)))
                parts.append(f"{call_type}:[" + ', '.join(str(t) for t in sorted_tiles) + "]")
            return '[' + '; '.join(parts) + ']'

        newly = self.newly_drawn_tile
        newly_s = str(newly) if newly is not None else 'None'
        last = self._reactable_tile
        last_s = str(last) if last is not None else 'None'
        last_from = 'None' if self._owner_of_reactable_tile is None else f"P{self._owner_of_reactable_tile}"
        my_called = _fmt_called_sets(self.called_sets.get(0, []))
        state_s = 'Action' if self.state is Action else 'Reaction'
        can_call_s = '1' if self.can_call else '0'
        return (
            f"GP[{state_s}] Hand {_fmt_hand(self.player_hand)} | Called {my_called} | "
            f"Draw {newly_s} | Last {last_s} from {last_from} | CanCall {can_call_s} | Round {self.round_wind.name}"
        )

    def _concealed_tiles(self) -> List[Tile]:
        return list(self.player_hand)

    def legal_action_mask(self) -> np.ndarray:
        """Return ACTION_HEAD_SIZE-length 0/1 mask over action head.

        For tile-parameterized actions, mark 1 if there exists at least one legal instantiation.
        """
        import numpy as _np
        m = _np.zeros((len(ACTION_HEAD_ORDER),), dtype=_np.float64)
        moves = self.legal_moves()
        # Presence flags
        has = {name: False for name in ACTION_HEAD_ORDER}
        last = self._reactable_tile
        # Pre-compute reaction options (flat list)
        reaction_opts: List[Reaction] = self.get_call_options() if self.state is Reaction else []
        # Chi variants
        for r in reaction_opts:
            if isinstance(r, Chi):
                has_aka = False
                if last is not None:
                    has_aka = any(getattr(t, 'aka', False) for t in (r.tiles + [last]))
                variant = getattr(r, 'chi_variant_index', None)
                if variant == 0:
                    has['chi_low_aka' if has_aka else 'chi_low_noaka'] = True
                elif variant == 1:
                    has['chi_mid_aka' if has_aka else 'chi_mid_noaka'] = True
                elif variant == 2:
                    has['chi_high_aka' if has_aka else 'chi_high_noaka'] = True
        # Pon variants
        for r in reaction_opts:
            if isinstance(r, Pon):
                has_aka = False
                if last is not None:
                    has_aka = any(getattr(t, 'aka', False) for t in (r.tiles + [last]))
                has['pon_aka' if has_aka else 'pon_noaka'] = True
        # Daiminkan present
        if any(isinstance(r, KanDaimin) for r in reaction_opts):
            has['kan_daimin'] = True
        # Iterate legal moves for other actions
        for mv in moves:
            if isinstance(mv, Discard):
                has['discard'] = True
            elif isinstance(mv, Riichi):
                has['riichi'] = True
            elif isinstance(mv, Tsumo):
                has['tsumo'] = True
            elif isinstance(mv, Ron):
                has['ron'] = True
            elif isinstance(mv, KanKakan):
                has['kan_kakan'] = True
            elif isinstance(mv, KanAnkan):
                has['kan_ankan'] = True
            elif isinstance(mv, PassCall):
                has['pass'] = True
        for name, ok in has.items():
            if name in ACTION_HEAD_INDEX and ok:
                m[ACTION_HEAD_INDEX[name]] = 1.0
        return m

    def legal_tile_mask(self, action_idx: int) -> np.ndarray:
        """Return TILE_HEAD_SIZE-length 0/1 mask for the tile head given chosen action.

        If the chosen action is not tile-parameterized, returns mask with only no-op enabled.
        For tile-parameterized actions, enables tile slots corresponding to legal instantiations.
        """
        import numpy as _np
        name = ACTION_HEAD_ORDER[action_idx] if 0 <= action_idx < len(ACTION_HEAD_ORDER) else None
        m = _np.zeros((38,), dtype=_np.float64)
        if name is None:
            return m
        # Non-parameterized -> no-op only
        if name in ('tsumo', 'ron', 'pass', 'kan_daimin') or name.startswith('chi_') or name.startswith('pon_'):
            m[TILE_HEAD_NOOP] = 1.0
            return m
        # Collect legal tiles per action
        cand: list[int] = []
        if name == 'discard' or name == 'riichi':
            for mv in self.legal_moves():
                if (name == 'discard' and isinstance(mv, Discard)) or (name == 'riichi' and isinstance(mv, Riichi)):
                    cand.append(tile_flat_index(mv.tile))
        elif name == 'kan_kakan':
            for mv in self.legal_moves():
                if isinstance(mv, KanKakan):
                    cand.append(tile_flat_index(mv.tile))
        elif name == 'kan_ankan':
            for mv in self.legal_moves():
                if isinstance(mv, KanAnkan):
                    cand.append(tile_flat_index(mv.tile))
        # Populate mask
        if cand:
            for idx in cand:
                if 0 <= idx <= 36:
                    m[idx] = 1.0
        else:
            # If no legal instantiation detected, still allow no-op to avoid invalid sampling
            m[TILE_HEAD_NOOP] = 1.0
        return m


    def _has_yaku_if_complete(self) -> bool:
        # Simple heuristic: evaluate yaku on this hand if it is standard or chiitoi
        ct = self._concealed_tiles()
        cs = self.called_sets.get(0, [])
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
            if _yakuhai_han(ct, cs, self.seat_winds[0], self.round_wind) > 0:
                return True
        return False

    def can_tsumo(self) -> bool:
        if self.newly_drawn_tile is None:
            return False
        # Require yaku
        # Evaluate win using current hand (already includes the drawn tile)
        ct = list(self.player_hand)
        cs = self.called_sets.get(0, [])
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
        if _yakuhai_han(ct, cs, self.seat_winds[0], self.round_wind) > 0:
            return True
        # Menzen tsumo yaku
        if not _is_open_hand(cs):
            return True
        return False

    def can_ron(self) -> bool:
        if self._reactable_tile is None or self._owner_of_reactable_tile == 0:
            return False
        # Furiten blocks ron
        if self._is_furiten():
            return False
        return self._win_possible(require_yaku=True, include_last_discard=True)

    def _win_possible(self, require_yaku: bool, include_last_discard: bool = False) -> bool:
        ct = list(self.player_hand)
        if include_last_discard and self._reactable_tile is not None:
            ct = ct + [self._reactable_tile]
        cs = self.called_sets.get(0, [])
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
        if _yakuhai_han(ct, cs, self.seat_winds[0], self.round_wind) > 0:
            return True
        # Pinfu (closed only) counts as yaku for win condition
        if not _is_open_hand(cs) and _is_pinfu(all_tiles, cs, self.seat_winds[0], self.round_wind):
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
        own_discards = self.player_discards.get(0, [])
        for w in waits:
            if any(d.suit == w.suit and d.tile_type == w.tile_type for d in own_discards):
                return True
        return False

    def discard_is_called(self, player_id: int, discard_index: int) -> bool:
        return discard_index in self.called_discards.get(player_id, [])

    def get_call_options(self) -> List[Reaction]:
        """Return a flat list of legal reaction moves to the last discard.

        Includes `Chi`, `Pon`, and `KanDaimin` when available. Empty when no
        reactions are possible or reacting to own discard.
        """
        reactions: List[Reaction] = []
        last = self._reactable_tile
        lp = self._owner_of_reactable_tile
        if last is None or lp is None or lp == 0:
            return reactions
        hand = list(self.player_hand)
        # Pon
        same = [t for t in hand if t.suit == last.suit and t.tile_type == last.tile_type]
        if len(same) >= 2:
            reactions.append(Pon([same[0], same[1]]))
        # Chi (left player only)
        if 0 == (lp + 1) % 4 and last.suit != Suit.HONORS:
            reactions.extend(_possible_chis(hand, last))
        # Daiminkan: need three in hand
        if len(same) >= 3:
            reactions.append(KanDaimin([same[0], same[1], same[2]]))
        return reactions

    def is_legal(self, move: Union[Action, Reaction]) -> bool:
        # Riichi restriction: after declaring, only Tsumo or Kan actions are allowed on own turn
        riichi_locked = self.riichi_declared.get(0, False)
        if isinstance(move, (Tsumo, Discard, Riichi, KanKakan, KanAnkan)):
            if self.state is not Action:
                return False
            if isinstance(move, Tsumo):
                return self.can_tsumo()
            if isinstance(move, Riichi):
                # Closed hand, not already in riichi, and discarding specified tile keeps tenpai
                if self.riichi_declared.get(0, False):
                    return False
                if self.called_sets.get(0, []):
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
                for cs in self.called_sets.get(0, []):
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
        if isinstance(move, (Chi, Pon, KanDaimin, Ron, PassCall)):
            # Reactions
            if self.state is not Reaction:
                return False
            if isinstance(move, PassCall):
                return True
            # Riichi restriction on reactions: cannot Chi/Pon/KanDaimin after declaring Riichi
            if riichi_locked and isinstance(move, (Pon, Chi, KanDaimin)):
                return False
            if isinstance(move, Ron):
                return self.can_ron()
            if isinstance(move, PassCall):
                opts = self.get_call_options()
                return self.can_ron() or bool(opts)
            opts = self.get_call_options()
            if isinstance(move, (Pon, Chi, KanDaimin)):
                return any(move == cand for cand in opts)
            return False
        return False


    def legal_moves(self) -> List[Union[Action, Reaction]]:
        moves: List[Union[Action, Reaction]] = []
        riichi_locked = self.riichi_declared.get(0, False)
        if self.state is Reaction:
            moves = [PassCall()]
            if self.can_ron():
                return moves + [Ron()]
            if riichi_locked:
                # After Riichi, only Pass or Ron as reactions
                return moves
            moves.extend(self.get_call_options())
            return moves
        if self.state is Action:
            if self.can_tsumo():
                moves.append(Tsumo())
            # Riichi declaration
            if not self.riichi_declared.get(0, False) and not self.called_sets.get(0, []):
                # Optimized riichi candidates without per-tile is_legal calls
                from .tenpai import legal_riichi_moves
                moves.extend(legal_riichi_moves(self.riichi_declared, self.called_sets, self.player_hand))
            # Kakan opportunities
            for t in self.are_legal_kakans():
                moves.append(KanKakan(t))
            # Ankan opportunities (do not list duplicates)
            for t in self.are_legal_ankans():
                moves.append(KanAnkan(t))
            # Discards
            if riichi_locked:
                if self.newly_drawn_tile is not None:
                    moves.append(Discard(self.newly_drawn_tile))
            else:
                for t in self.player_hand:
                    moves.append(Discard(t))
        return moves

    def are_legal_kakans(self) -> List[Tile]:
        """Compute all kakan-capable tiles without calling is_legal repeatedly.

        A kakan is legal if:
        - We are in Action state (this method is only used there in legal_moves)
        - The player already has a pon of that tile in `called_sets[0]`
        - The player holds at least one more tile of the same suit/tile_type in hand
        Riichi state allows Kan actions, so no extra restriction here.
        """
        cs = self.called_sets.get(0, [])
        if not cs:
            return []
        # Collect pon tile keys the player has called
        pon_keys: set = set()
        for called in cs:
            if getattr(called, 'call_type', None) == 'pon' and getattr(called, 'tiles', None):
                t0 = called.tiles[0]
                pon_keys.add((t0.suit, t0.tile_type))
        if not pon_keys:
            return []
        # For each pon key, if hand contains at least one matching tile, it's a legal kakan
        hand = self.player_hand
        results: List[Tile] = []
        seen_key: set = set()
        for t in hand:
            key = (t.suit, t.tile_type)
            if key in pon_keys and key not in seen_key:
                results.append(t)  # representative tile for this kakan
                seen_key.add(key)
        return results

    def are_legal_ankans(self) -> List[Tile]:
        """Compute all ankan-capable tiles without calling is_legal.

        An ankan is legal if there are four tiles of the same suit/tile_type in hand.
        Return one representative tile per qualifying key.
        """
        counts: Dict[Tuple[Suit, Any], int] = {}
        rep: Dict[Tuple[Suit, Any], Tile] = {}
        for t in self.player_hand:
            key = (t.suit, t.tile_type)
            counts[key] = counts.get(key, 0) + 1
            # keep the first representative we see
            if key not in rep:
                rep[key] = t
        results: List[Tile] = []
        for key, cnt in counts.items():
            if cnt >= 4:
                results.append(rep[key])
        return results


class Player:

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
        return moves[0]

    def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]) -> Reaction:
        if game_state.can_ron():
            return Ron()
        for r in options:
            if isinstance(r, KanDaimin):
                return r
        for r in options:
            if isinstance(r, Pon):
                return r
        for r in options:
            if isinstance(r, Chi):
                return r
        return PassCall()


class MediumJong:
    NUM_PLAYERS = MC_NUM_PLAYERS

    def __init__(self, players: List[Player], tile_copies: int = TILE_COPIES_DEFAULT):
        if len(players) != MediumJong.NUM_PLAYERS:
            raise ValueError("MediumJong requires exactly 4 players")
        self.players = players
        # Internal mapping from Player instance to absolute player id (0..3)
        # This avoids relying on Player having a player_id attribute.
        self._player_to_id: Dict[Player, int] = {p: i for i, p in enumerate(players)}
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
        self._reactable_tile: Optional[Tile] = None
        self._owner_of_reactable_tile: Optional[int] = None
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
        # Per-player point deltas at end of game (win/tsumo/ron or exhaustive draw)
        self.points: Optional[List[int]] = None
        # Cumulative per-player point deltas from start of game, including riichi stick payments
        self.cumulative_points: List[int] = [0, 0, 0, 0]
        # Stored structured outcome after hand ends
        self.game_outcome: Optional[GameOutcome] = None

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

        # Helper methods for riichi/ippatsu state management

    def get_player_id(self, player: Player) -> int:
        """Return absolute player id for a Player instance (0..3).

        MediumJong manages this mapping internally and does not rely on
        Player.player_id existing.
        """
        return self._player_to_id[player]

    def get_player(self, player_id: int) -> Player:
        """Return Player instance for an absolute player id (0..3)."""
        return self.players[player_id]

    def _cancel_ippatsu_all(self) -> None:
        """Cancel ippatsu eligibility for all players."""
        for pid in range(4):
            self.riichi_ippatsu_active[pid] = False

    def _cancel_ippatsu_for(self, player_id: int) -> None:
        """Cancel ippatsu eligibility for a specific player."""
        self.riichi_ippatsu_active[player_id] = False

    def _on_any_call_side_effects(self) -> None:
        """Common side effects when any call (chi/pon/daiminkan) happens."""
        self._cancel_ippatsu_all()
        self.last_discard_was_riichi = False

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
        # Rotate all per-player structures so that the requesting player is index 0
        def rot_idx(idx: int) -> int:
            return (idx - player_id) % 4
        called_sets = {rot_idx(pid): list(sets) for pid, sets in self._player_called_sets.items()}
        player_discards = {rot_idx(pid): list(ts) for pid, ts in self.player_discards.items()}
        called_discards = {rot_idx(pid): list(idxs) for pid, idxs in self.called_discards.items()}
        seat_winds = {rot_idx(pid): wind for pid, wind in self.seat_winds.items()}
        riichi_declared = {rot_idx(pid): val for pid, val in self.riichi_declared.items()}
        last_discard_player_rel = None if self._owner_of_reactable_tile is None else rot_idx(self._owner_of_reactable_tile)
        return GamePerspective(
            player_hand=self._player_hands[player_id],
            remaining_tiles=len(self.tiles),
            last_discarded_tile=self._reactable_tile,
            last_discard_player=last_discard_player_rel,
            called_sets=called_sets,
            player_discards=player_discards,
            called_discards=called_discards,
            state=Action if self._next_move_is_action else Reaction,
            newly_drawn_tile=self.last_drawn_tile,
            can_call=self._reactable_tile is not None and self._owner_of_reactable_tile is not None and self._owner_of_reactable_tile != player_id,
            seat_winds=seat_winds,
            round_wind=self.round_wind,
            riichi_declared=riichi_declared,
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
            # Account riichi stick payment immediately in cumulative points
            self.cumulative_points[actor_id] -= 1000
            self._player_hands[actor_id].remove(move.tile)
            self.player_discards[actor_id].append(move.tile)
            self._reactable_tile = move.tile
            self._owner_of_reactable_tile = actor_id
            self.last_discard_was_riichi = True
            self._next_move_is_action = False
        if isinstance(move, Discard):
            self._player_hands[actor_id].remove(move.tile)
            self.player_discards[actor_id].append(move.tile)
            self._reactable_tile = move.tile
            self._owner_of_reactable_tile = actor_id
            # Discard after riichi cancels ippatsu for this player
            if self.riichi_declared.get(actor_id, False):
                self._cancel_ippatsu_for(actor_id)
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
            self._cancel_ippatsu_all()
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
            self._cancel_ippatsu_all()
            self._add_kan_dora()
            self._rinshan_draw()
            # Continue action within the same overall turn after Kan
            self._action()

    def _step_reactions(self, actor_id, move: Reaction):
        # Reactions to discard
        if isinstance(move, Ron):
            self._on_win(actor_id, win_by_tsumo=False)
        if isinstance(move, Pon):
            # Any call cancels ippatsu
            self._on_any_call_side_effects()
            last = self._reactable_tile
            # Mark the discarder index as called
            discarder = self._owner_of_reactable_tile
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
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(3)], call_type='pon', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self._owner_of_reactable_tile))
            self.current_player_idx = actor_id
            self._action()
        if isinstance(move, Chi):
            # Any call cancels ippatsu
            self._on_any_call_side_effects()
            last = self._reactable_tile
            # Mark the discarder index as called
            discarder = self._owner_of_reactable_tile
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
            self._player_called_sets[actor_id].append(CalledSet(tiles=seq, call_type='chi', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self._owner_of_reactable_tile))
            self._reactable_tile = None
            self._owner_of_reactable_tile = None
            self.current_player_idx = actor_id
            self._next_move_is_action = True
            self._action()
        if isinstance(move, KanDaimin):
            # Any call cancels ippatsu
            self._on_any_call_side_effects()
            last = self._reactable_tile
            # Mark the discarder index as called
            discarder = self._owner_of_reactable_tile
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
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(4)], call_type='kan_daimin', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self._owner_of_reactable_tile))
            self._reactable_tile = None
            self._owner_of_reactable_tile = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True

    # performs exactly one action or reaction without triggering additional ones
    # useful if we want to say, discard and then return before the other 3 players have a chance to react
    def step(self, actor_id: int, move: Union[Action, Reaction]):
        if not self.is_legal(actor_id, move):
            gp = self.get_game_perspective(actor_id)
            # Use centralized debug snapshot utility for illegal move exports
            try:
                from .learn.data_utils import DebugSnapshot
                path = DebugSnapshot.save_illegal_move(
                    action_index=None,
                    game_perspective=gp,
                    action_obj=move,
                    encoded_state=None,
                    value=None,
                    main_logp=None,
                    main_probs=None,
                    reason='illegal_move_in_game_step',
                    out_dir=None,
                )
                if path:
                    print(f"[MediumJong] Illegal move dump written to {path}")
                else:
                    print("[MediumJong] Failed to write illegal move dump")
            except Exception as e:
                print(f"[MediumJong] Failed to write illegal move dump: {e}")
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
        # we assume all reactions are resolved here (if they weren't, we recursed)
        self._resolve_reactions()
        self.current_player_idx = (self.current_player_idx + 1) % 4

    def is_reactable(self):
        return self._reactable_tile is not None

    def reactable_tile(self):
        return self._reactable_tile

    def owner_of_reactable_tile(self):
        return self._owner_of_reactable_tile

    def _resolve_reactions(self):
        # we already handled reactions
        self._reactable_tile = None
        self._next_move_is_action = True
        self._owner_of_reactable_tile = None

    def _action(self) -> None:
        # we call this here because _poll_reactions can recurse on action
        self._resolve_reactions()
        action = self.players[self.current_player_idx].play(self.get_game_perspective(self.current_player_idx))
        self.step(self.current_player_idx, action)
        # Resolve reactions if any
        if self.is_reactable():
            self._poll_reactions()

    def _poll_reactions(self) -> None:
        discarder = self._owner_of_reactable_tile
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
                choices[pid] = self.players[pid].choose_reaction(gs, [])
            elif opts:
                choices[pid] = self.players[pid].choose_reaction(gs, opts)
        # Ron first
        rons = [pid for pid, ch in choices.items() if isinstance(ch, Ron) and can_ron.get(pid, False)]
        if rons:
            # Multiple-ron support: resolve all rons in turn order from discarder+1
            order = [(discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4] if discarder is not None else [0,1,2]
            ordered_rons = [pid for pid in order if pid in rons]
            # Finalize multi-ron outcome
            self._on_multiple_ron(ordered_rons, discarder)
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
        self.loser = None if win_by_tsumo else self._owner_of_reactable_tile
        self.game_over = True
        # Populate per-player points for this outcome
        deltas = [0, 0, 0, 0]
        score = self._score_hand(winner_id, win_by_tsumo=win_by_tsumo)
        if score.get('tsumo', False):
            # Winner gains total points; others pay according to split
            if winner_id == DEALER_ID_START:
                # Dealer tsumo: split evenly among the three others
                total = int(score['points'])
                each = total // 3
                for pid in range(4):
                    if pid == winner_id:
                        continue
                    deltas[pid] -= each
                deltas[winner_id] += total
            else:
                payments = score.get('payments', {})
                from_dealer = int(payments.get('from_dealer', 0))
                from_others = int(payments.get('from_others', 0))
                for pid in range(4):
                    if pid == winner_id:
                        continue
                    if pid == DEALER_ID_START:
                        deltas[pid] -= from_dealer
                    else:
                        deltas[pid] -= from_others
                deltas[winner_id] += int(payments.get('total_from_all', from_dealer + 2 * from_others))
            # Add riichi sticks if present (pot already cleared in score_hand)
            rs = int(score.get('payments', {}).get('riichi_sticks', 0))
            if rs:
                deltas[winner_id] += rs
        else:
            # Ron: loser pays total to winner
            total = int(score['points'])
            loser = self.loser
            if loser is not None:
                deltas[loser] -= total
                deltas[winner_id] += total
            # Add riichi sticks if present (awarded to winner)
            rs = int(score.get('riichi_sticks', score.get('payments', {}).get('riichi_sticks', 0)))
            if rs:
                deltas[winner_id] += rs
        self.points = deltas
        # Accumulate outcome into cumulative points (riichi stick awards already included in deltas)
        for pid in range(4):
            self.cumulative_points[pid] += deltas[pid]
        # Build and store structured outcome
        self._store_game_outcome()

    def _on_multiple_ron(self, winner_ids: List[int], discarder: Optional[int]) -> None:
        # Finalize outcome for multiple simultaneous rons (double/triple ron)
        if discarder is None or not winner_ids:
            return
        self.winners = list(winner_ids)
        self.loser = discarder
        self.game_over = True
        deltas = [0, 0, 0, 0]
        # Ensure loser context for scoring
        self._owner_of_reactable_tile = discarder
        # Award riichi sticks only to the first winner per rule; order already set
        for idx, w in enumerate(winner_ids):
            score = self._score_hand(w, win_by_tsumo=False)
            total = int(score['points'])
            deltas[discarder] -= total
            deltas[w] += total
            # Only the first ron winner collects riichi sticks (if any)
            if idx == 0:
                rs = int(score.get('riichi_sticks', score.get('payments', {}).get('riichi_sticks', 0)))
                if rs:
                    deltas[w] += rs
            # After first scoring, riichi pot will already be cleared if applicable
        self.points = deltas
        for pid in range(4):
            self.cumulative_points[pid] += deltas[pid]
        self._store_game_outcome()

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
        # Riichi stick payments are accounted at declaration time in cumulative_points; do not subtract here
        self.keiten_payments = payments
        self.winners = []
        self.loser = None
        self.game_over = True
        # Persist points as per-player deltas
        self.points = [payments[i] for i in range(4)]
        for pid in range(4):
            self.cumulative_points[pid] += self.points[pid]
        self._store_game_outcome()

    def is_game_over(self) -> bool:
        return self.game_over

    def get_winners(self) -> List[int]:
        if not self.game_over:
            raise ValueError("Game is not over")
        if self.game_outcome is not None:
            return list(self.game_outcome.winners)
        return list(self.winners)

    def get_loser(self) -> Optional[int]:
        if not self.game_over:
            raise ValueError("Game is not over")
        if self.game_outcome is not None:
            return self.game_outcome.loser
        return self.loser

    def get_points(self) -> Optional[List[int]]:
        if not self.is_game_over():
            raise ValueError("Game is not over")
        # Return per-hand deltas (not cumulative) to represent this hand's outcome
        assert self.game_outcome is not None
        return [int(self.game_outcome.players[i].points_delta) for i in range(4)]

    def _store_game_outcome(self) -> None:
        # Compute once and store a structured outcome
        try:
            self.game_outcome = self._build_game_outcome()
        except Exception:
            # In case of unforeseen errors, keep outcome unset to avoid crashing callers
            self.game_outcome = None

    def _build_game_outcome(self) -> 'GameOutcome':
        # Internal constructor for GameOutcome from current finalized state
        # Use cumulative_points to reflect total delta from start to end of hand, including riichi stick losses
        hand_points = list(self.cumulative_points)
        is_draw = len(self.winners) == 0 and self.loser is None
        players: Dict[int, PlayerOutcome] = {}
        if is_draw:
            # Determine tenpai from current hands to be robust even when 0 or 4 tenpai
            tenpai_flags: Dict[int, bool] = {}
            for pid in range(4):
                tenpai_flags[pid] = hand_is_tenpai(self._player_hands[pid])
            for pid in range(4):
                is_tenpai = bool(tenpai_flags.get(pid, False))
                players[pid] = PlayerOutcome(
                    player_id=pid,
                    outcome_type=OutcomeType.TENPAI if is_tenpai else OutcomeType.NOTEN,
                    won=False,
                    lost=False,
                    tenpai=is_tenpai,
                    noten=not is_tenpai,
                    points_delta=int(hand_points[pid]),
                )
            return GameOutcome(players=players, winners=list(self.winners), loser=self.loser, is_draw=True)

        # Win occurred (tsumo or ron); support multi-ron
        ron = False
        tsumo = False
        if self.winners:
            # If loser is None, it's tsumo; otherwise ron
            ron = self.loser is not None
            tsumo = not ron
        for pid in range(4):
            won = pid in self.winners
            lost = False
            otype: Optional[OutcomeType] = None
            if won:
                otype = OutcomeType.RON if ron else OutcomeType.TSUMO
            else:
                if ron and self.loser == pid:
                    lost = True
                    otype = OutcomeType.DEAL_IN  # discarder lost on ron
                elif tsumo:
                    # All non-winners paid on tsumo
                    lost = True
            players[pid] = PlayerOutcome(
                player_id=pid,
                outcome_type=otype,
                won=won,
                lost=lost,
                tenpai=False,
                noten=False if otype is None else (otype == OutcomeType.NOTEN),
                points_delta=int(hand_points[pid]),
            )
        return GameOutcome(players=players, winners=list(self.winners), loser=self.loser, is_draw=False)

    def get_game_outcome(self) -> 'GameOutcome':
        if not self.game_over:
            raise ValueError("Game is not over")
        if self.game_outcome is None:
            self.game_outcome = self._build_game_outcome()
        return self.game_outcome

    # Scoring API
    def _score_hand(self, winner_id: int, win_by_tsumo: bool) -> Dict[str, Any]:
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
        # Ippatsu: +1 han (yaku) if riichi declared, ippatsu active, and win occurs on next draw before any call or discard
        if self.riichi_declared[winner_id] and self.riichi_ippatsu_active.get(winner_id, False):
            han += IPPATSU_HAN
            # Consumed on win
            self.riichi_ippatsu_active[winner_id] = False

        # Base points use total han already contained in `han` (includes dora/aka per _score_fu_and_han)
        base_points = fu * (2 ** (BASE_POINTS_EXPONENT_OFFSET + han))
        # Apply simple mangan cap for limit hands (≥5 han)
        dealer = (winner_id == DEALER_ID_START)
        if han >= MANGAN_HAN_THRESHOLD:
            if win_by_tsumo:
                if dealer:
                    # Dealer tsumo mangan: 2000 each from three players
                    total = MANGAN_DEALER_TSUMO_PAYMENT_EACH * 3
                    payments = {'total_from_others': total}
                    return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': True, 'payments': payments}
                else:
                    # Non-dealer tsumo mangan: dealer 2000, others 1000 each
                    payments = {'from_dealer': MANGAN_NON_DEALER_TSUMO_DEALER_PAYMENT, 'from_others': MANGAN_NON_DEALER_TSUMO_OTHERS_PAYMENT, 'total_from_all': MANGAN_NON_DEALER_TSUMO_DEALER_PAYMENT + 2 * MANGAN_NON_DEALER_TSUMO_OTHERS_PAYMENT}
                    return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': payments['total_from_all'], 'tsumo': True, 'payments': payments}
            else:
                # Ron mangan
                total = MANGAN_DEALER_RON_POINTS if dealer else MANGAN_NON_DEALER_RON_POINTS
                return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': False, 'from': self.loser}

        # Simplified rounding for non-limit hands
        def round_up_100(x: int) -> int:
            return int(math.ceil(x / float(POINTS_ROUNDING)) * POINTS_ROUNDING)

        if win_by_tsumo:
            if dealer:
                total = round_up_100(base_points * DEALER_TSUMO_TOTAL_MULTIPLIER)
                payments = {'total_from_others': total}
                # Winner collects riichi sticks pot on win
                if self.riichi_sticks_pot > 0:
                    payments['riichi_sticks'] = self.riichi_sticks_pot
                    self.riichi_sticks_pot = 0
            else:
                # Non-dealer split: dealer pays 2x, others 1x
                dealer_pay = round_up_100(base_points * NON_DEALER_TSUMO_DEALER_MULTIPLIER)
                non_dealer_pay = round_up_100(base_points * NON_DEALER_TSUMO_OTHERS_MULTIPLIER)
                total = dealer_pay + 2 * non_dealer_pay
                payments = {'from_dealer': dealer_pay, 'from_others': non_dealer_pay, 'total_from_all': total}
                if self.riichi_sticks_pot > 0:
                    payments['riichi_sticks'] = self.riichi_sticks_pot
                    self.riichi_sticks_pot = 0
            return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': True, 'payments': payments}
        else:
            # Ron
            if dealer:
                total = round_up_100(base_points * DEALER_RON_MULTIPLIER)
            else:
                total = round_up_100(base_points * NON_DEALER_RON_MULTIPLIER)
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


