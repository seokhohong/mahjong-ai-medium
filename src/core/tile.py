from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

from .constants import SUIT_ORDER


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
                Honor.WHITE: 'Wh', Honor.GREEN: 'G', Honor.RED: 'R',
            }
            return mapping[self.tile_type]  # type: ignore[index]
        # Represent aka (red-dora) five as 0{suite} per common shorthand, e.g., "0p"
        if self.aka and self.suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU) and self.tile_type == TileType.FIVE:
            return f"0{self.suit.value}"
        return f"{int(self.tile_type.value)}{self.suit.value}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tile) and self.suit == other.suit and self.tile_type == other.tile_type

    def __hash__(self) -> int:
        return hash((self.suit, self.tile_type))


def _tile_sort_key(t: Tile) -> Tuple[int, int]:
    return (SUIT_ORDER[t.suit.value], int(t.tile_type.value))


def make_tile(suit: Suit, val: int) -> Tile:
    return Tile(suit, TileType(val)) if suit != Suit.HONORS else Tile(Suit.HONORS, Honor(val))


def _dora_next(tile: Tile) -> Tile:
    # Next tile cycling within suit/honors for dora mapping
    if tile.suit == Suit.HONORS:
        order = [Honor.EAST, Honor.SOUTH, Honor.WEST, Honor.NORTH, Honor.WHITE, Honor.GREEN, Honor.RED]
        idx = order.index(tile.tile_type)  # type: ignore[arg-type]
        return Tile(Suit.HONORS, order[(idx + 1) % len(order)])
    v = int(tile.tile_type.value)
    nv = 1 if v == 9 else v + 1
    return Tile(tile.suit, TileType(nv))


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
