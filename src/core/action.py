from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .tile import Tile


@dataclass
class Action:
    ...


@dataclass
class Reaction:
    ...


@dataclass
class Tsumo(Action):
    ...


@dataclass
class Ron(Reaction):
    ...


@dataclass
class PassCall(Reaction):
    ...


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
    chi_variant_index: int


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
