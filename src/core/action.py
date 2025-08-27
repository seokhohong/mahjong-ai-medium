from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Union

from .tile import Tile, encode_tile, encode_tiles, decode_tile, decode_tiles


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
    

# -----------------------------
# Canonical serialization utils
# -----------------------------
Move = Union[Action, Reaction]


def encode_move(m: Move) -> Dict[str, Any]:
    """Encode any Action/Reaction to a serializable dict."""
    if isinstance(m, Discard):
        return { 'type': 'Discard', 'tile': encode_tile(m.tile) }
    if isinstance(m, Riichi):
        return { 'type': 'Riichi', 'tile': encode_tile(m.tile) }
    if isinstance(m, Tsumo):
        return { 'type': 'Tsumo' }
    if isinstance(m, Ron):
        return { 'type': 'Ron' }
    if isinstance(m, Pon):
        return { 'type': 'Pon', 'tiles': encode_tiles(m.tiles) }
    if isinstance(m, Chi):
        return { 'type': 'Chi', 'tiles': encode_tiles(m.tiles), 'chi_variant_index': int(m.chi_variant_index) }
    if isinstance(m, KanDaimin):
        return { 'type': 'KanDaimin', 'tiles': encode_tiles(m.tiles) }
    if isinstance(m, KanKakan):
        return { 'type': 'KanKakan', 'tile': encode_tile(m.tile) }
    if isinstance(m, KanAnkan):
        return { 'type': 'KanAnkan', 'tile': encode_tile(m.tile) }
    return { 'type': type(m).__name__ }


def decode_move(d: Dict[str, Any]) -> Move:
    t = d.get('type')
    if t == 'Discard':
        return Discard(tile=decode_tile(d.get('tile')))  # type: ignore[arg-type]
    if t == 'Riichi':
        return Riichi(tile=decode_tile(d.get('tile')))  # type: ignore[arg-type]
    if t == 'Tsumo':
        return Tsumo()
    if t == 'Ron':
        return Ron()
    if t == 'Pon':
        return Pon(tiles=decode_tiles(d.get('tiles', [])))
    if t == 'Chi':
        return Chi(tiles=decode_tiles(d.get('tiles', [])), chi_variant_index=int(d.get('chi_variant_index', -1)))
    if t == 'KanDaimin':
        return KanDaimin(tiles=decode_tiles(d.get('tiles', [])))
    if t == 'KanKakan':
        return KanKakan(tile=decode_tile(d.get('tile')))  # type: ignore[arg-type]
    if t == 'KanAnkan':
        return KanAnkan(tile=decode_tile(d.get('tile')))  # type: ignore[arg-type]
    # Fallback to a basic Action
    return Action()  # type: ignore[return-value]
