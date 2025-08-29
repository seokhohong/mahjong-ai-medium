#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Any, Optional

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong, Player  # type: ignore
from core.tile import Suit, Tile
from core.learn.ac_player import ACPlayer  # type: ignore
from core.learn.recording_ac_player import RecordingHeuristicACPlayer  # type: ignore
from core.action import Reaction, Pon, Chi, KanDaimin


def _fmt_tile(t: Any) -> str:
    try:
        return str(t)
    except Exception:
        return f"{t}"


def _infer_call_type(tiles: List[Tile]) -> str:
    try:
        if not tiles:
            return 'set'
        # All identical: pon (3) or kan (4)
        if all(t.suit == tiles[0].suit and t.tile_type == tiles[0].tile_type for t in tiles):
            return 'kan' if len(tiles) == 4 else 'pon'
        # Suited run (sequence): chi
        suits = {t.suit for t in tiles}
        if len(suits) == 1 and tiles[0].suit != Suit.HONORS:
            vals = sorted(int(t.tile_type.value) for t in tiles)
            if len(vals) == 3 and vals[1] == vals[0] + 1 and vals[2] == vals[1] + 1:
                return 'chi'
        return 'set'
    except Exception:
        return 'set'


def _fmt_called_sets(csets) -> str:
    try:
        parts: List[str] = []
        for cs in csets or []:
            tiles = getattr(cs, 'tiles', [])
            ctype = getattr(cs, 'call_type', None)
            if not ctype or ctype == 'chi':
                ctype = _infer_call_type(tiles)
            parts.append(f"{ctype}:[" + ', '.join(_fmt_tile(t) for t in tiles) + "]")
        return '[' + '; '.join(parts) + ']'
    except Exception:
        return '[]'


class LoggingPlayer(Player):
    def __init__(self, inner: Player, identifier: Optional[int] = None):
        super().__init__(identifier=identifier)
        self.inner = inner
        self._abs_player_id: Any = '?'

    def set_abs_player_id(self, pid: int) -> None:
        self._abs_player_id = pid

    def act(self, gs):  # type: ignore[override]
        move = self.inner.act(gs)
        # Pretty print current perspective and chosen move
        # GamePerspective is rotated so that index 0 is always the current (self) player
        actor = 0  # local index in perspective
        # Prefer engine-provided player_id on the perspective when available
        abs_pid = self._abs_player_id
        hand_s = '[' + ', '.join(_fmt_tile(t) for t in sorted(gs.player_hand, key=lambda t: (t.suit.value, int(t.tile_type.value)))) + ']'
        called_s = {pid: _fmt_called_sets(gs.called_sets.get(pid, [])) for pid in range(4)}
        disc_s = {pid: '[' + ', '.join(_fmt_tile(t) for t in gs.player_discards.get(pid, [])) + ']' for pid in range(4)}
        last_discard = _fmt_tile(gs._reactable_tile) if gs._reactable_tile is not None else 'None'
        riichi_flag = int(getattr(gs, 'riichi_declared', {}).get(actor, False))
        print(f"P{abs_pid} (loc {actor}) | RW:{gs.round_wind.name} | SW[" + ', '.join(gs.seat_winds[j].name for j in range(4)) + f"] | Riichi:{riichi_flag} | Hand {hand_s}")
        print(f"    Reactable Tile: {last_discard} by {gs._owner_of_reactable_tile} | CanCall:{int(bool(gs.can_call))} | CanRon:{int(gs.can_ron())} | CanTsumo:{int(gs.can_tsumo())}")
        print(f"    Called: {called_s}")
        print(f"    Discards: {disc_s}")
        print(f"    Action: {type(move).__name__}: {move}")
        return move

    def react(self, gs, options: List[Reaction]):  # type: ignore[override]
        move = self.inner.react(gs, options)
        # Pretty print reaction decision
        actor = 0
        abs_pid = self._abs_player_id
        last_discard = _fmt_tile(gs._reactable_tile) if gs._reactable_tile is not None else 'None'
        # Format options succinctly
        def _fmt_opts():
            try:
                parts: list[str] = []
                if gs.can_ron():
                    parts.append('Ron')
                if options:
                    pon_sets = [r.tiles for r in options if isinstance(r, Pon)]
                    chi_sets = [r.tiles for r in options if isinstance(r, Chi)]
                    kan_sets = [r.tiles for r in options if isinstance(r, KanDaimin)]
                    if pon_sets:
                        parts.append('Pon(' + ' | '.join('[' + ', '.join(_fmt_tile(t) for t in ts) + ']' for ts in pon_sets) + ')')
                    if chi_sets:
                        parts.append('Chi(' + ' | '.join('[' + ', '.join(_fmt_tile(t) for t in ts) + ']' for ts in chi_sets) + ')')
                    if kan_sets:
                        parts.append('KanDaimin(' + ' | '.join('[' + ', '.join(_fmt_tile(t) for t in ts) + ']' for ts in kan_sets) + ')')
                return ', '.join(parts) if parts else 'None'
            except Exception:
                return 'None'
        print(f"P{abs_pid} (loc {actor}) | ReactionState | LastDiscard:{last_discard} by {gs._owner_of_reactable_tile} | Options: {_fmt_opts()}")
        print(f"    Reaction:{type(move).__name__}: {move}")
        return move


def build_players(model_dir: str | None, temperature: float) -> List[Player]:
    players: List[Player] = []
    if model_dir:
        players.append(ACPlayer.from_directory(model_dir, temperature=float(temperature)))
    else:
        players.append(RecordingHeuristicACPlayer(random_exploration=0.0))
    players.extend([RecordingHeuristicACPlayer(random_exploration=0.0),
                    RecordingHeuristicACPlayer(random_exploration=0.0),
                    RecordingHeuristicACPlayer(random_exploration=0.0)])
    # Wrap with logging
    return [LoggingPlayer(p) for p in players]


def main() -> int:
    ap = argparse.ArgumentParser(description='Run a single MediumJong game and pretty print each step')
    ap.add_argument('--model', type=str, default=None, help='Path to AC model directory (optional)')
    ap.add_argument('--temperature', type=float, default=0.1, help='Temperature for AC model')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    args = ap.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    players = build_players(args.model, float(args.temperature))
    game = MediumJong(players)
    # Initialize absolute player ids on loggers now that the game exists
    for p in game.players:
        if isinstance(p, LoggingPlayer):
            try:
                p.set_abs_player_id(game.get_player_id(p))
            except Exception:
                pass

    # Print winds at start
    print("=" * 80)
    print("Start Game | RW:" + game.round_wind.name + " | SW[" + ', '.join(game.seat_winds[j].name for j in range(4)) + "]")
    print("-" * 80)

    while not game.is_game_over():
        game.play_turn()

    print("-" * 80)
    print("Game Over")
    pts = game.get_points()
    if pts is not None:
        for pid, p in enumerate(pts):
            print(f"Player {pid} points delta: {int(p)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


