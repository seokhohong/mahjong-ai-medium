#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

from core.heuristics_player import MediumHeuristicsPlayer

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.game import MediumJong, Player, GamePerspective  # type: ignore
from core.action import Reaction
from core.tile import Tile
from core.learn.ac_player import ACPlayer  # type: ignore


class PlayalongACPlayer(ACPlayer):
    """
    ACPlayer variant that, at every decision (action or reaction), compares its
    chosen move to what the baseline `Player` would do in the same position.

    Tracks cumulative accuracy across all decisions made so far.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._baseline = MediumHeuristicsPlayer()
        self._total_decisions: int = 0
        self._correct_decisions: int = 0

    # --- helpers ---
    def _record_accuracy(self, gs: GamePerspective, chosen_move: Any) -> None:
        try:
            baseline_move = self._baseline.play(gs)
        except Exception:
            # Fallback: if baseline call fails for some reason, skip counting
            return
        self._total_decisions += 1
        if self._moves_equal(chosen_move, baseline_move):
            self._correct_decisions += 1

    def _record_reaction_accuracy(self, gs: GamePerspective, options: List[Reaction], chosen_move: Any) -> None:
        try:
            baseline_move = self._baseline.choose_reaction(gs, options)
        except Exception:
            return
        self._total_decisions += 1
        if self._moves_equal(chosen_move, baseline_move):
            self._correct_decisions += 1

    def _moves_equal(self, a: Any, b: Any) -> bool:
        # Dataclasses should compare by value; fall back to repr comparison if needed
        try:
            return a == b
        except Exception:
            return repr(a) == repr(b)

    # --- overrides ---
    def play(self, game_state: GamePerspective):
        move = self.compute_play(game_state)[0]
        self._record_accuracy(game_state, move)
        return move

    def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]) -> Reaction:
        move = self.compute_play(game_state)[0]
        self._record_reaction_accuracy(game_state, options, move)
        return move

    # --- API ---
    def accuracy(self) -> float:
        if self._total_decisions == 0:
            return 0.0
        return float(self._correct_decisions) / float(self._total_decisions)

    def counts(self) -> Dict[str, int]:
        return {"correct": int(self._correct_decisions), "total": int(self._total_decisions)}


def run_playalong(num_games: int, model_dir: Optional[str], temperature: float) -> float:
    """
    Run num_games where player 0 is PlayalongACPlayer and others are baseline Players.
    Returns final accuracy of PlayalongACPlayer.
    """
    # Build our playalong AC player (from model if provided, else default)
    if model_dir:
        base_ac: ACPlayer = ACPlayer.from_directory(model_dir, temperature=temperature)
        pac = PlayalongACPlayer(network=base_ac.network, gsv_scaler=base_ac.gsv_scaler, temperature=temperature)
    else:
        base_ac = ACPlayer.default(temperature=temperature)
        pac = PlayalongACPlayer(network=base_ac.network, gsv_scaler=base_ac.gsv_scaler, temperature=temperature)

    for _ in range(max(1, int(num_games))):
        players: List[Player] = [pac] + [Player() for _ in range(3)]
        game = MediumJong(players)
        game.play_round()

    acc = pac.accuracy()
    c = pac.counts()
    print(f"Playalong accuracy: {acc:.4f} ({c['correct']}/{c['total']}) over {num_games} game(s)")
    return acc


def main() -> float:
    ap = argparse.ArgumentParser(description="Run PlayalongACPlayer for N games and report accuracy vs baseline Player")
    ap.add_argument("--games", type=int, default=10, help="Number of games to simulate")
    ap.add_argument("--model", type=str, default=None, help="Path to model directory (containing .pt and scaler.pkl)")
    ap.add_argument("--temperature", type=float, default=1, help="Sampling temperature for AC policy")
    args = ap.parse_args()

    return run_playalong(num_games=args.games, model_dir=args.model, temperature=args.temperature)


if __name__ == "__main__":
    main()
