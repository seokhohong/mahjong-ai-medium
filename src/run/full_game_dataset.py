#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure src on path (parent of this file is the src directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong, GameOutcome, CalledSet
from core.tile import Tile, Suit, TileType, Honor, encode_tile, encode_tiles
from core.action import Action, Reaction, Discard, Riichi, Tsumo, Ron, Pon, Chi, KanDaimin, KanKakan, KanAnkan, encode_move
from core.learn.recording_ac_player import (
    RecordingACPlayer,
    RecordingHeuristicACPlayer,
)
from core.learn.policy_utils import encode_two_head_action
from core import constants


# (moved canonical serialization to core.tile and core.action)


# --------------------
# Extended record types
# --------------------
@dataclass
class RecordedStep:
    actor_id: int
    move: Dict[str, Any]  # {'type': 'Discard'|'Riichi'|'Tsumo'|'Ron'|'Pon'|'Chi'|'KanDaimin'|'KanKakan'|'KanAnkan', ...}
    action_idx: int
    tile_idx: int
    value: float
    joint_logp: float


@dataclass
class MediumJongExtended:
    # Complete initial snapshot required for deterministic reconstruction
    initial: Dict[str, Any]
    # Chronological step log
    steps: List[RecordedStep]
    # Final structured outcome
    outcome: Dict[str, Any]


# =========================
# In-code player configuration
# =========================
def build_prebuilt_players() -> List[Union[RecordingACPlayer, RecordingHeuristicACPlayer]]:
    """Return a fixed set of 4 Recording* players for full-game recording.

    Edit this function to switch between heuristic and AC players. By default,
    returns heuristic players (deterministic unless your heuristic explores).
    """
    # Example A: heuristic players
    return [RecordingHeuristicACPlayer(random_exploration=0.0) for _ in range(4)]

    # Example B: AC players (uncomment and adjust model path)
    # from core.learn.recording_ac_player import ACPlayer  # available via that module
    # temperature = 0.8
    # net = ACPlayer.from_directory("models/ac_ppo_YYYYMMDD_HHMMSS", temperature=temperature).network
    # import torch
    # net = net.to(torch.device('cpu'))
    # return [RecordingACPlayer(net, temperature=temperature) for _ in range(4)]


def _snapshot_initial(game: MediumJong) -> Dict[str, Any]:
    """Minimal initial state to deterministically rehydrate.

    We only need what cannot be derived from the action log:
    - Initial concealed hands (post-deal)
    - Remaining wall after deal (game.tiles)
    - Dead wall order (game.dead_wall)
    - Round/seat winds (in case they change in future variants)
    - Whose turn and whether it's action vs reaction phase
    """
    return {
        'current_player_idx': int(game.current_player_idx),
        'next_move_is_action': bool(getattr(game, '_next_move_is_action', True)),
        'player_hands': {i: encode_tiles(game._player_hands[i]) for i in range(4)},
        'tiles': encode_tiles(list(getattr(game, 'tiles'))),
        'dead_wall': encode_tiles(list(getattr(game, 'dead_wall'))),
        'round_wind': int(game.round_wind.value),
        'seat_winds': {i: int(game.seat_winds[i].value) for i in range(4)},
    }


# Use encode_move from core.action

class _LoggingProxyPlayer:
    """Wrap a Recording* player to capture a chronological log with actor ids.

    The engine will call these proxy instances. We forward to the underlying player,
    then record the chosen move plus the player's tail value/logp into a shared list.
    """
    def __init__(self, actor_id: int, inner: Union[RecordingACPlayer, RecordingHeuristicACPlayer], sink: List[RecordedStep]):
        self.actor_id = int(actor_id)
        self.inner = inner
        self._sink = sink

    def play(self, game_state):  # type: ignore[override]
        mv = self.inner.act(game_state)
        a_idx, t_idx = encode_two_head_action(mv)
        value = 0.0
        jlp = 0.0
        try:
            exp = self.inner.experience
            if len(exp) > 0:
                value = float(exp.values[-1])
                jlp = float(exp.joint_log_probs[-1])
        except Exception:
            pass
        self._sink.append(RecordedStep(
            actor_id=self.actor_id,
            move=encode_move(mv),
            action_idx=int(a_idx),
            tile_idx=int(t_idx),
            value=float(value),
            joint_logp=float(jlp),
        ))
        return mv

    def choose_reaction(self, game_state, options):  # type: ignore[override]
        mv = self.inner.react(game_state, options)
        a_idx, t_idx = encode_two_head_action(mv)
        value = 0.0
        jlp = 0.0
        try:
            exp = self.inner.experience
            if len(exp) > 0:
                value = float(exp.values[-1])
                jlp = float(exp.joint_log_probs[-1])
        except Exception:
            pass
        self._sink.append(RecordedStep(
            actor_id=self.actor_id,
            move=encode_move(mv),
            action_idx=int(a_idx),
            tile_idx=int(t_idx),
            value=float(value),
            joint_logp=float(jlp),
        ))
        return mv


def build_games_pickled(
    *,
    games: int,
    seed: Optional[int] = None,
    prebuilt_players: Optional[Sequence[Union[RecordingACPlayer, RecordingHeuristicACPlayer]]] = None,
) -> List[MediumJongExtended]:
    import random
    if seed is not None:
        random.seed(int(seed))
    records: List[MediumJongExtended] = []

    # Determine players: prefer provided prebuilt, otherwise call local factory
    if prebuilt_players is None:
        prebuilt_players = build_prebuilt_players()
    if prebuilt_players is None:
        raise ValueError("Please configure build_prebuilt_players() to return 4 Recording* players")
    inner_players = list(prebuilt_players)

    for _ in range(max(1, int(games))):
        step_sink: List[RecordedStep] = []
        proxy_players = [_LoggingProxyPlayer(i, inner_players[i], step_sink) for i in range(4)]

        game = MediumJong(proxy_players, tile_copies=constants.TILE_COPIES_DEFAULT)
        initial = _snapshot_initial(game)

        # Drive the game until completion using engine's flow
        guard = 0
        while not game.is_game_over() and guard < 1000:
            game.play_turn()
            guard += 1
        steps = step_sink
        # Assign terminal rewards to last decisions based on points
        if game.is_game_over():
            try:
                pts = game.get_points()
                def _reward_fn(delta: int) -> float:
                    return float(delta) / 10000.0
                for pid, p in enumerate(inner_players):
                    if len(p.experience) == 0:
                        continue
                    p.finalize_episode(_reward_fn(pts[pid]))  # type: ignore[attr-defined]
            except Exception:
                pass
        # Outcome
        outcome = game.get_game_outcome().serialize()  # type: ignore[attr-defined]
        records.append(MediumJongExtended(initial=initial, steps=steps, outcome=outcome))

        # Clear player experiences for next game reuse
        try:
            for p in inner_players:
                if hasattr(p, 'experience') and p.experience is not None:
                    p.experience.clear()
        except Exception:
            pass

    return records


def save_games_pickle(records: List[MediumJongExtended], out_path: str) -> None:
    with open(out_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(records)} games to {out_path}")


def main() -> str:
    ap = argparse.ArgumentParser(description='Deterministic full-game recorder (players configured in-code via build_prebuilt_players)')
    ap.add_argument('--games', type=int, default=10)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    records = build_games_pickled(
        games=int(max(1, args.games)),
        seed=args.seed,
        prebuilt_players=build_prebuilt_players(),
    )

    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = os.path.join(out_dir, args.out)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'mj_extended_{ts}.pkl')

    save_games_pickle(records, out_path)
    return out_path


if __name__ == '__main__':
    raise SystemExit(main())
