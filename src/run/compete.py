#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import random
import multiprocessing as mp
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from argparse import ArgumentParser

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong, Player  # type: ignore
from core.learn.ac_network import ACNetwork  # type: ignore
from core.learn.ac_player import ACPlayer  # type: ignore
from core.learn.recording_ac_player import RecordingHeuristicACPlayer  # type: ignore
from tqdm import tqdm

# =========================
# USER-EDITABLE CONFIG BELOW
# =========================

# Default values (can be overridden via CLI)
GAMES: int = 100
SHUFFLE_EACH_GAME: bool = True
SEED: Optional[int] = None


"""Instantiate and return exactly four Players.

Edit this function to change which agents compete. Examples:

- Four heuristics:
    return [RecordingHeuristicACPlayer(i) for i in range(4)]

"""

def build_players() -> List[Player]:
	from core.learn.data_utils import load_gsv_scaler

	return [
        ACPlayer.from_directory("models/ac_ppo_20250825_181954", temperature=0.2),
        ACPlayer.from_directory("models/ac_ppo_20250825_164819", temperature=0.2),
        ACPlayer.from_directory("models/ac_ppo_20250825_172005", temperature=0.2),
        ACPlayer.from_directory("models/ac_ppo_20250825_174835", temperature=0.2)

	]



def play_matches(players: List[Player], games: int, *, shuffle_each_game: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
	if seed is not None:
		random.seed(int(seed))

	assert len(players) == 4, "Exactly 4 players are required"
	# Track per-game scores (point deltas) for each original player index
	per_player_scores: List[List[float]] = [[] for _ in range(4)]

	for gi in tqdm(range(max(1, int(games)))):
		order = list(range(4))
		if shuffle_each_game:
			random.shuffle(order)
			ordered_players = [players[i] for i in order]
		else:
			ordered_players = players

		game = MediumJong(ordered_players)
		game.play_round()
		assert game.is_game_over()
		# Aggregate results
		final_points = game.get_points()
		for i in range(4):
			per_player_scores[order[i]].append(float(final_points[i]))

	# Compute summary stats
	stats = []
	for i in range(4):
		arr = np.asarray(per_player_scores[i], dtype=np.float64)
		mean = float(arr.mean()) if arr.size > 0 else 0.0
		median = float(np.median(arr)) if arr.size > 0 else 0.0
		std = float(arr.std(ddof=0)) if arr.size > 0 else 0.0
		stats.append({'mean': mean, 'median': median, 'std': std, 'n': int(arr.size)})

	return {
		'per_player_scores': per_player_scores,
		'stats': stats,
	}


def _worker_play_chunk(games: int, seed: Optional[int], shuffle_each_game: bool, queue: mp.Queue, rank: int) -> None:
    """Worker process: build its own players and play a chunk of games, send results via queue."""
    try:
        # Derive per-process seed for variability if base seed provided
        proc_seed = None if seed is None else (int(seed) + int(rank) * 100003)
        if proc_seed is not None:
            random.seed(proc_seed)
            np.random.seed(proc_seed)

        players = build_players()
        per_player_scores: List[List[float]] = [[] for _ in range(4)]

        iterator = range(max(1, int(games)))
        if rank == 0:
            iterator = tqdm(iterator, desc=f"Worker {rank} games", dynamic_ncols=True, mininterval=0.1)

        for _ in iterator:
            order = list(range(4))
            if shuffle_each_game:
                random.shuffle(order)
                ordered_players = [players[i] for i in order]
            else:
                ordered_players = players

            game = MediumJong(ordered_players)
            game.play_round()
            assert game.is_game_over()
            final_points = game.get_points()
            for i in range(4):
                per_player_scores[order[i]].append(float(final_points[i]))

        queue.put((rank, per_player_scores))
    except Exception as e:
        try:
            queue.put((rank, e))
        except Exception:
            pass


def play_matches_parallel(total_games: int, *, num_processes: int = 1, chunk_size: int = 50, shuffle_each_game: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run matches in multiple processes and aggregate results.

    Each worker builds its own players to avoid pickling issues.
    """
    num_processes = max(1, int(num_processes))
    if num_processes == 1:
        return play_matches(build_players(), games=int(total_games), shuffle_each_game=shuffle_each_game, seed=seed)

    # Partition work into chunks
    chunks: List[Tuple[int, int]] = []  # (rank, games_in_chunk)
    remaining = max(1, int(total_games))
    rank = 0
    while remaining > 0:
        g = min(int(chunk_size), remaining)
        chunks.append((rank, g))
        remaining -= g
        rank += 1

    queue: mp.Queue = mp.Queue()
    procs: List[Tuple[mp.Process, int]] = []
    next_chunk_idx = 0
    results_received = 0
    agg_scores: List[List[float]] = [[] for _ in range(4)]

    def _start_chunk(idx: int):
        rnk, games_in_chunk = chunks[idx]
        p = mp.Process(target=_worker_play_chunk, args=(int(games_in_chunk), seed, shuffle_each_game, queue, int(rnk)))
        p.daemon = False
        p.start()
        procs.append((p, rnk))

    while next_chunk_idx < len(chunks) and len(procs) < num_processes:
        _start_chunk(next_chunk_idx)
        next_chunk_idx += 1

    while results_received < len(chunks):
        try:
            rnk, payload = queue.get(timeout=1.0)
            results_received += 1
            if isinstance(payload, Exception):
                raise payload
            per_player_scores = payload  # type: ignore
            for i in range(4):
                agg_scores[i].extend(per_player_scores[i])
        except Exception:
            pass

        alive: List[Tuple[mp.Process, int]] = []
        for p, r in procs:
            if p.is_alive():
                alive.append((p, r))
            else:
                try:
                    p.join(timeout=0.1)
                except Exception:
                    pass
        procs = alive

        while next_chunk_idx < len(chunks) and len(procs) < num_processes:
            _start_chunk(next_chunk_idx)
            next_chunk_idx += 1

    for p, _ in procs:
        try:
            p.join(timeout=0.1)
        except Exception:
            pass

    stats = []
    for i in range(4):
        arr = np.asarray(agg_scores[i], dtype=np.float64)
        mean = float(arr.mean()) if arr.size > 0 else 0.0
        median = float(np.median(arr)) if arr.size > 0 else 0.0
        std = float(arr.std(ddof=0)) if arr.size > 0 else 0.0
        stats.append({'mean': mean, 'median': median, 'std': std, 'n': int(arr.size)})

    return {
        'per_player_scores': agg_scores,
        'stats': stats,
    }


def main():
    parser = ArgumentParser(description='Compete four players in MediumJong (supports multiprocessing)')
    parser.add_argument('--games', type=int, default=GAMES, help='Number of games to play')
    parser.add_argument('--seed', type=int, default=SEED if SEED is not None else None, help='Random seed')
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling seat order each game')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of worker processes (1 = serial)')
    parser.add_argument('--chunk_size', type=int, default=50, help='Games per worker chunk')
    args = parser.parse_args()

    if int(args.num_processes) <= 1:
        players = build_players()
        res = play_matches(
            players,
            games=int(args.games),
            shuffle_each_game=(not args.no_shuffle),
            seed=args.seed,
        )
    else:
        # Set safe start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        res = play_matches_parallel(
            total_games=int(args.games),
            num_processes=int(args.num_processes),
            chunk_size=int(max(1, args.chunk_size)),
            shuffle_each_game=(not args.no_shuffle),
            seed=args.seed,
        )

    print('Results (per-player score stats):')
    for i, s in enumerate(res['stats']):
        print(f"Player {i}: mean={s['mean']:.2f} median={s['median']:.2f} std={s['std']:.2f} n={s['n']}")


if __name__ == '__main__':
	main()
