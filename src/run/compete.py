#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import random
from typing import List, Optional, Any, Dict
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
        ACPlayer.from_directory("models/ac_ppo_best_20250818_150635", player_id=0, temperature=0),
		RecordingHeuristicACPlayer(1),
		RecordingHeuristicACPlayer(2),
		RecordingHeuristicACPlayer(3),
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
		# Map back to original indices
		inv = {ordered_players[i].player_id: i for i in range(4)}
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


def main():
	parser = ArgumentParser(description='Compete four players in MediumJong')
	parser.add_argument('--games', type=int, default=GAMES, help='Number of games to play')
	parser.add_argument('--seed', type=int, default=SEED if SEED is not None else None, help='Random seed')
	parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling seat order each game')
	args = parser.parse_args()

	players = build_players()
	res = play_matches(
		players,
		games=int(args.games),
		shuffle_each_game=(not args.no_shuffle),
		seed=args.seed,
	)
	print('Results (per-player score stats):')
	for i, s in enumerate(res['stats']):
		print(f"Player {i}: mean={s['mean']:.2f} median={s['median']:.2f} std={s['std']:.2f} n={s['n']}")


if __name__ == '__main__':
	main()


