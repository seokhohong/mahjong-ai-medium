#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Tuple

import numpy as np

import core.game

# Ensure src on path (parent of this file is the src directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong
from core.learn import ACNetwork, RecordingACPlayer, RecordingHeuristicACPlayer, ACPlayer
from core.learn.policy_utils import serialize_action
from core import constants
from tqdm import tqdm


def compute_n_step_returns(
    rewards: List[float],
    n_step: int,
    gamma: float,
    values: List[float],
) -> Tuple[List[float], List[float]]:
    """Compute n-step discounted returns and advantages for a single trajectory.

    Returns a tuple (returns, advantages), where advantages = returns - values_t.

    G_t = sum_{k=0..n-1} gamma^k * r_{t+k}, truncated at episode end.
    """
    T = len(rewards)
    n = max(1, int(n_step))
    g = float(gamma)
    returns: List[float] = [0.0] * T
    for t in range(T):
        acc = 0.0
        powg = 1.0
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            acc += powg * float(rewards[idx])
            powg *= g
        returns[t] = acc
    advantages = [float(returns[t]) - float(values[t]) for t in range(T)]
    return returns, advantages


def build_ac_dataset(
    games: int,
    seed: int | None = None,
    temperature: float = 1.0,
    zero_network_reward: bool = False,
    n_step: int = 3,
    gamma: float = 0.99,
    use_heuristic: bool = False,
    model_path: str | None = None,
    negative_reward_weight: float = 1.0,
) -> dict:
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    net = None
    if not use_heuristic:
        if not model_path:
            print("must provide model directory if not using heuristic")
            raise ValueError()

        net = ACPlayer.from_directory(model_path, temperature=temperature).network
        import torch
        net = net.to(torch.device('cpu'))

    # Accumulate per-field arrays for compact storage
    hand_idx_list: List[np.ndarray] = []
    disc_idx_list: List[np.ndarray] = []
    called_idx_list: List[np.ndarray] = []
    game_state_list: List[np.ndarray] = []
    called_discards_list: List[np.ndarray] = []
    flat_idx_list: List[int] = []
    all_returns: List[float] = []
    all_advantages: List[float] = []
    all_old_log_probs: List[float] = []
    all_flat_policies: List[List[float]] = []
    # Flat policy only needs a single old_log_prob per step
    all_game_ids: List[int] = []
    all_step_ids: List[int] = []
    all_actor_ids: List[int] = []

    # Create players once and reuse across games; clear their buffers between episodes
    if use_heuristic:
        players = [RecordingHeuristicACPlayer(i, random_exploration=temperature) for i in range(4)]
    else:
        assert net is not None
        players = [RecordingACPlayer(i, net, temperature=temperature, zero_network_reward=bool(zero_network_reward)) for i in range(4)]

    for gi in tqdm(range(max(1, int(games)))):
        game = MediumJong(players, tile_copies=constants.TILE_COPIES_DEFAULT)
        game.play_round()

        winners = list(game.get_winners()) if hasattr(game, 'get_winners') else []
        loser = game.get_loser() if hasattr(game, 'get_loser') else None
        # Let players update internal bookkeeping if needed
        for p in players:
            try:
                p.finalize_episode(winners, loser)  # type: ignore[attr-defined]
            except Exception:
                pass
        for pid, p in enumerate(players):
            T = len(p.experience)
            if T == 0:
                continue
            rewards = list(p.experience.rewards)
            # Compute terminal reward directly from final per-player deltas via get_points()
            if T > 0:
                import math
                pts = game.get_points()
                my_delta = float(pts[pid])
                if my_delta > 0:
                    terminal_reward = math.log(max(1e-9, my_delta / 1000.0))
                elif my_delta < 0:
                    terminal_reward = -negative_reward_weight * math.log(max(1e-9, (-my_delta) / 1000.0))
                else:
                    terminal_reward = 0.0
                rewards[-1] = float(terminal_reward)
            # Use stored values from the experience buffer
            values: List[float] = [float(v) for v in p.experience.values]
            nstep, adv = compute_n_step_returns(rewards, n_step, gamma, values)
            for t in range(T):
                gs = p.experience.states[t]
                flat_idx = int(p.experience.actions[t])
                # Extract raw indexed features per sample
                hand_idx_list.append(np.asarray(gs['hand_idx'], dtype=np.int32))
                disc_idx_list.append(np.asarray(gs['disc_idx'], dtype=np.int32))
                called_idx_list.append(np.asarray(gs['called_idx'], dtype=np.int32))
                game_state_list.append(np.asarray(gs['game_state'], dtype=np.float32))
                called_discards_list.append(np.asarray(gs['called_discards'], dtype=np.int32))
                flat_idx_list.append(flat_idx)
                all_returns.append(float(nstep[t]))
                all_advantages.append(float(adv[t]))
                all_old_log_probs.append(float(p.experience.main_log_probs[t]))
                # Store full flat policy distribution when available (RecordingACPlayer)
                probs = list(p.experience.main_probs[t])
                all_flat_policies.append(probs)
                # No tile/chi heads in flat policy
                all_game_ids.append(int(gi))
                all_step_ids.append(int(t))
                all_actor_ids.append(int(pid))
        # Clear experience buffers for next game
        for p in players:
            if hasattr(p, 'experience') and p.experience is not None:
                p.experience.clear()



    return {
        'hand_idx': np.asarray(hand_idx_list, dtype=object),
        'disc_idx': np.asarray(disc_idx_list, dtype=object),
        'called_idx': np.asarray(called_idx_list, dtype=object),
        'game_state': np.asarray(game_state_list, dtype=object),
        'called_discards': np.asarray(called_discards_list, dtype=object),
        'flat_idx': np.asarray(flat_idx_list, dtype=np.int64),
        'returns': np.asarray(all_returns, dtype=np.float32),
        'advantages': np.asarray(all_advantages, dtype=np.float32),
        'old_log_probs': np.asarray(all_old_log_probs, dtype=np.float32),
        'game_ids': np.asarray(all_game_ids, dtype=np.int32),
        'step_ids': np.asarray(all_step_ids, dtype=np.int32),
        'actor_ids': np.asarray(all_actor_ids, dtype=np.int32),
        'flat_policies': np.asarray(all_flat_policies, dtype=object),
    }

def save_dataset(built, out_path):
    # Save
    np.savez(
        out_path,
        hand_idx=built['hand_idx'],
        disc_idx=built['disc_idx'],
        called_idx=built['called_idx'],
        game_state=built['game_state'],
        called_discards=built['called_discards'],
        flat_idx=built['flat_idx'],
        returns=built['returns'],
        advantages=built['advantages'],
        old_log_probs=built['old_log_probs'],
        game_ids=built['game_ids'],
        step_ids=built['step_ids'],
        actor_ids=built['actor_ids'],
        flat_policies=built['flat_policies'],
    )

    print(f"Saved AC dataset to {out_path}")

def main():
    ap = argparse.ArgumentParser(description='Create AC experience dataset (states, actions, n-step returns)')
    ap.add_argument('--games', type=int, default=10, help='Number of games to simulate')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--zero_network_reward', action='store_true', help='Zero out network value as immediate reward')
    ap.add_argument('--n_step', type=int, default=3, help='N for n-step returns')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor in (0,1]')
    ap.add_argument('--negative_reward_weight', type=float, default=1.0, help='Multiplier applied to negative terminal rewards (set 0 to ignore losses)')
    ap.add_argument('--out', type=str, default=None, help='Output .npz path (placed under training_data/)')
    ap.add_argument('--use_heuristic', action='store_true', help='Use RecordingHeuristicACPlayer (generation 0)')
    ap.add_argument('--model', type=str, default=None, help='Path to AC network .pt to load')
    args = ap.parse_args()

    built = build_ac_dataset(
        games=args.games,
        seed=args.seed,
        temperature=args.temperature,
        zero_network_reward=bool(args.zero_network_reward),
        n_step=args.n_step,
        gamma=args.gamma,
        use_heuristic=bool(args.use_heuristic),
        model_path=args.model,
        negative_reward_weight=float(args.negative_reward_weight),
    )

    # Prepare output file relative to the current working directory
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = os.path.join(out_dir, args.out)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'ac_{ts}.npz')

    save_dataset(built, out_path)
    return out_path



if __name__ == '__main__':
    main()


