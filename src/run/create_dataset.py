#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Tuple, Optional, Sequence

import numpy as np

# Ensure src on path (parent of this file is the src directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong
from core.learn.ac_player import ACNetwork
from core.learn.recording_ac_player import RecordingACPlayer, RecordingHeuristicACPlayer, ACPlayer
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

def reward_function(points_delta):
    return points_delta / 10000.0

def build_ac_dataset(
    games: int,
    seed: int | None = None,
    temperature: float = 1.0,
    zero_network_reward: bool = False,
    n_step: int = 3,
    gamma: float = 0.99,
    use_heuristic: bool = False,
    model_path: str | None = None,
    prebuilt_players: Optional[Sequence[RecordingACPlayer | RecordingHeuristicACPlayer]] = None,
) -> dict:
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(int(time.time()))

    net = None
    # Only load a network if we are not given prebuilt players and not using heuristic
    if prebuilt_players is None and not use_heuristic:
        if not model_path:
            print("must provide model directory if not using heuristic")
            raise ValueError()

        net = ACPlayer.from_directory(model_path, temperature=temperature).network
        import torch
        
        # Choose device: prefer CUDA, then macOS MPS, else CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        #elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #    device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        net = net.to(device)

    # Accumulate per-field arrays for compact storage
    hand_idx_list: List[np.ndarray] = []
    disc_idx_list: List[np.ndarray] = []
    called_idx_list: List[np.ndarray] = []
    game_state_list: List[np.ndarray] = []
    called_discards_list: List[np.ndarray] = []
    action_idx_list: List[int] = []
    tile_idx_list: List[int] = []
    all_returns: List[float] = []
    all_advantages: List[float] = []
    joint_log_probs: List[float] = []
    # Two-head policy stores separate log-probs for each head
    all_game_ids: List[int] = []
    all_step_ids: List[int] = []
    all_actor_ids: List[int] = []
    # Per-game metadata (structured only)
    game_outcomes_obj: List[dict] = []  # serialized GameOutcome per game

    # Create players once and reuse across games; clear their buffers between episodes
    if prebuilt_players is not None:
        players = list(prebuilt_players)
    else:
        if use_heuristic:
            players = [RecordingHeuristicACPlayer(random_exploration=temperature) for i in range(4)]
        else:
            assert net is not None
            players = [RecordingACPlayer(net, temperature=temperature, zero_network_reward=bool(zero_network_reward)) for i in range(4)]
            #players = [RecordingACPlayer(net, temperature=temperature, zero_network_reward=bool(zero_network_reward))] * 4
    # Use robust tqdm settings so progress continues to render in long runs
    for gi in tqdm(range(max(1, int(games))), dynamic_ncols=True, mininterval=0.1, miniters=1, leave=True):
        game = MediumJong(players, tile_copies=constants.TILE_COPIES_DEFAULT)
        game.play_round()

        # Structured outcome from the game
        outcome = game.get_game_outcome()
        game_outcomes_obj.append(outcome.serialize())
        # Compute terminal reward from points delta and assign directly to last reward slot
        pts = game.get_points()
        for pid, p in enumerate(players):
            if len(p.experience) == 0:
                continue
            terminal_reward = reward_function(pts[pid])
            p.finalize_episode(float(terminal_reward))  # type: ignore[attr-defined]
        for pid, p in enumerate(players):
            T = len(p.experience)
            if T == 0:
                continue
            rewards = list(p.experience.rewards)
            # Use stored values from the experience buffer
            values: List[float] = [float(v) for v in p.experience.values]
            nstep, adv = compute_n_step_returns(rewards, n_step, gamma, values)
            
            # Collect all data for this player's episode in batches
            player_hand_idx = []
            player_disc_idx = []
            player_called_idx = []
            player_game_state = []
            player_called_discards = []
            player_action_idx = []
            player_tile_idx = []
            player_returns = []
            player_advantages = []
            player_joint_log_probs = []
            player_game_ids = []
            player_step_ids = []
            player_actor_ids = []
            
            for t in range(T):
                gs = p.experience.states[t]
                a_idx, t_idx = p.experience.actions[t]
                # Extract raw indexed features per sample
                player_hand_idx.append(np.asarray(gs['hand_idx'], dtype=np.int32))
                player_disc_idx.append(np.asarray(gs['disc_idx'], dtype=np.int32))
                player_called_idx.append(np.asarray(gs['called_idx'], dtype=np.int32))
                player_game_state.append(np.asarray(gs['game_state'], dtype=np.float32))
                player_called_discards.append(np.asarray(gs['called_discards'], dtype=np.int32))
                player_action_idx.append(int(a_idx))
                player_tile_idx.append(int(t_idx))
                player_returns.append(float(nstep[t]))
                player_advantages.append(float(adv[t]))
                player_joint_log_probs.append(float(p.experience.joint_log_probs[t]))
                player_game_ids.append(int(gi))
                player_step_ids.append(int(t))
                player_actor_ids.append(int(pid))
            
            # Use extend to add all player data at once
            hand_idx_list.extend(player_hand_idx)
            disc_idx_list.extend(player_disc_idx)
            called_idx_list.extend(player_called_idx)
            game_state_list.extend(player_game_state)
            called_discards_list.extend(player_called_discards)
            action_idx_list.extend(player_action_idx)
            tile_idx_list.extend(player_tile_idx)
            all_returns.extend(player_returns)
            all_advantages.extend(player_advantages)
            joint_log_probs.extend(player_joint_log_probs)
            all_game_ids.extend(player_game_ids)
            all_step_ids.extend(player_step_ids)
            all_actor_ids.extend(player_actor_ids)
            
            # Clear experience buffers for next game
            p.experience.clear()



    # Final safety: ensure no lingering data in player buffers after batch completes
    try:
        import gc  # local import to keep top clean
        for p in players:
            if hasattr(p, 'experience') and p.experience is not None:
                p.experience.clear()
        # Only drop players if we created them internally (not when passed-in reusable players)
        if prebuilt_players is None:
            del players
        gc.collect()
    except Exception:
        pass

    return {
        'hand_idx': np.asarray(hand_idx_list, dtype=object),
        'disc_idx': np.asarray(disc_idx_list, dtype=object),
        'called_idx': np.asarray(called_idx_list, dtype=object),
        'game_state': np.asarray(game_state_list, dtype=object),
        'called_discards': np.asarray(called_discards_list, dtype=object),
        'action_idx': np.asarray(action_idx_list, dtype=np.int64),
        'tile_idx': np.asarray(tile_idx_list, dtype=np.int64),
        'returns': np.asarray(all_returns, dtype=np.float32),
        'advantages': np.asarray(all_advantages, dtype=np.float32),
        'joint_log_probs': np.asarray(joint_log_probs, dtype=np.float32),
        'game_ids': np.asarray(all_game_ids, dtype=np.int32),
        'step_ids': np.asarray(all_step_ids, dtype=np.int32),
        'actor_ids': np.asarray(all_actor_ids, dtype=np.int32),
        # Per-game metadata
        'game_outcomes_obj': np.asarray(game_outcomes_obj, dtype=object),
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
        action_idx=built['action_idx'],
        tile_idx=built['tile_idx'],
        returns=built['returns'],
        advantages=built['advantages'],
        joint_log_probs=built['joint_log_probs'],
        game_ids=built['game_ids'],
        step_ids=built['step_ids'],
        actor_ids=built['actor_ids'],
        game_outcomes_obj=built['game_outcomes_obj'],
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
    ap.add_argument('--out', type=str, default=None, help='Output .npz path (placed under training_data/)')
    ap.add_argument('--use_heuristic', action='store_true', help='Use RecordingHeuristicACPlayer (generation 0)')
    ap.add_argument('--model', type=str, default=None, help='Path to AC network .pt to load')
    # expert injection removed
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


