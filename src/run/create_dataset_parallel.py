# !/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp
import gc
import queue as pyqueue
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple

import numpy as np

from core.tile import tile_flat_index, UNIQUE_TILE_COUNT

# (Removed memory monitoring utilities)


# Ensure src on path (parent of this file is the src directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Direct imports for dataset building
from core.game import MediumJong
from core.constants import NUM_PLAYERS
from core.learn.recording_ac_player import RecordingACPlayer, RecordingHeuristicACPlayer, ACPlayer
from core import constants
from tqdm import tqdm

# --- Inlined utilities from create_dataset.py ---
from typing import Optional, Sequence

# =========================
# In-code configuration
# =========================
# Edit this factory to define a fixed set of players to use for dataset creation.
# Return a list of 4 Recording* players, or return None to disable and fall back
# to CLI options (e.g., --use_heuristic).
def build_prebuilt_players() -> Optional[List[RecordingACPlayer | RecordingHeuristicACPlayer]]:
    # Example: use heuristic players (uncomment to enable)
    #return [RecordingHeuristicACPlayer(random_exploration=0.1) for _ in range(4)]

    temperature = 1
    net = ACPlayer.from_directory("models/ac_ppo_20250829_011644", temperature=temperature).network
    import torch
    device = torch.device('cpu')
    net = net.to(device)

    players = [
        RecordingACPlayer(net, temperature=0.4, zero_network_reward=False),
        RecordingACPlayer(net, temperature=0.6, zero_network_reward=False),
        RecordingACPlayer(net, temperature=0.8, zero_network_reward=False),
        RecordingACPlayer(net, temperature=1, zero_network_reward=False),
        RecordingACPlayer(net, temperature=0.8, zero_network_reward=False, exploration_consumption_factor=0.8),
        RecordingACPlayer(net, temperature=0.8, zero_network_reward=False, exploration_consumption_factor=0.6),
    ]

    # Select exactly NUM_PLAYERS unique players and assign public identifiers 5..8
    chosen = list(np.random.choice(players, NUM_PLAYERS, replace=False))
    for off, p in enumerate(chosen):
        p.identifier = 5 + off

    return chosen

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
    n_step: int = 3,
    gamma: float = 0.99,
    prebuilt_players: Sequence[RecordingACPlayer | RecordingHeuristicACPlayer] | None = None,
) -> dict:
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(int(time.time()))

    # Accumulate per-field arrays for compact storage via field specs
    array_fields = [
        ('hand_idx', np.int8),
        ('disc_idx', np.int8),
        ('called_idx', np.int8),
        ('seat_winds', np.int8),
        ('riichi_declarations', np.int8),
        ('dora_indicator_tiles', np.int8),
        ('legal_action_mask', np.int8),
        ('called_discards', np.int8),
        # New per-sample feature: remaining drawable wall counts per flat tile index
        ('wall_count', np.int8)
    ]
    scalar_fields = [
        ('round_wind', np.int8, -1),
        ('remaining_tiles', np.int8, 0),
        ('owner_of_reactable_tile', np.int8, -1),
        ('reactable_tile', np.int8, -1),
        ('newly_drawn_tile', np.int8, -1),
    ]
    master_arrays: Dict[str, List[np.ndarray]] = {k: [] for k, _ in array_fields}
    master_scalars: Dict[str, List[int]] = {k: [] for k, _, _ in scalar_fields}
    action_idx_list: List[int] = []
    tile_idx_list: List[int] = []
    all_returns: List[float] = []
    all_advantages: List[float] = []
    joint_log_probs: List[float] = []
    # deal_in_tiles: per-sample variable-length list of flat tile indices
    all_deal_in_tiles: List[List[int]] = []
    # Scalars per-sample
    # scalar lists are in master_scalars
    # Two-head policy stores separate log-probs for each head
    all_game_ids: List[int] = []
    all_step_ids: List[int] = []
    all_actor_ids: List[int] = []
    # Per-game metadata (structured only)
    game_outcomes_obj: List[dict] = []  # serialized GameOutcome per game

    # Create players once and reuse across games; clear their buffers between episodes
    if prebuilt_players is None:
        raise ValueError("build_ac_dataset requires prebuilt_players; configure build_prebuilt_players()")
    players = list(prebuilt_players)
    # Use robust tqdm settings so progress continues to render in long runs
    for gi in tqdm(range(max(1, int(games))), dynamic_ncols=True, mininterval=0.1, miniters=1, leave=True):
        # Shuffle seat order each game to avoid positional bias
        seats = list(players)
        random.shuffle(seats)
        game = MediumJong(seats, tile_copies=constants.TILE_COPIES_DEFAULT)
        game.play_round()

        # Structured outcome from the game
        outcome = game.get_game_outcome()
        game_outcomes_obj.append(outcome.serialize())
        # Compute terminal reward from points delta and assign directly to last reward slot
        pts = game.get_points()
        # Iterate in the same seat order used by the game
        for pid, p in enumerate(seats):
            if len(p.experience) == 0:
                continue
            terminal_reward = reward_function(pts[pid])
            p.finalize_episode(float(terminal_reward))  # type: ignore[attr-defined]
        for pid, p in enumerate(seats):
            T = len(p.experience)
            if T == 0:
                continue
            rewards = list(p.experience.rewards)
            # Use stored values from the experience buffer
            values: List[float] = [float(v) for v in p.experience.values]
            nstep, adv = compute_n_step_returns(rewards, n_step, gamma, values)

            # Collect per-player episode in batches via dicts
            player_arrays: Dict[str, List[np.ndarray]] = {k: [] for k, _ in array_fields}
            player_scalars: Dict[str, List[int]] = {k: [] for k, _, _ in scalar_fields}
            player_action_idx = []
            player_tile_idx = []
            player_returns = []
            player_advantages = []
            player_joint_log_probs = []
            player_deal_in_tiles: List[List[int]] = []
            player_game_ids = []
            player_step_ids = []
            player_actor_ids = []

            for t in range(T):
                gs = p.experience.states[t]
                a_idx, t_idx = p.experience.actions[t]
                # Arrays
                for fname, dt in array_fields:
                    arr_any = np.asarray(gs[fname])
                    # Assert tile index ranges for specific fields (allow -1 pad)
                    if fname in ('hand_idx', 'disc_idx', 'called_idx', 'dora_indicator_tiles'):
                        flat = arr_any.astype(np.int32).ravel()
                        if flat.size:
                            mx = int(flat.max())
                            mn = int(flat.min())
                            # allow -1 for padding; valid tiles are 0..37 inclusive
                            assert mn >= -1 and mx <= 37, f"{fname} out of range: min={mn} max={mx}"
                    player_arrays[fname].append(arr_any.astype(dt))
                # Scalars
                for fname, _dt, default in scalar_fields:
                    val = int(gs.get(fname, default))
                    if fname in ('reactable_tile', 'newly_drawn_tile'):
                        assert -1 <= val <= 37, f"{fname} out of range: {val}"
                    player_scalars[fname].append(val)
                player_action_idx.append(int(a_idx))
                player_tile_idx.append(int(t_idx))
                player_returns.append(float(nstep[t]))
                player_advantages.append(float(adv[t]))
                player_joint_log_probs.append(float(p.experience.joint_log_probs[t]))
                player_game_ids.append(int(gi))
                player_step_ids.append(int(t))
                # Use the player's public identifier rather than seat index
                player_actor_ids.append(int(p.get_identifier()))
                # Collect deal_in_tiles from the experience buffer (list[Tile]) and convert to flat indices
                tile_list = p.experience.deal_in_tiles[t] if t < len(p.experience.deal_in_tiles) else []
                din = [int(tile_flat_index(tt)) for tt in tile_list]
                # Assert variable list values are valid tile indices (no -1 in deal-in set)
                if din:
                    mx = max(din)
                    mn = min(din)
                    assert mn >= 0 and mx <= UNIQUE_TILE_COUNT, f"deal_in_tiles out of range: min={mn} max={mx}"
                player_deal_in_tiles.append(din)

            # Extend master arrays and scalars
            for k in master_arrays.keys():
                master_arrays[k].extend(player_arrays[k])
            for k in master_scalars.keys():
                master_scalars[k].extend(player_scalars[k])
            action_idx_list.extend(player_action_idx)
            tile_idx_list.extend(player_tile_idx)
            all_returns.extend(player_returns)
            all_advantages.extend(player_advantages)
            joint_log_probs.extend(player_joint_log_probs)
            all_deal_in_tiles.extend(player_deal_in_tiles)
            all_game_ids.extend(player_game_ids)
            all_step_ids.extend(player_step_ids)
            all_actor_ids.extend(player_actor_ids)

            # Clear experience buffers for next game
            p.experience.clear()

    built: Dict[str, Any] = {}
    for k, dt in array_fields:
        built[k] = np.asarray(master_arrays[k], dtype=dt)
    for k, dt, _d in scalar_fields:
        built[k] = np.asarray(master_scalars[k], dtype=dt)
    built['action_idx'] = np.asarray(action_idx_list, dtype=np.int8)
    built['tile_idx'] = np.asarray(tile_idx_list, dtype=np.int8)
    built['returns'] = np.asarray(all_returns, dtype=np.float32)
    built['advantages'] = np.asarray(all_advantages, dtype=np.float32)
    built['joint_log_probs'] = np.asarray(joint_log_probs, dtype=np.float32)
    built['game_ids'] = np.asarray(all_game_ids, dtype=np.int16)
    built['step_ids'] = np.asarray(all_step_ids, dtype=np.int16)
    built['actor_ids'] = np.asarray(all_actor_ids, dtype=np.int8)
    built['game_outcomes_obj'] = np.asarray(game_outcomes_obj, dtype=object)
    # Save deal_in_tiles as an object array (variable-length per sample)
    built['deal_in_tiles'] = np.asarray(all_deal_in_tiles, dtype=object)
    return built

def save_dataset(built, out_path):
    # Save all keys as-is
    np.savez(out_path, **built)

    print(f"Saved AC dataset to {out_path}")


@contextmanager
def _suppress_output():
    """Temporarily suppress stdout and stderr (used to silence tqdm in worker processes)."""
    try:
        devnull = open(os.devnull, 'w')
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        try:
            sys.stdout, sys.stderr = old_out, old_err
        except Exception:
            pass
        try:
            devnull.close()
        except Exception:
            pass


def _aggressive_cleanup():
    """More aggressive memory cleanup"""
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Try to clear any cached objects
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _worker_build_with_cleanup(games: int,
                               seed: int | None,
                               n_step: int,
                               gamma: float,
                               silence_io: bool) -> Dict[str, Any]:
    """Build dataset with aggressive cleanup"""

    # Construct prebuilt players inside the process (avoids pickling issues)
    prebuilt_players = build_prebuilt_players()

    if silence_io:
        with _suppress_output():
            result = build_ac_dataset(
                games=games,
                seed=seed,
                n_step=n_step,
                gamma=gamma,
                prebuilt_players=prebuilt_players,
            )
    else:
        result = build_ac_dataset(
            games=games,
            seed=seed,
            n_step=n_step,
            gamma=gamma,
            prebuilt_players=prebuilt_players,
        )

    # Aggressive cleanup after building
    _aggressive_cleanup()
    return result


def _stream_combine_results(file_paths: List[str], output_path: str) -> None:
    """Stream-process and combine results to avoid loading everything into memory"""

    # First pass: collect metadata and count total size
    total_samples = 0
    all_game_counts = []

    # Start metadata collection

    for path in file_paths:
        try:
            data = np.load(path, allow_pickle=True)
            game_count = len(data['game_outcomes_obj'])
            sample_count = len(data['action_idx'])
            all_game_counts.append(game_count)
            total_samples += sample_count
            data.close()  # Explicitly close the file
        except Exception as e:
            print(f"Error reading {path}: {e}")
            all_game_counts.append(0)

    if total_samples == 0:
        print("No samples found in any files")
        return

    # Calculate game ID offsets
    offsets = []
    acc = 0
    for cnt in all_game_counts:
        offsets.append(acc)
        acc += cnt

    # For per-sample object-like arrays, collect in lists first
    per_sample_fields = [
        'hand_idx','disc_idx','called_idx','seat_winds','riichi_declarations',
        'dora_indicator_tiles','legal_action_mask','called_discards','round_wind',
        'remaining_tiles','owner_of_reactable_tile','reactable_tile','newly_drawn_tile', 'wall_count',
        'deal_in_tiles'
    ]
    collected_lists: Dict[str, List[Any]] = {k: [] for k in per_sample_fields}
    # Per-game outcomes are collected separately
    outcomes_list: List[Any] = []

    # For numeric arrays, pre-allocate
    action_idx = np.empty(total_samples, dtype=np.int8)
    tile_idx = np.empty(total_samples, dtype=np.int8)
    returns = np.empty(total_samples, dtype=np.float32)
    advantages = np.empty(total_samples, dtype=np.float32)
    joint_log_probs = np.empty(total_samples, dtype=np.float32)
    game_ids = np.empty(total_samples, dtype=np.int16)
    step_ids = np.empty(total_samples, dtype=np.int16)
    actor_ids = np.empty(total_samples, dtype=np.int8)

    # Second pass: load and combine data
    sample_offset = 0

    for i, path in enumerate(file_paths):

        try:
            data = np.load(path, allow_pickle=True)

            # Get the size of this chunk
            chunk_size = len(data['action_idx'])
            if chunk_size == 0:
                data.close()
                continue

            # Copy numeric data directly into pre-allocated arrays
            action_idx[sample_offset:sample_offset + chunk_size] = data['action_idx']
            tile_idx[sample_offset:sample_offset + chunk_size] = data['tile_idx']
            returns[sample_offset:sample_offset + chunk_size] = data['returns']
            advantages[sample_offset:sample_offset + chunk_size] = data['advantages']
            joint_log_probs[sample_offset:sample_offset + chunk_size] = data['joint_log_probs']

            # Apply game ID offset and copy
            gid = data['game_ids'] + offsets[i]
            game_ids[sample_offset:sample_offset + chunk_size] = gid
            step_ids[sample_offset:sample_offset + chunk_size] = data['step_ids']
            actor_ids[sample_offset:sample_offset + chunk_size] = data['actor_ids']

            # Collect per-sample arrays
            for k in per_sample_fields:
                collected_lists[k].extend(list(data[k]))
            # Collect per-game outcomes
            outcomes_list.extend(list(data['game_outcomes_obj']))

            sample_offset += chunk_size
            data.close()

            # Aggressive cleanup every few files
            if i % 5 == 0:
                _aggressive_cleanup()

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue


    # Convert object lists to arrays. The lists will be of the same dimension,
    # avoiding the need to store them as objects
    combined = {
        'action_idx': action_idx,
        'tile_idx': tile_idx,
        'returns': returns,
        'advantages': advantages,
        'joint_log_probs': joint_log_probs,
        'game_ids': game_ids,
        'step_ids': step_ids,
        'actor_ids': actor_ids,
    }
    # Cast per-sample arrays to appropriate dtypes
    dtypes_map = {
        'wall_count': np.int8,
        # deal_in_tiles is variable-length; keep as object in the combined output
        'deal_in_tiles': object,
        'hand_idx': np.int8,
        'disc_idx': np.int8,
        'called_idx': np.int8,
        'seat_winds': np.int8,
        'riichi_declarations': np.int8,
        'dora_indicator_tiles': np.int8,
        'legal_action_mask': np.int8,
        'called_discards': np.int8,
        'round_wind': np.int8,
        'remaining_tiles': np.int8,
        'owner_of_reactable_tile': np.int8,
        'reactable_tile': np.int8,
        'newly_drawn_tile': np.int8,
    }
    for k in per_sample_fields:
        if k == 'deal_in_tiles':
            combined[k] = np.array(collected_lists[k], dtype=object)
        else:
            combined[k] = np.array(collected_lists[k], dtype=dtypes_map[k])
    # Add per-game outcomes (concatenate across files)
    combined['game_outcomes_obj'] = np.array(outcomes_list, dtype=object)

    # Clear the lists
    del collected_lists, outcomes_list
    _aggressive_cleanup()

    save_dataset(combined, output_path)
    # Save complete


def run_chunk_process(rank: int,
                      games_for_chunk: int,
                      seed: int | None,
                      n_step: int,
                      gamma: float,
                      reporter_rank: int,
                      queue: mp.Queue,
                      run_id: str,
                      chunk_index: int) -> None:
    """Run a single chunk in its own process and exit.

    This isolates Python memory arenas so RSS drops once the process terminates.
    """

    # Silence non-reporter
    silence = (rank != reporter_rank)

    # Prepare per-chunk temp directory
    base_dir = os.path.abspath(os.getcwd())
    tmp_dir = os.path.join(base_dir, 'training_data', 'tmp_parallel', f'run_{run_id}', f'chunkproc_{rank}')
    os.makedirs(tmp_dir, exist_ok=True)

    # Build dataset for this chunk (no player reuse across chunks, process will exit)
    built = _worker_build_with_cleanup(
        games=int(max(1, games_for_chunk)),
        seed=seed,
        n_step=int(n_step),
        gamma=float(gamma),
        silence_io=silence,
    )

    out_path = os.path.join(tmp_dir, f'chunk_{chunk_index:06d}.npz')
    save_dataset(built, out_path)

    # Drop references aggressively just before exit
    try:
        del built
    except Exception:
        pass
    _aggressive_cleanup()

    queue.put((rank, [out_path]))


def create_dataset_parallel(*,
                            games: int,
                            num_processes: int = 1,
                            seed: int | None = None,
                            n_step: int = 3,
                            gamma: float = 0.99,
                            out: str | None = None,
                            chunk_size: int = 250,
                            keep_partials: bool = False,
                            stream_combine: bool = True) -> str:
    """Programmatic API to build a dataset in parallel and save it to disk.

    Returns the output .npz path.
    """
    # Ensure a safe start method for Windows and memory isolation
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Already set by parent; ignore
        pass
    class _Args:
        pass
    args = _Args()
    args.games = games
    args.num_processes = num_processes
    args.seed = seed
    args.n_step = n_step
    args.gamma = gamma
    args.out = out
    args.chunk_size = chunk_size
    args.keep_partials = keep_partials
    args.stream_combine = stream_combine

    # Starting main process

    total_games = max(1, int(args.games))
    num_procs = max(1, int(args.num_processes))

    # Partition games across processes
    base = total_games // num_procs
    rem = total_games % num_procs
    games_splits = [base + (1 if i < rem else 0) for i in range(num_procs)]

    # We'll dynamically ensure there is always one alive reporter emitting tqdm
    reporter_rank = next((i for i, g in enumerate(games_splits) if g > 0), 0)

    # Prepare run id and temp base directory
    run_id = time.strftime('%Y%m%d_%H%M%S')
    tmp_base = os.path.join(os.path.abspath(os.getcwd()), 'training_data', 'tmp_parallel', f'run_{run_id}')
    os.makedirs(tmp_base, exist_ok=True)

    # New: per-chunk process scheduling so memory is freed after each chunk
    queue: mp.Queue = mp.Queue()
    collected: Dict[int, List[str]] = {}

    # Build a list of chunk tasks (rank_id, games_in_chunk, seed, chunk_index)
    chunk_tasks: List[Tuple[int, int, int | None, int]] = []
    chunk_index = 0
    for rank, games_for_rank in enumerate(games_splits):
        remaining = int(games_for_rank)
        while remaining > 0:
            this_chunk = int(min(args.chunk_size, remaining))
            # Derive seed per chunk if provided
            seed_for_chunk = None if args.seed is None else int(args.seed) + int(rank) + int(chunk_index)
            chunk_tasks.append((rank, this_chunk, seed_for_chunk, chunk_index))
            remaining -= this_chunk
            chunk_index += 1

    # Launch up to num_processes concurrent chunk processes
    # Track processes along with their ranks so we can manage the reporter
    procs: List[Tuple[mp.Process, int]] = []
    current_reporter: int | None = None
    next_task_idx = 0
    total_tasks = len(chunk_tasks)

    while next_task_idx < total_tasks or procs:
        # Start new processes while we have capacity
        while next_task_idx < total_tasks and len(procs) < num_procs:
            rank_id, games_in_chunk, seed_for_chunk, cidx = chunk_tasks[next_task_idx]
            # Ensure exactly one reporter at a time; if none alive, make this one the reporter
            reporter = current_reporter if current_reporter is not None else rank_id
            p = mp.Process(
                target=run_chunk_process,
                args=(
                    rank_id, int(games_in_chunk), seed_for_chunk,
                    int(args.n_step), float(args.gamma),
                    reporter, queue, run_id, int(cidx)
                ),
            )
            p.daemon = False
            p.start()
            procs.append((p, rank_id))
            # If we just assigned this as reporter, remember it
            if current_reporter is None:
                current_reporter = rank_id
            next_task_idx += 1

        # Reap finished processes first to avoid waiting unnecessarily
        alive_procs: List[Tuple[mp.Process, int]] = []
        reporter_alive = False
        for p, rnk in procs:
            if p.is_alive():
                alive_procs.append((p, rnk))
                if current_reporter is not None and rnk == current_reporter:
                    reporter_alive = True
            else:
                try:
                    p.join(timeout=0.1)
                except Exception:
                    pass
        procs = alive_procs
        # If reporter has exited, clear so the next starter becomes reporter
        if not reporter_alive:
            current_reporter = None

        # Drain any available results without blocking
        while True:
            try:
                r_rank, r_paths = queue.get_nowait()
            except pyqueue.Empty:
                break
            except Exception:
                break
            else:
                if int(r_rank) not in collected:
                    collected[int(r_rank)] = []
                collected[int(r_rank)].extend(list(r_paths))

    # Prepare output path
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = os.path.join(out_dir, args.out)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'ac_parallel_{ts}.npz')

    # Ensure the queue is properly closed to avoid background thread hangs on Windows
    try:
        queue.close()
        queue.join_thread()
    except Exception:
        pass

    # Combine results using streaming approach (single supported path)
    # Collect all file paths in order
    all_file_paths = []
    for r in sorted(collected.keys()):
        all_file_paths.extend(collected[r])

    _stream_combine_results(all_file_paths, out_path)

    # Cleanup partials unless requested to keep
    if not args.keep_partials:
        try:
            for r in collected.keys():
                for pth in collected[r]:
                    if os.path.isfile(pth):
                        os.remove(pth)
            # Remove empty directories
            for root, dirs, files in os.walk(tmp_base, topdown=False):
                if not files and not dirs:
                    try:
                        os.rmdir(root)
                    except Exception:
                        pass
        except Exception:
            pass

    # Final cleanup complete
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Parallel AC dataset creation (players configured in-code via build_prebuilt_players)')
    ap.add_argument('--games', type=int, default=10, help='Total number of games to simulate')
    ap.add_argument('--num_processes', type=int, default=1, help='Number of parallel worker processes')
    ap.add_argument('--seed', type=int, default=None, help='Base random seed')
    ap.add_argument('--n_step', type=int, default=3, help='N for n-step returns')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    ap.add_argument('--out', type=str, default=None, help='Output .npz path')
    ap.add_argument('--chunk_size', type=int, default=250, help='Smaller chunks to reduce memory')
    ap.add_argument('--keep_partials', action='store_true')
    ap.add_argument('--stream_combine', action='store_true',
                    help='Use streaming combination to reduce memory usage (kept for compatibility, streaming is always used)')
    args = ap.parse_args()

    return create_dataset_parallel(
        games=args.games,
        num_processes=args.num_processes,
        seed=args.seed,
        n_step=args.n_step,
        gamma=args.gamma,
        out=args.out,
        chunk_size=int(max(1, args.chunk_size)),
        keep_partials=bool(args.keep_partials),
        stream_combine=bool(args.stream_combine),
    )


# Removed legacy worker and combination helpers; streaming combine is the single code path


if __name__ == '__main__':
    # Use spawn for better Windows compatibility and memory isolation
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()