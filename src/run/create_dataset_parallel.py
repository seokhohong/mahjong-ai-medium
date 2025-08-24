# !/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
import importlib.util
import multiprocessing as mp
import gc
import psutil
import queue as pyqueue
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple

import numpy as np


# Memory monitoring utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(stage: str, verbose: bool = True):
    """Log current memory usage"""
    if verbose:
        mem_mb = get_memory_usage()
        print(f"[MEMORY] {stage}: {mem_mb:.1f} MB")


# Ensure src on path (parent of this file is the src directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Dynamically load sibling create_dataset.py to reuse its functions without requiring run/ as a package
_CD_PATH = os.path.join(os.path.dirname(__file__), 'create_dataset.py')
_spec = importlib.util.spec_from_file_location("_create_dataset_module", _CD_PATH)
assert _spec and _spec.loader
_cd_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd_mod)  # type: ignore[arg-type]
build_ac_dataset = getattr(_cd_mod, 'build_ac_dataset')
save_dataset = getattr(_cd_mod, 'save_dataset')

# No direct player imports needed here; players are managed inside build_ac_dataset


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
                               temperature: float,
                               zero_network_reward: bool,
                               n_step: int,
                               gamma: float,
                               use_heuristic: bool,
                               model_path: str | None,
                               silence_io: bool,
                               prebuilt_players: list | None = None) -> Dict[str, Any]:
    """Build dataset with aggressive cleanup"""

    if silence_io:
        with _suppress_output():
            result = build_ac_dataset(
                games=games,
                seed=seed,
                temperature=temperature,
                zero_network_reward=zero_network_reward,
                n_step=n_step,
                gamma=gamma,
                use_heuristic=use_heuristic,
                model_path=model_path,
                prebuilt_players=prebuilt_players,
            )
    else:
        result = build_ac_dataset(
            games=games,
            seed=seed,
            temperature=temperature,
            zero_network_reward=zero_network_reward,
            n_step=n_step,
            gamma=gamma,
            use_heuristic=use_heuristic,
            model_path=model_path,
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

    log_memory("Starting metadata collection")

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

    log_memory("Metadata collection complete")

    if total_samples == 0:
        print("No samples found in any files")
        return

    # Calculate game ID offsets
    offsets = []
    acc = 0
    for cnt in all_game_counts:
        offsets.append(acc)
        acc += cnt

    # Pre-allocate output arrays
    log_memory("Pre-allocating output arrays")

    # For object arrays, we'll collect in lists first
    hand_idx_list = []
    disc_idx_list = []
    called_idx_list = []
    game_state_list = []
    called_discards_list = []
    outcomes_list = []

    # For numeric arrays, pre-allocate
    action_idx = np.empty(total_samples, dtype=np.int64)
    tile_idx = np.empty(total_samples, dtype=np.int64)
    returns = np.empty(total_samples, dtype=np.float32)
    advantages = np.empty(total_samples, dtype=np.float32)
    joint_log_probs = np.empty(total_samples, dtype=np.float32)
    game_ids = np.empty(total_samples, dtype=np.int32)
    step_ids = np.empty(total_samples, dtype=np.int32)
    actor_ids = np.empty(total_samples, dtype=np.int32)

    # Second pass: load and combine data
    sample_offset = 0

    for i, path in enumerate(file_paths):
        log_memory(f"Processing file {i + 1}/{len(file_paths)}")

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

            # Collect object arrays
            hand_idx_list.extend(data['hand_idx'])
            disc_idx_list.extend(data['disc_idx'])
            called_idx_list.extend(data['called_idx'])
            game_state_list.extend(data['game_state'])
            called_discards_list.extend(data['called_discards'])
            outcomes_list.extend(data['game_outcomes_obj'])

            sample_offset += chunk_size
            data.close()

            # Aggressive cleanup every few files
            if i % 5 == 0:
                _aggressive_cleanup()

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    log_memory("Converting object lists to arrays")

    # Convert object lists to arrays
    combined = {
        'hand_idx': np.array(hand_idx_list, dtype=object),
        'disc_idx': np.array(disc_idx_list, dtype=object),
        'called_idx': np.array(called_idx_list, dtype=object),
        'game_state': np.array(game_state_list, dtype=object),
        'called_discards': np.array(called_discards_list, dtype=object),
        'action_idx': action_idx,
        'tile_idx': tile_idx,
        'returns': returns,
        'advantages': advantages,
        'joint_log_probs': joint_log_probs,
        'game_ids': game_ids,
        'step_ids': step_ids,
        'actor_ids': actor_ids,
        'game_outcomes_obj': np.array(outcomes_list, dtype=object),
    }

    # Clear the lists
    del hand_idx_list, disc_idx_list, called_idx_list
    del game_state_list, called_discards_list, outcomes_list
    _aggressive_cleanup()

    log_memory("Saving combined dataset")
    save_dataset(combined, output_path)
    log_memory("Save complete")


def run_chunk_process(rank: int,
                      games_for_chunk: int,
                      seed: int | None,
                      temperature: float,
                      zero_network_reward: bool,
                      n_step: int,
                      gamma: float,
                      use_heuristic: bool,
                      model_path: str | None,
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
        temperature=temperature,
        zero_network_reward=bool(zero_network_reward),
        n_step=int(n_step),
        gamma=float(gamma),
        use_heuristic=bool(use_heuristic),
        model_path=model_path,
        silence_io=silence,
        prebuilt_players=None,
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
                            temperature: float = 0.1,
                            zero_network_reward: bool = False,
                            n_step: int = 3,
                            gamma: float = 0.99,
                            out: str | None = None,
                            use_heuristic: bool = False,
                            model: str | None = None,
                            chunk_size: int = 250,
                            keep_partials: bool = False,
                            stream_combine: bool = True,
                            verbose_memory: bool = False) -> str:
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
    args.temperature = temperature
    args.zero_network_reward = zero_network_reward
    args.n_step = n_step
    args.gamma = gamma
    args.out = out
    args.use_heuristic = use_heuristic
    args.model = model
    args.chunk_size = chunk_size
    args.keep_partials = keep_partials
    args.stream_combine = stream_combine
    args.verbose_memory = verbose_memory
    

    log_memory("Starting main process", args.verbose_memory)

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

    log_memory(f"Scheduling {total_tasks} chunk processes (concurrency={num_procs})", args.verbose_memory)

    while next_task_idx < total_tasks or procs:
        # Start new processes while we have capacity
        while next_task_idx < total_tasks and len(procs) < num_procs:
            rank_id, games_in_chunk, seed_for_chunk, cidx = chunk_tasks[next_task_idx]
            # Ensure exactly one reporter at a time; if none alive, make this one the reporter
            reporter = current_reporter if current_reporter is not None else rank_id

            p = mp.Process(
                target=run_chunk_process,
                args=(
                    rank_id, int(games_in_chunk), seed_for_chunk, args.temperature, bool(args.zero_network_reward),
                    int(args.n_step), float(args.gamma), bool(args.use_heuristic), args.model,
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

    log_memory("All processes joined, starting combination", args.verbose_memory)

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

    log_memory("Dataset saved", args.verbose_memory)

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

    log_memory("Final cleanup complete", args.verbose_memory)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Memory-optimized parallel AC dataset creation')
    ap.add_argument('--games', type=int, default=10, help='Total number of games to simulate')
    ap.add_argument('--num_processes', type=int, default=1, help='Number of parallel worker processes')
    ap.add_argument('--seed', type=int, default=None, help='Base random seed')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--zero_network_reward', action='store_true')
    ap.add_argument('--n_step', type=int, default=3, help='N for n-step returns')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    ap.add_argument('--out', type=str, default=None, help='Output .npz path')
    ap.add_argument('--use_heuristic', action='store_true')
    ap.add_argument('--model', type=str, default=None, help='Path to AC network')
    ap.add_argument('--chunk_size', type=int, default=250, help='Smaller chunks to reduce memory')
    ap.add_argument('--keep_partials', action='store_true')
    ap.add_argument('--stream_combine', action='store_true',
                    help='Use streaming combination to reduce memory usage (kept for compatibility, streaming is always used)')
    ap.add_argument('--verbose_memory', action='store_true',
                    help='Print detailed memory usage information')
    args = ap.parse_args()

    return create_dataset_parallel(
        games=args.games,
        num_processes=args.num_processes,
        seed=args.seed,
        temperature=args.temperature,
        zero_network_reward=bool(args.zero_network_reward),
        n_step=args.n_step,
        gamma=args.gamma,
        out=args.out,
        use_heuristic=bool(args.use_heuristic),
        model=args.model,
        chunk_size=int(max(1, args.chunk_size)),
        keep_partials=bool(args.keep_partials),
        stream_combine=bool(args.stream_combine),
        verbose_memory=bool(args.verbose_memory),
    )


# Removed legacy worker and combination helpers; streaming combine is the single code path


if __name__ == '__main__':
    # Use spawn for better Windows compatibility and memory isolation
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()