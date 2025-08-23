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

# Import player classes to construct reusable players per worker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.learn.recording_ac_player import RecordingACPlayer, RecordingHeuristicACPlayer
from core.learn.ac_player import ACPlayer  # type: ignore


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


def _efficient_array_concat(parts: List[np.ndarray], dtype: Any) -> np.ndarray:
    """More memory-efficient array concatenation"""
    if not parts:
        return np.array([], dtype=dtype)

    # Calculate total size first
    total_size = sum(len(part) for part in parts)
    if total_size == 0:
        return np.array([], dtype=dtype)

    # Pre-allocate result array
    if dtype == object:
        result = np.empty(total_size, dtype=object)
    else:
        result = np.empty(total_size, dtype=dtype)

    # Fill the result array and delete parts as we go
    offset = 0
    for i, part in enumerate(parts):
        if len(part) > 0:
            result[offset:offset + len(part)] = part
            offset += len(part)

        # Clear the part immediately to free memory
        parts[i] = None
        if i % 5 == 0:  # Periodic cleanup
            gc.collect()

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
    action_log_probs = np.empty(total_samples, dtype=np.float32)
    tile_log_probs = np.empty(total_samples, dtype=np.float32)
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
            action_log_probs[sample_offset:sample_offset + chunk_size] = data['action_log_probs']
            tile_log_probs[sample_offset:sample_offset + chunk_size] = data['tile_log_probs']

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
        'action_log_probs': action_log_probs,
        'tile_log_probs': tile_log_probs,
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


def run_worker_optimized(rank: int,
                         games_for_rank: int,
                         seed_base: int | None,
                         temperature: float,
                         zero_network_reward: bool,
                         n_step: int,
                         gamma: float,
                         use_heuristic: bool,
                         model_path: str | None,
                         reporter_rank: int,
                         queue: mp.Queue,
                         run_id: str,
                         chunk_size: int = 500,
                         reuse_players: bool = False,
                         batch_idx: int = 0) -> None:
    """Memory-optimized worker that aggressively cleans up after each chunk"""

    # Derive per-process seed base
    seed0 = None if seed_base is None else int(seed_base)
    silence = (rank != reporter_rank)

    # Prepare per-worker temp directory
    base_dir = os.path.abspath(os.getcwd())
    tmp_dir = os.path.join(base_dir, 'training_data', 'tmp_parallel', f'run_{run_id}', f'rank_{rank}', f'batch_{batch_idx}')
    os.makedirs(tmp_dir, exist_ok=True)

    players = None
    if reuse_players:
        # Build reusable players once per worker to avoid repeated model loads
        if use_heuristic:
            players = [RecordingHeuristicACPlayer(random_exploration=temperature) for _ in range(4)]
        else:
            if not model_path:
                raise ValueError("model_path must be provided when not using heuristic")
            net = ACPlayer.from_directory(model_path, temperature=temperature).network
            try:
                import torch  # type: ignore
                # Ensure model is on CPU and not holding GPU memory
                net = net.to(torch.device('cpu'))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            players = [RecordingACPlayer(net, temperature=temperature, zero_network_reward=bool(zero_network_reward)) for _ in range(4)]

    remaining = int(max(0, games_for_rank))
    chunk_idx = 0
    file_paths: List[str] = []

    while remaining > 0:
        this_chunk = int(min(chunk_size, remaining))
        # Each chunk uses an incremented seed (when provided) for determinism across chunks
        seed = None if seed0 is None else seed0 + int(rank) + chunk_idx

        built = _worker_build_with_cleanup(
            games=this_chunk,
            seed=seed,
            temperature=temperature,
            zero_network_reward=bool(zero_network_reward),
            n_step=int(n_step),
            gamma=float(gamma),
            use_heuristic=bool(use_heuristic),
            model_path=model_path,
            silence_io=silence,
            prebuilt_players=players if reuse_players else None,
        )

        out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npz')
        save_dataset(built, out_path)

        # Explicitly delete and clean up
        del built
        _aggressive_cleanup()

        file_paths.append(out_path)
        remaining -= this_chunk
        chunk_idx += 1

    # Clean up players at the end
    if players:
        del players
        _aggressive_cleanup()

    queue.put((rank, file_paths))


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
    # expert injection removed
    ap.add_argument('--stream_combine', action='store_true',
                    help='Use streaming combination to reduce memory usage')
    ap.add_argument('--verbose_memory', action='store_true',
                    help='Print detailed memory usage information')
    ap.add_argument('--restart_every_chunks', type=int, default=0,
                    help='Nuclear mode: restart worker processes after this many chunks per worker (0 disables)')
    args = ap.parse_args()

    log_memory("Starting main process", args.verbose_memory)

    total_games = max(1, int(args.games))
    num_procs = max(1, int(args.num_processes))

    # Partition games across processes
    base = total_games // num_procs
    rem = total_games % num_procs
    games_splits = [base + (1 if i < rem else 0) for i in range(num_procs)]

    reporter_rank = next((i for i, g in enumerate(games_splits) if g > 0), 0)

    # Prepare run id and temp base directory
    run_id = time.strftime('%Y%m%d_%H%M%S')
    tmp_base = os.path.join(os.path.abspath(os.getcwd()), 'training_data', 'tmp_parallel', f'run_{run_id}')
    os.makedirs(tmp_base, exist_ok=True)

    # Batch-based launching (nuclear restarts)
    queue: mp.Queue = mp.Queue()
    collected: Dict[int, List[str]] = {}

    # Remaining games per rank
    remaining_per_rank: List[int] = list(games_splits)
    batch_idx = 0

    restart_every_chunks = max(0, int(getattr(args, 'restart_every_chunks', 0)))
    per_proc_batch_games = None
    if restart_every_chunks > 0:
        per_proc_batch_games = int(max(1, args.chunk_size)) * restart_every_chunks

    while any(g > 0 for g in remaining_per_rank):
        procs: List[mp.Process] = []

        # Determine reporter for this batch
        try:
            reporter_rank = next(i for i, g in enumerate(remaining_per_rank) if g > 0)
        except StopIteration:
            break

        log_memory(f"Launching worker processes for batch {batch_idx}", args.verbose_memory)

        launched = 0
        for i, remaining_games in enumerate(remaining_per_rank):
            if remaining_games <= 0:
                continue
            if per_proc_batch_games is None:
                games_this_spawn = remaining_games
            else:
                games_this_spawn = min(remaining_games, per_proc_batch_games)

            p = mp.Process(
                target=run_worker_optimized,
                args=(
                    i, int(games_this_spawn), args.seed, args.temperature, bool(args.zero_network_reward),
                    int(args.n_step), float(args.gamma), bool(args.use_heuristic),
                    args.model, reporter_rank, queue, run_id,
                    int(max(1, args.chunk_size)), True,  # reuse_players=True
                    int(batch_idx)
                ),
            )
            p.daemon = False
            p.start()
            procs.append(p)
            launched += 1

        # Collect for this batch
        remaining_to_collect = launched
        while remaining_to_collect > 0:
            r_rank, r_paths = queue.get()
            if int(r_rank) not in collected:
                collected[int(r_rank)] = []
            collected[int(r_rank)].extend(list(r_paths))
            remaining_to_collect -= 1

        log_memory("Workers completed, joining processes", args.verbose_memory)

        for p in procs:
            p.join()

        # Update remaining and advance batch
        for i, remaining_games in enumerate(remaining_per_rank):
            if remaining_games <= 0:
                continue
            if per_proc_batch_games is None:
                consumed = remaining_games
            else:
                consumed = min(remaining_games, per_proc_batch_games)
            remaining_per_rank[i] = max(0, remaining_games - consumed)

        batch_idx += 1

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

    # Combine results using streaming approach if requested
    if args.stream_combine:
        # Collect all file paths in order
        all_file_paths = []
        for r in sorted(collected.keys()):
            all_file_paths.extend(collected[r])

        _stream_combine_results(all_file_paths, out_path)
    else:
        # Original approach but with more aggressive cleanup
        all_dicts: List[Dict[str, Any]] = []
        for r in sorted(collected.keys()):
            for pth in collected[r]:
                all_dicts.append(_load_npz_as_dict(pth))
                if len(all_dicts) % 10 == 0:  # Periodic cleanup
                    _aggressive_cleanup()

        log_memory("All chunks loaded, combining", args.verbose_memory)
        combined = _combine_results_optimized(all_dicts)
        save_dataset(combined, out_path)

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


def _combine_results_optimized(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Memory-optimized version of _combine_results"""

    # Compute offsets
    game_counts = [int(len(r.get('game_outcomes_obj', []))) for r in results]
    offsets = []
    acc = 0
    for cnt in game_counts:
        offsets.append(acc)
        acc += cnt

    # Process in chunks to avoid memory spikes
    chunk_size = 50  # Process 50 files at a time

    final_parts = {
        'hand_idx': [], 'disc_idx': [], 'called_idx': [], 'game_state': [],
        'called_discards': [],
        'action_idx': [], 'tile_idx': [],
        'returns': [], 'advantages': [],
        'action_log_probs': [], 'tile_log_probs': [],
        'game_ids': [], 'step_ids': [], 'actor_ids': [],
        'game_outcomes_obj': []
    }

    for chunk_start in range(0, len(results), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(results))
        chunk_results = results[chunk_start:chunk_end]

        # Process this chunk
        chunk_parts = {key: [] for key in final_parts.keys()}

        for i, r in enumerate(chunk_results):
            actual_i = chunk_start + i

            chunk_parts['hand_idx'].append(np.asarray(r['hand_idx'], dtype=object))
            chunk_parts['disc_idx'].append(np.asarray(r['disc_idx'], dtype=object))
            chunk_parts['called_idx'].append(np.asarray(r['called_idx'], dtype=object))
            chunk_parts['game_state'].append(np.asarray(r['game_state'], dtype=object))
            chunk_parts['called_discards'].append(np.asarray(r['called_discards'], dtype=object))
            chunk_parts['action_idx'].append(np.asarray(r['action_idx'], dtype=np.int64))
            chunk_parts['tile_idx'].append(np.asarray(r['tile_idx'], dtype=np.int64))
            chunk_parts['returns'].append(np.asarray(r['returns'], dtype=np.float32))
            chunk_parts['advantages'].append(np.asarray(r['advantages'], dtype=np.float32))
            chunk_parts['action_log_probs'].append(np.asarray(r['action_log_probs'], dtype=np.float32))
            chunk_parts['tile_log_probs'].append(np.asarray(r['tile_log_probs'], dtype=np.float32))

            # Apply offset to game_ids
            gid = np.asarray(r['game_ids'], dtype=np.int32) + np.int32(offsets[actual_i])
            chunk_parts['game_ids'].append(gid)

            chunk_parts['step_ids'].append(np.asarray(r['step_ids'], dtype=np.int32))
            chunk_parts['actor_ids'].append(np.asarray(r['actor_ids'], dtype=np.int32))
            chunk_parts['game_outcomes_obj'].append(np.asarray(r['game_outcomes_obj'], dtype=object))

        # Combine this chunk and add to final parts
        for key in final_parts.keys():
            if key in ['action_idx', 'tile_idx', 'returns', 'advantages', 'action_log_probs', 'tile_log_probs', 'game_ids', 'step_ids', 'actor_ids']:
                combined_chunk = _efficient_array_concat(
                    chunk_parts[key],
                    np.int64 if ('idx' in key or 'ids' in key) else np.float32
                )
            else:
                combined_chunk = np.concat(chunk_parts[key])

            final_parts[key].append(combined_chunk)

        # Clear chunk data
        del chunk_results, chunk_parts
        _aggressive_cleanup()

    # Final combination
    result = {}
    for key in final_parts.keys():
        result[key] = np.concat(final_parts[key])

    return result


def _load_npz_as_dict(path: str) -> Dict[str, Any]:
    """Load NPZ file and immediately return data dict to avoid keeping file handle open"""
    data = np.load(path, allow_pickle=True)
    result = {
        'hand_idx': data['hand_idx'],
        'disc_idx': data['disc_idx'],
        'called_idx': data['called_idx'],
        'game_state': data['game_state'],
        'called_discards': data['called_discards'],
        'action_idx': data['action_idx'],
        'tile_idx': data['tile_idx'],
        'returns': data['returns'],
        'advantages': data['advantages'],
        'action_log_probs': data['action_log_probs'],
        'tile_log_probs': data['tile_log_probs'],
        'game_ids': data['game_ids'],
        'step_ids': data['step_ids'],
        'actor_ids': data['actor_ids'],
        'game_outcomes_obj': data['game_outcomes_obj'],
    }
    data.close()  # Explicitly close the file
    return result


if __name__ == '__main__':
    # Use spawn for better Windows compatibility and memory isolation
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()