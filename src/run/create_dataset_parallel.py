#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
import importlib.util
import multiprocessing as mp
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple

import numpy as np

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


def _worker_build(games: int,
                  seed: int | None,
                  temperature: float,
                  zero_network_reward: bool,
                  n_step: int,
                  gamma: float,
                  use_heuristic: bool,
                  model_path: str | None,
                  silence_io: bool) -> Dict[str, Any]:
    if silence_io:
        with _suppress_output():
            return build_ac_dataset(
                games=games,
                seed=seed,
                temperature=temperature,
                zero_network_reward=zero_network_reward,
                n_step=n_step,
                gamma=gamma,
                use_heuristic=use_heuristic,
                model_path=model_path,
            )
    else:
        return build_ac_dataset(
            games=games,
            seed=seed,
            temperature=temperature,
            zero_network_reward=zero_network_reward,
            n_step=n_step,
            gamma=gamma,
            use_heuristic=use_heuristic,
            model_path=model_path,
        )


def _concat_object_arrays(parts: List[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=object)
    return np.concatenate(parts, axis=0).astype(object)


def _concat_numeric_arrays(parts: List[np.ndarray], dtype: Any) -> np.ndarray:
    if not parts:
        return np.array([], dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype)


def _combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Compute per-chunk game counts and create game_id offsets so game_ids remain unique globally
    game_counts = [int(len(r.get('game_outcomes_obj', []))) for r in results]
    offsets = []
    acc = 0
    for cnt in game_counts:
        offsets.append(acc)
        acc += cnt

    # Prepare concatenation lists
    hand_idx_parts: List[np.ndarray] = []
    disc_idx_parts: List[np.ndarray] = []
    called_idx_parts: List[np.ndarray] = []
    game_state_parts: List[np.ndarray] = []
    called_discards_parts: List[np.ndarray] = []
    flat_idx_parts: List[np.ndarray] = []
    returns_parts: List[np.ndarray] = []
    advantages_parts: List[np.ndarray] = []
    old_log_probs_parts: List[np.ndarray] = []
    game_ids_parts: List[np.ndarray] = []
    step_ids_parts: List[np.ndarray] = []
    actor_ids_parts: List[np.ndarray] = []
    flat_policies_parts: List[np.ndarray] = []
    outcomes_parts: List[np.ndarray] = []

    for i, r in enumerate(results):
        hand_idx_parts.append(np.asarray(r['hand_idx'], dtype=object))
        disc_idx_parts.append(np.asarray(r['disc_idx'], dtype=object))
        called_idx_parts.append(np.asarray(r['called_idx'], dtype=object))
        game_state_parts.append(np.asarray(r['game_state'], dtype=object))
        called_discards_parts.append(np.asarray(r['called_discards'], dtype=object))
        flat_idx_parts.append(np.asarray(r['flat_idx'], dtype=np.int64))
        returns_parts.append(np.asarray(r['returns'], dtype=np.float32))
        advantages_parts.append(np.asarray(r['advantages'], dtype=np.float32))
        old_log_probs_parts.append(np.asarray(r['old_log_probs'], dtype=np.float32))
        # Apply offset to game_ids to keep global uniqueness
        gid = np.asarray(r['game_ids'], dtype=np.int32)
        gid = gid + np.int32(offsets[i])
        game_ids_parts.append(gid)
        step_ids_parts.append(np.asarray(r['step_ids'], dtype=np.int32))
        actor_ids_parts.append(np.asarray(r['actor_ids'], dtype=np.int32))
        flat_policies_parts.append(np.asarray(r['flat_policies'], dtype=object))
        outcomes_parts.append(np.asarray(r['game_outcomes_obj'], dtype=object))

    return {
        'hand_idx': _concat_object_arrays(hand_idx_parts),
        'disc_idx': _concat_object_arrays(disc_idx_parts),
        'called_idx': _concat_object_arrays(called_idx_parts),
        'game_state': _concat_object_arrays(game_state_parts),
        'called_discards': _concat_object_arrays(called_discards_parts),
        'flat_idx': _concat_numeric_arrays(flat_idx_parts, np.int64),
        'returns': _concat_numeric_arrays(returns_parts, np.float32),
        'advantages': _concat_numeric_arrays(advantages_parts, np.float32),
        'old_log_probs': _concat_numeric_arrays(old_log_probs_parts, np.float32),
        'game_ids': _concat_numeric_arrays(game_ids_parts, np.int32),
        'step_ids': _concat_numeric_arrays(step_ids_parts, np.int32),
        'actor_ids': _concat_numeric_arrays(actor_ids_parts, np.int32),
        'flat_policies': _concat_object_arrays(flat_policies_parts),
        'game_outcomes_obj': _concat_object_arrays(outcomes_parts),
    }


def _load_npz_as_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        'hand_idx': data['hand_idx'],
        'disc_idx': data['disc_idx'],
        'called_idx': data['called_idx'],
        'game_state': data['game_state'],
        'called_discards': data['called_discards'],
        'flat_idx': data['flat_idx'],
        'returns': data['returns'],
        'advantages': data['advantages'],
        'old_log_probs': data['old_log_probs'],
        'game_ids': data['game_ids'],
        'step_ids': data['step_ids'],
        'actor_ids': data['actor_ids'],
        'flat_policies': data['flat_policies'],
        'game_outcomes_obj': data['game_outcomes_obj'],
    }


def run_worker(rank: int,
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
               chunk_size: int = 500) -> None:
    """Worker executes in chunks, flushes each chunk to disk, and reports file paths via queue."""
    # Derive per-process seed base
    seed0 = None if seed_base is None else int(seed_base)
    silence = (rank != reporter_rank)

    # Prepare per-worker temp directory
    base_dir = os.path.abspath(os.getcwd())
    tmp_dir = os.path.join(base_dir, 'training_data', 'tmp_parallel', f'run_{run_id}', f'rank_{rank}')
    os.makedirs(tmp_dir, exist_ok=True)

    remaining = int(max(0, games_for_rank))
    chunk_idx = 0
    file_paths: List[str] = []
    while remaining > 0:
        this_chunk = int(min(chunk_size, remaining))
        # Each chunk uses an incremented seed (when provided) for determinism across chunks
        seed = None if seed0 is None else seed0 + int(rank) + chunk_idx
        built = _worker_build(
            games=this_chunk,
            seed=seed,
            temperature=temperature,
            zero_network_reward=bool(zero_network_reward),
            n_step=int(n_step),
            gamma=float(gamma),
            use_heuristic=bool(use_heuristic),
            model_path=model_path,
            silence_io=silence,
        )
        out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npz')
        save_dataset(built, out_path)
        file_paths.append(out_path)
        remaining -= this_chunk
        chunk_idx += 1

    queue.put((rank, file_paths))


def main():
    ap = argparse.ArgumentParser(description='Create AC experience dataset in parallel and combine to a single NPZ')
    ap.add_argument('--games', type=int, default=10, help='Total number of games to simulate (across all processes)')
    ap.add_argument('--num_processes', type=int, default=1, help='Number of parallel worker processes')
    ap.add_argument('--seed', type=int, default=None, help='Base random seed (per-process seeds use base+rank)')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--zero_network_reward', action='store_true', help='Zero out network value as immediate reward')
    ap.add_argument('--n_step', type=int, default=3, help='N for n-step returns')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor in (0,1]')
    ap.add_argument('--out', type=str, default=None, help='Output .npz path (placed under training_data/)')
    ap.add_argument('--use_heuristic', action='store_true', help='Use RecordingHeuristicACPlayer (generation 0)')
    ap.add_argument('--model', type=str, default=None, help='Path to AC network .pt to load')
    ap.add_argument('--chunk_size', type=int, default=500, help='Flush each worker chunk to disk after this many games (to reduce memory)')
    ap.add_argument('--keep_partials', action='store_true', help='Keep partial chunk files after merge (default: delete)')
    args = ap.parse_args()

    total_games = max(1, int(args.games))
    num_procs = max(1, int(args.num_processes))

    # Partition games across processes as evenly as possible
    base = total_games // num_procs
    rem = total_games % num_procs
    games_splits = [base + (1 if i < rem else 0) for i in range(num_procs)]

    # Designate one process to report progress (the first with >0 games)
    reporter_rank = next((i for i, g in enumerate(games_splits) if g > 0), 0)

    # Prepare run id and temp base directory
    run_id = time.strftime('%Y%m%d_%H%M%S')
    tmp_base = os.path.join(os.path.abspath(os.getcwd()), 'training_data', 'tmp_parallel', f'run_{run_id}')
    os.makedirs(tmp_base, exist_ok=True)

    # Launch workers
    procs: List[mp.Process] = []
    queue: mp.Queue = mp.Queue()

    # On Windows, need to protect process creation under __main__ guard; it is below.
    for i, g in enumerate(games_splits):
        if g <= 0:
            continue
        p = mp.Process(
            target=run_worker,
            args=(
                i,
                g,
                args.seed,
                args.temperature,
                bool(args.zero_network_reward),
                int(args.n_step),
                float(args.gamma),
                bool(args.use_heuristic),
                args.model,
                reporter_rank,
                queue,
                run_id,
                int(max(1, args.chunk_size)),
            ),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    # Collect results (ensure deterministic order by rank)
    collected: Dict[int, List[str]] = {}
    remaining = sum(1 for g in games_splits if g > 0)
    while remaining > 0:
        r_rank, r_paths = queue.get()
        collected[int(r_rank)] = list(r_paths)
        remaining -= 1

    # Join processes
    for p in procs:
        p.join()

    # Load all partial chunk files and combine
    all_dicts: List[Dict[str, Any]] = []
    for r in sorted(collected.keys()):
        for pth in collected[r]:
            all_dicts.append(_load_npz_as_dict(pth))
    combined = _combine_results(all_dicts)

    # Prepare output path under training_data/
    base_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = os.path.join(out_dir, args.out)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'ac_parallel_{ts}.npz')

    save_dataset(combined, out_path)

    # Cleanup partials unless requested to keep
    if not args.keep_partials:
        try:
            for r in collected.keys():
                for pth in collected[r]:
                    if os.path.isfile(pth):
                        os.remove(pth)
            # remove empty directories
            for root, dirs, files in os.walk(tmp_base, topdown=False):
                if not files and not dirs:
                    try:
                        os.rmdir(root)
                    except Exception:
                        pass
        except Exception:
            pass
    return out_path


if __name__ == '__main__':
    # Use spawn for better Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
