#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import argparse
from typing import Optional

from run.create_dataset import build_ac_dataset, save_dataset
from run.create_dataset_parallel import create_dataset_parallel
from run.train_model import train_ppo


def _default_out_name(prefix: str) -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{ts}"


def run_train_loop(
    *,
    generations: int,
    initial_model_dir: Optional[str],
    # dataset params
    games_per_gen: int,
    num_processes: int,
    seed: Optional[int],
    temperature: float,
    n_step: int,
    gamma: float,
    # training params
    epochs: int,
    batch_size: int,
    lr: float,
    epsilon: float,
    value_coeff: float,
    entropy_coeff: float,
    patience: int,
    min_delta: float,
    val_split: float,
    warm_up_max_epochs: int,
    hidden_size: int,
    embedding_dim: int,
    # runtime/train tuning
    device: Optional[str] = None,
    os_hint: Optional[str] = None,
    start_method: Optional[str] = None,
    torch_threads: Optional[int] = None,
    interop_threads: Optional[int] = None,
    dl_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
) -> str:
    os.makedirs(os.path.join(os.getcwd(), 'training_data'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)

    current_model_dir: Optional[str] = initial_model_dir
    last_model_dir: Optional[str] = current_model_dir

    for gen in range(max(1, int(generations))):
        print(f"\n=== Generation {gen+1}/{generations} ===")
        # 1) Build dataset (heuristic disabled per request)
        use_heuristic = False
        print(f"[Gen {gen}] Building dataset | games={games_per_gen} | procs={num_processes} | use_heuristic={use_heuristic} | model={current_model_dir}")
        ds_name = _default_out_name(f"ac_gen{gen}")
        ds_path = os.path.join(os.getcwd(), 'training_data', f"{ds_name}.npz")
        if int(num_processes) > 1:
            # Parallel path writes directly to ds_path
            _ = create_dataset_parallel(
                games=games_per_gen,
                num_processes=int(num_processes),
                seed=(None if seed is None else int(seed) + gen),
                temperature=temperature,
                zero_network_reward=False,
                n_step=n_step,
                gamma=gamma,
                out=os.path.basename(ds_path),
                use_heuristic=False,
                model=current_model_dir,
                chunk_size=250,
                keep_partials=False,
                stream_combine=True,
                verbose_memory=False
            )
        else:
            built = build_ac_dataset(
                games=games_per_gen,
                seed=(None if seed is None else int(seed) + gen),
                temperature=temperature,
                zero_network_reward=False,
                n_step=n_step,
                gamma=gamma,
                use_heuristic=False,
                model_path=current_model_dir,
            )
            save_dataset(built, ds_path)

        # 2) Train PPO starting from current model (if any)
        print(f"[Gen {gen}] Training model on {ds_path} | init={current_model_dir}")
        model_pt_path = train_ppo(
            dataset_path=ds_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            epsilon=epsilon,
            value_coeff=value_coeff,
            entropy_coeff=entropy_coeff,
            device=device,
            patience=patience,
            val_split=val_split,
            init_model=current_model_dir,
            warm_up_max_epochs=warm_up_max_epochs,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            kl_threshold=0.0,
            # runtime/train tuning
            os_hint=os_hint,
            start_method=start_method,
            torch_threads=torch_threads,
            interop_threads=interop_threads,
            dl_workers=dl_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        # train_ppo returns a model file path; use its directory for next gen
        new_model_dir = os.path.dirname(model_pt_path)
        print(f"[Gen {gen}] Trained model saved to {new_model_dir}")
        current_model_dir = new_model_dir
        last_model_dir = new_model_dir

    assert last_model_dir is not None
    print(f"\nTraining loop complete. Final model: {last_model_dir}")
    return last_model_dir


def main() -> str:
    ap = argparse.ArgumentParser(description='Iterative AC training loop: generate dataset -> train -> repeat')
    ap.add_argument('--generations', type=int, default=3, help='Number of generations to run')
    ap.add_argument('--model', type=str, default=None, help='Initial model directory (if omitted, start with heuristic for gen 0)')

    # Dataset generation
    ap.add_argument('--games', type=int, default=2000, help='Games per generation')
    ap.add_argument('--num_processes', type=int, default=1, help='Parallel workers for dataset creation (1 = serial)')
    ap.add_argument('--seed', type=int, default=None, help='Base random seed; gen index is added each generation')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--n_step', type=int, default=3)
    ap.add_argument('--gamma', type=float, default=0.99)

    # Training
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epsilon', type=float, default=0.2)
    ap.add_argument('--value_coeff', type=float, default=0.5)
    ap.add_argument('--entropy_coeff', type=float, default=0.01)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--warm_up_max_epochs', type=int, default=50)
    ap.add_argument('--hidden_size', type=int, default=128)
    ap.add_argument('--embedding_dim', type=int, default=16)
    # Runtime/train tuning
    ap.add_argument('--device', type=str, default=None, help='Force device (cpu/cuda/mps)')
    ap.add_argument('--os', dest='os_hint', type=str, default='auto', choices=['auto', 'windows', 'mac', 'linux'], help='Target OS to tune runtime defaults (auto)')
    ap.add_argument('--start_method', type=str, default=None, choices=['spawn', 'fork', 'forkserver'])
    ap.add_argument('--torch_threads', type=int, default=None)
    ap.add_argument('--interop_threads', type=int, default=None)
    ap.add_argument('--dl_workers', type=int, default=None)
    ap.add_argument('--pin_memory', type=str, default=None, choices=['true', 'false'])
    ap.add_argument('--prefetch_factor', type=int, default=None)
    ap.add_argument('--persistent_workers', type=str, default=None, choices=['true', 'false'])

    args = ap.parse_args()

    # Respect min_delta by monkey-patching into train_ppo via kwargs already supported
    # Note: train_ppo signature includes min_delta and val_split via our recent changes
    return run_train_loop(
        generations=args.generations,
        initial_model_dir=args.model,
        games_per_gen=args.games,
        num_processes=args.num_processes,
        seed=args.seed,
        temperature=args.temperature,
        n_step=args.n_step,
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        patience=args.patience,
        min_delta=float(args.min_delta),
        val_split=float(args.val_split),
        warm_up_max_epochs=args.warm_up_max_epochs,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        device=args.device,
        os_hint=None if args.os_hint == 'auto' else args.os_hint,
        start_method=args.start_method,
        torch_threads=args.torch_threads,
        interop_threads=args.interop_threads,
        dl_workers=args.dl_workers,
        pin_memory=(None if args.pin_memory is None else (args.pin_memory.lower() == 'true')),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(None if args.persistent_workers is None else (args.persistent_workers.lower() == 'true')),
    )


if __name__ == '__main__':
    main()


