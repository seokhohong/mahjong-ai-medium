#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, Any

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # type: ignore

from core.learn.ac_player import ACPlayer
from core.learn.ac_network import ACNetwork
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.run.train_model import (
    ACDataset,
    _apply_runtime_config,
    _resolve_loader_defaults,
    _prepare_batch_tensors,
)


def train_value(
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str | None = None,
    val_split: float = 0.1,
    init_model: str | None = None,
    *,
    hidden_size: int = 128,
    embedding_dim: int = 16,
    low_mem_mode: bool = False,
    # Runtime/loader tuning
    os_hint: str | None = None,
    start_method: str | None = None,
    torch_threads: int | None = None,
    interop_threads: int | None = None,
    dl_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
) -> None:
    # Device selection
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dev = torch.device('mps')
        else:
            dev = torch.device('cpu')
    else:
        dev = torch.device(device)

    _apply_runtime_config(
        os_hint=os_hint,
        start_method=start_method,
        torch_threads=torch_threads,
        interop_threads=interop_threads,
    )

    # Init/load network
    if init_model:
        player = ACPlayer.from_directory(init_model)
        net = player.network.to(dev)
        model = net.torch_module
        ds = ACDataset(dataset_path, net, fit_scaler=False, precompute_features=not low_mem_mode, mmap=low_mem_mode)
        print(f"Loaded initial weights from {init_model}")
    else:
        gsv_scaler = StandardScaler()
        net = ACNetwork(gsv_scaler=gsv_scaler, hidden_size=hidden_size, embedding_dim=embedding_dim, temperature=0.05)
        net = net.to(dev)
        model = net.torch_module
        ds = ACDataset(dataset_path, net, fit_scaler=True, precompute_features=not low_mem_mode, mmap=low_mem_mode)

    # Split
    n = len(ds)
    k = int(max(0, min(n, round(float(val_split) * n))))
    if k > 0 and n > 1:
        idx = np.random.permutation(n)
        val_idx = idx[:k].tolist()
        train_idx = idx[k:].tolist()
        ds_train = Subset(ds, train_idx) if len(train_idx) > 0 else ds
        ds_val = Subset(ds, val_idx)
    else:
        ds_train = ds
        ds_val = None

    resolved = _resolve_loader_defaults(
        os_hint=os_hint,
        dl_workers=dl_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    if low_mem_mode:
        resolved['dl_workers'] = 0
        resolved['pin_memory'] = False
        resolved['prefetch_factor'] = 1
        resolved['persistent_workers'] = False

    dl = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=resolved['dl_workers'],
        pin_memory=resolved['pin_memory'] and (dev.type in ('cuda', 'mps')),
        prefetch_factor=resolved['prefetch_factor'] if resolved['dl_workers'] > 0 else None,
        persistent_workers=resolved['persistent_workers'] if resolved['dl_workers'] > 0 else False,
    )
    dl_val = (
        DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=resolved['dl_workers'],
            pin_memory=resolved['pin_memory'] and (dev.type in ('cuda', 'mps')),
            prefetch_factor=resolved['prefetch_factor'] if resolved['dl_workers'] > 0 else None,
            persistent_workers=resolved['persistent_workers'] if resolved['dl_workers'] > 0 else False,
        ) if ds_val is not None else None
    )

    # Optimize full network parameters, value loss only
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_val_loss = 0.0
        total_examples = 0
        progress = tqdm(dl, desc=f"ValEpoch {epoch+1}/{epochs}", leave=False)
        for batch in progress:
            (
                hand, calls, disc, gsv,
                _action_idx, _tile_idx,
                _joint_old_log_probs,
                _advantages, returns,
            ) = _prepare_batch_tensors(batch, dev)

            # Forward
            _a_pp, _t_pp, values = model(hand.float(), calls.float(), disc.float(), gsv.float())
            values = values.squeeze(1)
            value_loss = F.mse_loss(values, returns)

            opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = int(gsv.size(0))
            total_examples += bsz
            total_val_loss += float(value_loss.item()) * bsz
            progress.set_postfix(val=f"{float(value_loss.item()):.4f}")

        den = max(1, total_examples)
        print(f"Epoch {epoch+1}/{epochs} [train] - value: {total_val_loss/den:.4f}")

        # Validation
        if dl_val is not None:
            model.eval()
            v_total = 0.0
            v_count = 0
            with torch.no_grad():
                for vb in dl_val:
                    (
                        hand, calls, disc, gsv,
                        _action_idx, _tile_idx,
                        _joint_old_log_probs,
                        _advantages, returns,
                    ) = _prepare_batch_tensors(vb, dev)
                    _a_pp, _t_pp, values = model(hand.float(), calls.float(), disc.float(), gsv.float())
                    values = values.squeeze(1)
                    l = F.mse_loss(values, returns)
                    bsz = int(gsv.size(0))
                    v_total += float(l.item()) * bsz
                    v_count += bsz
            print(f"Epoch {epoch+1}/{epochs} [val] - value: {v_total/max(1,v_count):.4f}")
            model.train()

    # Do not save the final model — this script is for debugging value optimization
    print("Training complete (value head only). Model was not saved by design.")


def main():
    ap = argparse.ArgumentParser(description='Train only the value head on AC dataset (.npz) for debugging')
    ap.add_argument('--data', type=str, required=True, help='Path to .npz built by create_dataset.py')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--init', type=str, default=None, help='Path to initial AC model weights/module to load')
    ap.add_argument('--hidden_size', type=int, default=128, help='Hidden size for ACNetwork (if initializing a fresh model)')
    ap.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for ACNetwork (if initializing a fresh model)')
    ap.add_argument('--low_mem_mode', action='store_true', help='Reduce RAM usage: no precompute, memmap dataset, workers=0, no pin_memory/persistence, prefetch=1')
    # DataLoader tuning (subset)
    ap.add_argument('--dl_workers', type=int, default=None)
    ap.add_argument('--prefetch_factor', type=int, default=None)
    args = ap.parse_args()

    train_value(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_model=args.init,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        low_mem_mode=bool(args.low_mem_mode),
        dl_workers=args.dl_workers,
        prefetch_factor=args.prefetch_factor,
    )


if __name__ == '__main__':
    main()
