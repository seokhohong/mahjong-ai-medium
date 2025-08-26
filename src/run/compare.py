#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import Tuple

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from core.learn.ac_player import ACPlayer  # type: ignore
from core.learn.ac_network import ACNetwork  # type: ignore


def _load_model(model_path: str) -> ACNetwork:
    """Load an AC model (and scaler) using ACPlayer convenience, then return its ACNetwork."""
    player = ACPlayer.from_directory(model_path, temperature=1.0)
    return player.network  # ACNetwork


def _extract_feature_tensors(net: ACNetwork, hand_idx, disc_idx, called_idx, game_state_vec) -> Tuple[torch.Tensor, ...]:
    """Use ACNetwork's feature extractor to build tensors for a single sample (batch=1)."""
    h, c, d, g = net.extract_features_from_indexed(hand_idx, disc_idx, called_idx, game_state_vec)
    device = net.torch_module.head_action.weight.device  # current device of the model
    ht = torch.from_numpy(h[None, ...]).to(device)
    ct = torch.from_numpy(c[None, ...]).to(device)
    dt = torch.from_numpy(d[None, ...]).to(device)
    gt = torch.from_numpy(g[None, ...]).to(device)
    return ht, ct, dt, gt


def _joint_log_prob(action_pp: torch.Tensor, tile_pp: torch.Tensor, a_idx: int, t_idx: int) -> float:
    a_p = float(action_pp[0, int(a_idx)].clamp_min(1e-12).log().item())
    t_p = float(tile_pp[0, int(t_idx)].clamp_min(1e-12).log().item())
    return a_p + t_p


def main() -> int:
    ap = argparse.ArgumentParser(description='Compare new model policy/value vs stored (old) policy in a dataset (.npz).')
    ap.add_argument('--data', type=str, required=True, help='Path to dataset .npz containing hand_idx/disc_idx/called_idx/game_state and labels')
    ap.add_argument('--model', type=str, required=True, help='Path to model directory or .pt file (with scaler.pkl alongside)')
    ap.add_argument('--limit', type=int, default=50, help='Max number of samples to display (default: 50)')
    ap.add_argument('--skip', type=int, default=0, help='Skip this many samples from the start')
    ap.add_argument('--topk', type=int, default=5, help='Show top-K entries for action and tile heads')
    ap.add_argument('--only-mismatches', action='store_true', help='Only display rows where new argmax disagrees with dataset labels')
    ap.add_argument('--mmap', action='store_true', help='Memory-map the dataset to reduce RAM')
    args = ap.parse_args()

    # Load model (with scaler)
    net = _load_model(args.model)
    net.torch_module.eval()

    # Load dataset arrays
    data = np.load(args.data, allow_pickle=True, mmap_mode=('r' if args.mmap else None))
    hand_idx = data['hand_idx']
    disc_idx = data['disc_idx']
    called_idx = data['called_idx']
    gsv = data['game_state']
    action_idx = data['action_idx']
    tile_idx = data['tile_idx']
    joint_log_prob = data['joint_log_probs']
    returns = data['returns'] if 'returns' in data else data['return']
    advantages = data['advantages'] if 'advantages' in data else None

    N = int(hand_idx.shape[0])
    start = int(max(0, args.skip))
    end = N if args.limit <= 0 else min(N, start + int(args.limit))

    print("=" * 100)
    print(f"Dataset: {args.data} | Model: {args.model}")
    print(f"Range: [{start}, {end}) out of {N} samples | topK={int(args.topk)} | only_mismatches={bool(args.only_mismatches)}")
    print("-" * 100)

    shown = 0
    for i in range(start, end):
        # Prepare features
        ht, ct, dt, gt = _extract_feature_tensors(net, hand_idx[i], disc_idx[i], called_idx[i], gsv[i])
        with torch.no_grad():
            a_pp, t_pp, val = net.torch_module(ht, ct, dt, gt)
            # Convert to CPU for safe numpy ops
            a_pp_cpu = a_pp.detach().cpu()
            t_pp_cpu = t_pp.detach().cpu()
            val_f = float(val.detach().cpu().numpy()[0][0])

        # Labeled indices
        a_lbl = int(action_idx[i])
        t_lbl = int(tile_idx[i])
        old_logp = float(joint_log_prob[i])
        new_logp = _joint_log_prob(a_pp_cpu, t_pp_cpu, a_lbl, t_lbl)
        dlogp = new_logp - old_logp

        # Argmax predictions
        a_pred = int(torch.argmax(a_pp_cpu, dim=1).item())
        t_pred = int(torch.argmax(t_pp_cpu, dim=1).item())
        mismatch = (a_pred != a_lbl) or (t_pred != t_lbl)
        if args.only_mismatches and not mismatch:
            continue

        shown += 1
        old_val_str = "n/a"
        if advantages is not None:
            old_value = float(returns[i]) - float(advantages[i])
            old_val_str = f"{old_value:+.4f}"
        print(f"[{i}] old_logp={old_logp:+.6f} | new_logp={new_logp:+.6f} | delta={dlogp:+.6f} | new_value={val_f:+.4f} | old_value={old_val_str} | return={float(returns[i]):+.4f}")
        print(f"    labels: action={a_lbl} tile={t_lbl} | preds: action={a_pred} tile={t_pred}")

        # Top-K for action head
        k = int(max(1, args.topk))
        a_topv, a_topi = torch.topk(a_pp_cpu[0], k)
        t_topv, t_topi = torch.topk(t_pp_cpu[0], k)
        a_pairs = [f"{int(idx)}:{float(p):.4f}" for p, idx in zip(a_topv.tolist(), a_topi.tolist())]
        t_pairs = [f"{int(idx)}:{float(p):.4f}" for p, idx in zip(t_topv.tolist(), t_topi.tolist())]
        print(f"    action_top{ k }: " + ", ".join(a_pairs))
        print(f"    tile_top{ k }:   " + ", ".join(t_pairs))
        print("-" * 100)

    if shown == 0:
        print("No samples matched filters to display.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
