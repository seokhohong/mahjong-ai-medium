#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np


ACTION_HEAD_ORDER: List[str] = [
    'discard',       # uses tile head
    'riichi',        # uses tile head
    'tsumo',
    'ron',
    'pass',
    'kan',           # uses tile head (kakan/ankan pass tile; daiminkan uses no-op)
    'chi_low_noaka',
    'chi_mid_noaka',
    'chi_high_noaka',
    'chi_low_aka',
    'chi_mid_aka',
    'chi_high_aka',
    'pon_noaka',
    'pon_aka',
]
ACTION_HEAD_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ACTION_HEAD_ORDER)}


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, np.ndarray] = {k: data[k] for k in data.keys()}
    data.close()
    return out


def per_actor_metrics(data: Dict[str, np.ndarray]) -> None:
    actor_ids = data.get('actor_ids')
    if actor_ids is None:
        raise KeyError("actor_ids not found in dataset; regenerate dataset with actor_ids enabled")
    # Ensure string dtype
    actor_ids = actor_ids.astype(np.str_)

    action_idx = data['action_idx']
    returns = data['returns']

    # Group indices by actor id
    idx_by_actor: Dict[str, List[int]] = defaultdict(list)
    for i, aid in enumerate(actor_ids):
        idx_by_actor[str(aid)].append(i)

    print(f"Found {len(idx_by_actor)} unique actors in samples")
    print("")

    # Global metrics
    print("== Global metrics ==")
    print(f"Samples: {len(action_idx)}")
    print(f"Mean |Return|: {float(np.mean(np.abs(returns))):.6f}")
    pos_frac = float(np.sum(returns > 0.05)) / max(1, len(returns))
    print(f"Return fraction (>0.05): {pos_frac:.6f}")
    global_counts = Counter(action_idx.tolist())
    for idx, ct in sorted(global_counts.items(), key=lambda x: -x[1]):
        name = ACTION_HEAD_ORDER[int(idx)] if int(idx) < len(ACTION_HEAD_ORDER) else f"action_{int(idx)}"
        print(f"{name:>16} {ct/len(action_idx)*100:.6f}%")
    print("")

    # Per-actor metrics
    print("== Per-actor metrics ==")
    for aid, idxs in sorted(idx_by_actor.items()):
        a_idx = np.asarray(idxs, dtype=np.int64)
        a_returns = returns[a_idx]
        a_actions = action_idx[a_idx]
        mean_abs_ret = float(np.mean(np.abs(a_returns))) if a_returns.size else 0.0
        pos_frac = float(np.sum(a_returns > 0.05)) / max(1, a_returns.size)
        counts = Counter(a_actions.tolist())
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(
            f"{(ACTION_HEAD_ORDER[i] if i < len(ACTION_HEAD_ORDER) else f'action_{i}')}: {ct}" for i, ct in top3
        )
        print(f"Actor {aid} | samples={a_returns.size:5d} | mean|ret|={mean_abs_ret:.6f} | frac>0.05={pos_frac:.6f} | top3: {top3_str}")

    print("")


def main():
    ap = argparse.ArgumentParser(description='Inspect AC dataset grouped by actor_ids')
    ap.add_argument('--npz', required=True, help='Path to dataset .npz file')
    ap.add_argument('--show-actions', action='store_true', help='Print full per-actor action distributions')
    args = ap.parse_args()

    data = load_npz(args.npz)
    per_actor_metrics(data)

    if args.show_actions:
        actor_ids = data['actor_ids'].astype(np.str_)
        action_idx = data['action_idx']
        idx_by_actor: Dict[str, List[int]] = defaultdict(list)
        for i, aid in enumerate(actor_ids):
            idx_by_actor[str(aid)].append(i)
        print("== Per-actor full action distributions ==")
        for aid, idxs in sorted(idx_by_actor.items()):
            a_actions = action_idx[np.asarray(idxs, dtype=np.int64)]
            counts = Counter(a_actions.tolist())
            print(f"\nActor {aid}")
            for i in range(len(ACTION_HEAD_ORDER)):
                name = ACTION_HEAD_ORDER[i]
                ct = counts.get(i, 0)
                frac = ct / max(1, len(idxs))
                print(f"  {name:>16}: {ct:6d} ({frac*100:6.3f}%)")


if __name__ == '__main__':
    main()
