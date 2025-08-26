#!/usr/bin/env python3
from __future__ import annotations

import os
import copy
import math
import time
import argparse
import pickle
from typing import Dict, Any, Tuple

import numpy as np

# PyTorch is a required dependency; run within the project's .venv312 environment.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm  # type: ignore

from core.learn.ac_player import ACPlayer
from core.learn.ac_network import ACNetwork
from sklearn.preprocessing import StandardScaler  # type: ignore

# ---- Helpers: runtime/loader configuration ----
def _detect_os_from_env(os_hint: str | None) -> str:
    if os_hint in ('windows', 'mac', 'linux'):
        return os_hint
    plat = os.sys.platform
    if plat.startswith('win'):
        return 'windows'
    if plat == 'darwin':
        return 'mac'
    return 'linux'


def _apply_runtime_config(*, os_hint: str | None, start_method: str | None, torch_threads: int | None, interop_threads: int | None) -> None:
    # Set multiprocessing start method if specified or if platform needs it
    resolved_os = _detect_os_from_env(os_hint)
    try:
        import torch.multiprocessing as mp
        if start_method is None:
            # Defaults: mac/windows -> spawn; linux -> leave default (often fork)
            if resolved_os in ('mac', 'windows'):
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn', force=True)
        else:
            if mp.get_start_method(allow_none=True) != start_method:
                mp.set_start_method(start_method, force=True)
    except Exception:
        pass

    # Torch intra/inter-op threads
    try:
        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))
        if interop_threads is not None and interop_threads > 0:
            torch.set_num_interop_threads(int(interop_threads))
    except Exception:
        pass


def _resolve_loader_defaults(*, os_hint: str | None, dl_workers: int | None, pin_memory: bool | None, prefetch_factor: int | None, persistent_workers: bool | None) -> Dict[str, Any]:
    resolved_os = _detect_os_from_env(os_hint)
    cpu_count = max(1, (os.cpu_count() or 1))
    # Reasonable defaults per-OS
    if dl_workers is None:
        if resolved_os == 'mac':
            # macOS often benefits from a few workers; avoid oversubscription
            dl_workers = min(8, max(0, cpu_count // 2))
        elif resolved_os == 'windows':
            dl_workers = min(8, max(0, cpu_count // 2))
        else:  # linux
            dl_workers = min(16, max(0, cpu_count - 2))
    if pin_memory is None:
        pin_memory = (resolved_os != 'mac')  # Pinned memory less impactful for MPS
    if prefetch_factor is None:
        prefetch_factor = 2
    if persistent_workers is None:
        persistent_workers = True
    return {
        'dl_workers': int(max(0, dl_workers)),
        'pin_memory': bool(pin_memory),
        'prefetch_factor': int(max(1, prefetch_factor)),
        'persistent_workers': bool(persistent_workers),
    }

# not using called discards yet
class ACDataset(Dataset):
    def __init__(self, npz_path: str, net: ACNetwork, fit_scaler: bool = True, *, precompute_features: bool = True, mmap: bool = False):
        # Use memory-mapped loading if requested to reduce peak RAM
        data = np.load(npz_path, allow_pickle=True, mmap_mode=('r' if mmap else None))
        # New compact format fields
        self._hand_idx = data['hand_idx']
        self._disc_idx = data['disc_idx']
        self._called_idx = data['called_idx']
        self._gsv = data['game_state']
        self.action_idx = data['action_idx']
        self.tile_idx = data['tile_idx']
        self.game_ids = data['game_ids']
        # Load returns/advantages
        self.returns = data['returns'].astype(np.float32)
        self.advantages = data['advantages'].astype(np.float32)
        self.joint_old_log_probs = data['joint_log_probs'].astype(np.float32)

        # Fit the standard scaler so we can properly use extract_features
        if fit_scaler:
            # StandardScaler handles memmap arrays; this may stream from disk when mmap=True
            net.fit_scaler(self._gsv)

        self._net = net
        self._precompute = bool(precompute_features)

        if self._precompute:
            # Pre-extract indexed states and transform to embedded sequences once
            hand_list = []
            calls_list = []
            disc_list = []
            gsv_list = []
            N = len(self._hand_idx)
            for i in range(N):
                h, c, d, g = net.extract_features_from_indexed(
                    np.asarray(self._hand_idx[i], dtype=np.int32),
                    np.asarray(self._disc_idx[i], dtype=np.int32),
                    np.asarray(self._called_idx[i], dtype=np.int32),
                    np.asarray(self._gsv[i], dtype=np.float32),
                )
                hand_list.append(h.astype(np.float32))
                calls_list.append(c.astype(np.float32))
                disc_list.append(d.astype(np.float32))
                gsv_list.append(g.astype(np.float32))
            self.hand = np.asarray(hand_list, dtype=np.float32)
            self.calls = np.asarray(calls_list, dtype=np.float32)
            self.disc = np.asarray(disc_list, dtype=np.float32)
            self.gsv = np.asarray(gsv_list, dtype=np.float32)
        else:
            # Defer feature extraction to __getitem__ to minimize resident memory
            self.hand = None  # type: ignore
            self.calls = None  # type: ignore
            self.disc = None  # type: ignore
            self.gsv = None  # type: ignore

    def train_val_subsets_by_game(self, val_frac: float, *, seed: int | None = None) -> tuple[Subset, Subset] | tuple[Subset, None]:
        """Return train/validation Subset(s) grouped by game.

        - Ensures that all samples from the same game_id are placed entirely in either train or validation.
        - If `val_frac <= 0`, returns (train_subset, None).
        - Requires that the dataset npz include a `game_ids` array; otherwise raises a ValueError.
        """
        import numpy as _np
        if val_frac <= 0.0:
            # No validation split requested
            return Subset(self, list(range(len(self)))), None

        if self.game_ids is None:
            raise ValueError("ACDataset: game_ids not found in dataset; cannot split by game. Please regenerate dataset with 'game_ids'.")

        game_ids_arr = _np.asarray(self.game_ids)
        unique_games = _np.unique(game_ids_arr)
        if unique_games.size == 0:
            # Degenerate case; treat as no validation
            return Subset(self, list(range(len(self)))), None

        rng = _np.random.default_rng(seed)
        perm_games = rng.permutation(unique_games)
        k_val = int(round(float(unique_games.size) * float(max(0.0, min(1.0, val_frac)))))
        k_val = int(max(0, min(unique_games.size, k_val)))

        val_games = set(perm_games[:k_val].tolist()) if k_val > 0 else set()

        train_indices = [int(i) for i, gid in enumerate(game_ids_arr) if gid not in val_games]
        val_indices = [int(i) for i, gid in enumerate(game_ids_arr) if gid in val_games]
        # removed guard against empty splits since we expect the dataset to be large enough

        ds_train = Subset(self, train_indices)
        ds_val = Subset(self, val_indices) if len(val_indices) > 0 else None
        return ds_train, ds_val

    def __len__(self) -> int:
        return int(len(self._hand_idx))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._precompute:
            hand = self.hand[idx]
            calls = self.calls[idx]
            disc = self.disc[idx]
            gsv = self.gsv[idx]
        else:
            # Compute features on demand to save RAM
            h, c, d, g = self._net.extract_features_from_indexed(
                np.asarray(self._hand_idx[idx], dtype=np.int32),
                np.asarray(self._disc_idx[idx], dtype=np.int32),
                np.asarray(self._called_idx[idx], dtype=np.int32),
                np.asarray(self._gsv[idx], dtype=np.float32),
            )
            hand = h.astype(np.float32)
            calls = c.astype(np.float32)
            disc = d.astype(np.float32)
            gsv = g.astype(np.float32)
        return {
            'hand': hand,
            'calls': calls,
            'disc': disc,
            'gsv': gsv,
            'action_idx': int(self.action_idx[idx]),
            'tile_idx': int(self.tile_idx[idx]),
            'return': float(self.returns[idx]),
            'advantage': float(self.advantages[idx]),
            'joint_old_log_prob': float(self.joint_old_log_probs[idx]),
        }


def batch_to_tensors(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, ...]:
    hand = np.stack([b['hand_idx'] for b in batch])  # (B,12)
    disc = np.stack([b['disc_idx'] for b in batch])  # (B,4,K)
    called = np.stack([b['called_idx'] for b in batch])  # (B,4,3,3)
    gsv = np.stack([b['game_state'] for b in batch])  # (B,G)
    B = hand.shape[0]
    # Convert to embeddings sequence layout expected by ACNetwork.torch_module
    # Hand: (B, embed, 12), Calls: (B, embed, 36), Discards: (B, embed, 4*K)
    # We will embed indices using ACNetwork's internal embedding table via a small helper
    return (
        torch.from_numpy(hand).to(device),
        torch.from_numpy(disc).to(device),
        torch.from_numpy(called).to(device),
        torch.from_numpy(gsv).to(device),
    )



def _prepare_batch_tensors(batch: Dict[str, Any], dev: torch.device):
    def _to_dev(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    hand = torch.from_numpy(np.stack(batch['hand'])).to(dev)
    calls = torch.from_numpy(np.stack(batch['calls'])).to(dev)
    disc = torch.from_numpy(np.stack(batch['disc'])).to(dev)
    gsv = torch.from_numpy(np.stack(batch['gsv'])).to(dev)

    action_idx = _to_dev(batch['action_idx'], torch.long)
    tile_idx = _to_dev(batch['tile_idx'], torch.long)
    joint_old_log_probs = _to_dev(batch['joint_old_log_prob'], torch.float32)
    advantages = _to_dev(batch['advantage'], torch.float32)
    returns = _to_dev(batch['return'], torch.float32)

    return (
        hand, calls, disc, gsv,
        action_idx, tile_idx,
        joint_old_log_probs,
        advantages, returns,
    )

def safe_entropy_calculation(probs, min_prob=1e-8):
    """
    Compute entropy safely, avoiding NaN from 0 * log(0) cases.

    Standard entropy: H = -sum(p * log(p))
    Safe entropy: H = -sum(p * log(max(p, epsilon)))
    """
    # Clamp probabilities to avoid log(0)
    probs_safe = probs.clamp_min(min_prob)

    # Compute entropy: -sum(p * log(p))
    # Use the original probs for weighting, but safe probs for log
    entropy = -(probs * probs_safe.log()).sum(dim=1).mean()

    return entropy

def _evaluate_model_on_combined_dataset(model: torch.nn.Module, dl_train, dl_val, dev: torch.device, bc_fallback_ratio: float, description: str = ""):
    """Evaluate model performance on the combined train+validation dataset with weighted averaging."""
    model.eval()
    
    # Combine datasets with proper weighting
    datasets = []
    weights = []
    
    if dl_train is not None:
        datasets.append(('train', dl_train))
        weights.append(len(dl_train.dataset))
    
    if dl_val is not None:
        datasets.append(('validation', dl_val))
        weights.append(len(dl_val.dataset))
    
    if not datasets:
        print(f"[{description}] No datasets available for evaluation")
        return
    
    total_weight = sum(weights)
    
    # Accumulate metrics across all datasets
    total_loss = total_policy_loss = total_value_loss = 0.0
    total_samples = total_correct = 0
    
    with torch.no_grad():
        for (dataset_name, dataloader), weight in zip(datasets, weights):
            dataset_loss = dataset_policy = dataset_value = 0.0
            dataset_samples = dataset_correct = 0
            
            for batch in dataloader:
                (hand, calls, disc, gsv, action_idx, tile_idx, 
                 joint_old_log_probs, advantages, returns) = _prepare_batch_tensors(batch, dev)
                
                # Compute losses
                total_v, policy_loss_v, value_loss_v, bsz, _ = _compute_losses(
                    model, hand, calls, disc, gsv, action_idx, tile_idx,
                    joint_old_log_probs, advantages, returns,
                    bc_fallback_ratio=bc_fallback_ratio,
                    bc_weight=0.0, policy_weight=1.0,
                )
                
                # Compute accuracy
                a_logits, t_logits, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                pred_a = torch.argmax(a_logits, dim=1)
                pred_t = torch.argmax(t_logits, dim=1)
                both_correct = (pred_a == action_idx) & (pred_t == tile_idx)
                
                # Accumulate for this dataset
                dataset_loss += float(total_v.item()) * bsz
                dataset_policy += float(policy_loss_v.item()) * bsz
                dataset_value += float(value_loss_v.item()) * bsz
                dataset_samples += bsz
                dataset_correct += int(both_correct.sum().item())
            
            # Weight this dataset's contribution
            dataset_weight_factor = weight / total_weight
            total_loss += (dataset_loss / max(1, dataset_samples)) * dataset_weight_factor
            total_policy_loss += (dataset_policy / max(1, dataset_samples)) * dataset_weight_factor
            total_value_loss += (dataset_value / max(1, dataset_samples)) * dataset_weight_factor
            total_samples += dataset_samples
            total_correct += dataset_correct
            
            print(f"[{description}] {dataset_name}: loss={dataset_loss/max(1,dataset_samples):.4f} | "
                  f"policy={dataset_policy/max(1,dataset_samples):.4f} | "
                  f"value={dataset_value/max(1,dataset_samples):.4f} | "
                  f"acc={dataset_correct/max(1,dataset_samples):.4f} | samples={dataset_samples}")
    
    # Overall weighted metrics
    overall_acc = total_correct / max(1, total_samples)
    print(f"[{description}] COMBINED: loss={total_loss:.4f} | policy={total_policy_loss:.4f} | "
          f"value={total_value_loss:.4f} | acc={overall_acc:.4f} | total_samples={total_samples}")
    
    model.train()
    return {
        'total_loss': total_loss,
        'policy_loss': total_policy_loss, 
        'value_loss': total_value_loss,
        'accuracy': overall_acc,
        'total_samples': total_samples
    }

def _compute_losses(
        model: torch.nn.Module,
        hand: torch.Tensor,
        calls: torch.Tensor,
        disc: torch.Tensor,
        gsv: torch.Tensor,
        action_idx: torch.Tensor,
        tile_idx: torch.Tensor,
        joint_old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        *,
        epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        bc_fallback_ratio: float = 5.0,
        bc_weight: float = 0.0,
        policy_weight: float = 1.0,
        print_debug: bool = False,
):
    a_pp, t_pp, val = model(hand.float(), calls.float(), disc.float(), gsv.float())
    val = val.squeeze(1)
    batch_size_curr = int(gsv.size(0))

    # Normalize advantages per batch
    # This ensures consistent scale and prevents exploding gradients
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Get chosen action probabilities
    a_idx = action_idx.to(device=a_pp.device, dtype=torch.long).view(-1, 1)
    t_idx = tile_idx.to(device=t_pp.device, dtype=torch.long).view(-1, 1)
    chosen_a = torch.gather(a_pp.clamp_min(1e-8), 1, a_idx).squeeze(1)
    chosen_t = torch.gather(t_pp.clamp_min(1e-8), 1, t_idx).squeeze(1)

    # Compute log probabilities
    logp_a = chosen_a.log()
    logp_t = chosen_t.log()

    # Joint log-prob and ratio against old joint
    logp_joint = logp_a + logp_t
    ratio_joint = torch.exp(logp_joint - joint_old_log_probs)

    # Prepare lightweight debug aggregates for epoch-level reporting
    # We return sums to enable exact aggregation across variable batch sizes
    _max_ratio = float(ratio_joint.max().item())
    dbg = {
        'bsz': batch_size_curr,
        'ratio_sum': float(ratio_joint.sum().item()),
        'ratio_sumsq': float((ratio_joint * ratio_joint).sum().item()),
        'adv_sum': float(advantages.sum().item()),
        'adv_sumsq': float((advantages * advantages).sum().item()),
        'clipped_cnt': int(((ratio_joint < (1 - epsilon)) | (ratio_joint > (1 + epsilon))).sum().item()),
        # Added: track BC fallback usage for this batch and the max ratio observed
        'bc_fallback_used': False,
        'max_ratio': _max_ratio,
    }
    if print_debug:
        r_mean = dbg['ratio_sum'] / max(1, dbg['bsz'])
        r_var = max(0.0, dbg['ratio_sumsq'] / max(1, dbg['bsz']) - r_mean * r_mean)
        r_std = math.sqrt(r_var)
        a_mean = dbg['adv_sum'] / max(1, dbg['bsz'])
        a_var = max(0.0, dbg['adv_sumsq'] / max(1, dbg['bsz']) - a_mean * a_mean)
        a_std = math.sqrt(a_var)
        clip_rate = dbg['clipped_cnt'] / max(1, dbg['bsz'])
        print(f"Joint ratio: mean={r_mean:.4f}, std={r_std:.4f}")
        print(f"Advantage stats: mean={a_mean:.6f}, std={a_std:.6f}")
        print(f"Joint clipped: {clip_rate:.3f}")

    # Value-only path: when both policy weights are 0, train only the value head
    if (bc_weight == 0.0) and (policy_weight == 0.0):
        policy_loss = torch.zeros((), device=val.device, dtype=val.dtype)
        value_loss = F.mse_loss(val, returns)
        total = value_loss
        dbg = {
            'bsz': batch_size_curr,
            'ratio_sum': 0.0,
            'ratio_sumsq': 0.0,
            'adv_sum': 0.0,
            'adv_sumsq': 0.0,
            'clipped_cnt': 0,
            'bc_fallback_used': False,
            'max_ratio': 0.0,
        }
        return total, policy_loss, value_loss, batch_size_curr, dbg

    # Compute BC policy loss (NLL on current logits)
    log_a = (a_pp.clamp_min(1e-8)).log()
    log_t = (t_pp.clamp_min(1e-8)).log()
    bc_loss_a = F.nll_loss(log_a, action_idx.to(dtype=torch.long, device=a_pp.device))
    bc_loss_t = F.nll_loss(log_t, tile_idx.to(dtype=torch.long, device=t_pp.device))
    bc_policy_loss = bc_loss_a + bc_loss_t

    # Compute PPO policy loss (clipped surrogate on joint ratio)
    clipped_ratio = torch.clamp(ratio_joint, 1 - epsilon, 1 + epsilon)
    ppo_policy_loss = -torch.min(
        ratio_joint * advantages,
        clipped_ratio * advantages
    ).mean()

    # Active weights
    bw = float(bc_weight)
    pw = float(policy_weight)

    # Fallback to BC when ratios explode and PPO component is active
    if (pw > 0.0) and (_max_ratio > bc_fallback_ratio):
        if print_debug:
            print(f"Switching weights to BC due to high joint ratio: {_max_ratio:.2f}")
        # use 10% BC weight to prevent gradient shock
        bw, pw = 0.1, 0.0
        dbg['bc_fallback_used'] = True

    # Mix policy losses
    policy_loss = bw * bc_policy_loss + pw * ppo_policy_loss

    # Value and entropy terms
    value_loss = F.mse_loss(val, returns)
    entropy_a = safe_entropy_calculation(a_pp)
    entropy_t = safe_entropy_calculation(t_pp)
    entropy = entropy_a + entropy_t

    total = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    return total, policy_loss, value_loss, batch_size_curr, dbg


def _run_value_pretraining(
    model: torch.nn.Module,
    dl: DataLoader,
    dl_val: DataLoader | None,
    dev: torch.device,
    lr: float,
    value_lr: float | None,
    bc_fallback_ratio: float,
) -> None:
    """Run value-only pretraining for specified number of epochs."""
    value_pre_epochs = int(os.environ.get('MJ_VALUE_EPOCHS', '0'))
    if hasattr(train_ppo, '_value_epochs_override'):
        value_pre_epochs = int(getattr(train_ppo, '_value_epochs_override'))
    
    if value_pre_epochs <= 0:
        return
    
    # Use separate learning rate for value pretraining if specified
    effective_value_lr = value_lr if value_lr is not None else lr
    value_opt = torch.optim.Adam(model.parameters(), lr=effective_value_lr)
    print(f"Starting value-only pretraining for {value_pre_epochs} epoch(s) with lr={effective_value_lr:.2e}...")
    
    for ve in range(value_pre_epochs):
        total_v = 0.0
        total_n = 0
        progress = tqdm(dl, desc=f"ValuePre {ve+1}/{value_pre_epochs}", leave=False)
        for batch in progress:
            (
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns,
            ) = _prepare_batch_tensors(batch, dev)
            total_loss_v, policy_loss_v, value_loss_v, bsz, _ = _compute_losses(
                model,
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns,
                bc_fallback_ratio=bc_fallback_ratio,
                bc_weight=0.0, policy_weight=0.0,
            )
            value_opt.zero_grad(set_to_none=True)
            total_loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            value_opt.step()
            total_v += float(value_loss_v.item()) * bsz
            total_n += int(bsz)
            progress.set_postfix(value=f"{float(value_loss_v.item()):.4f}")
        
        train_value_loss = total_v / max(1, total_n)
        
        # Validation phase - for value pretraining, evaluate on ALL batches (no BC filtering)
        validation_value_loss = None
        if dl_val is not None:
            model.eval()
            validation_total_v = 0.0
            validation_total_n = 0
            with torch.no_grad():
                for validation_batch in dl_val:
                    (
                        hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                    ) = _prepare_batch_tensors(validation_batch, dev)
                    
                    # For value pretraining, use all batches regardless of BC mode
                    _, _, value_loss_v, bsz, _ = _compute_losses(
                        model,
                        hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                        bc_fallback_ratio=bc_fallback_ratio,
                        bc_weight=0.0, policy_weight=0.0,
                    )
                    validation_total_v += float(value_loss_v.item()) * bsz
                    validation_total_n += int(bsz)
            
            validation_value_loss = validation_total_v / max(1, validation_total_n)
            model.train()
            print(f"ValuePre {ve+1}/{value_pre_epochs} - train: {train_value_loss:.4f} | validation: {validation_value_loss:.4f}")
        else:
            print(f"ValuePre {ve+1}/{value_pre_epochs} - train: {train_value_loss:.4f}")


def _run_warmup_bc(
    model: torch.nn.Module,
    dl: DataLoader,
    dl_val: DataLoader | None,
    dev: torch.device,
    opt: torch.optim.Optimizer,
    warm_up_acc: float,
    warm_up_max_epochs: int,
    value_coeff: float,
    bc_fallback_ratio: float,
    lr: float,
    value_lr: float | None,
    warm_up_value: bool,
) -> None:
    """Run behavior cloning warm-up until accuracy threshold is met."""
    if not warm_up_acc or warm_up_acc <= 0.0:
        return
    
    # Use value_lr for warm-up if available, otherwise fallback to main lr
    warmup_lr = value_lr if value_lr is not None else lr
    warmup_opt = torch.optim.Adam(model.parameters(), lr=warmup_lr)
    print(f"Warm-up BC using learning rate: {warmup_lr}, value training: {'enabled' if warm_up_value else 'disabled'}")
    
    def _eval_policy_accuracy(dloader: DataLoader) -> float:
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in dloader:
                (
                    hand, calls, disc, gsv,
                    action_idx, tile_idx,
                    _joint_old_log_probs,
                    _advantages,
                    _returns,
                ) = _prepare_batch_tensors(batch, dev)
                a_pp, t_pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                pred_a = torch.argmax(a_pp, dim=1)
                pred_t = torch.argmax(t_pp, dim=1)
                both = (pred_a == action_idx) & (pred_t == tile_idx)
                correct += int(both.sum().item())
                count += int(gsv.size(0))
        model.train()
        return float(correct) / float(max(1, count))
    
    threshold = float(max(0.0, min(1.0, warm_up_acc)))
    # Pre-check: if current model already meets threshold on validation (or train) accuracy, skip warm-up
    initial_acc = _eval_policy_accuracy(dl_val if dl_val is not None else dl)
    if initial_acc >= threshold:
        print(f"Skipping warm-up: initial accuracy {initial_acc:.4f} >= {threshold:.4f}")
        return
    
    print(f"Starting warm-up until accuracy >= {threshold:.2f} (behavior cloning + value regression)...")
    epoch = 0
    reached = False
    while initial_acc < threshold and epoch < int(max(1, warm_up_max_epochs)) and not reached:
        total_loss = 0.0
        total_examples = 0
        pol_loss_acc = 0.0
        value_loss_acc = 0.0
        correct = 0
        progress = tqdm(dl, desc=f"WarmUp {epoch+1}", leave=False)
        for batch in progress:
            (
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages,
                returns,
            ) = _prepare_batch_tensors(batch, dev)

            loss, policy_loss, value_loss, _, _dbg = _compute_losses(
                model,
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages,
                returns,
                value_coeff=value_coeff if warm_up_value else 0.0,
                bc_fallback_ratio=bc_fallback_ratio,
                bc_weight=1.0, policy_weight=0.0,
            )

            warmup_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            warmup_opt.step()

            bsz = int(gsv.size(0))
            total_examples += bsz
            total_loss += float(loss.item()) * bsz
            pol_loss_acc += float(policy_loss.item()) * bsz
            value_loss_acc += float(value_loss.item()) * bsz
            # Compute training accuracy in eval mode for parity with validation accuracy
            with torch.no_grad():
                was_training = model.training
                model.eval()
                a_pp, t_pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                pred_a = torch.argmax(a_pp, dim=1)
                pred_t = torch.argmax(t_pp, dim=1)
                both = (pred_a == action_idx) & (pred_t == tile_idx)
                correct += int(both.sum().item())
                if was_training:
                    model.train()
            # Show running averages to reduce noise
            denom = max(1, total_examples)
            progress.set_postfix(
                loss=f"{(total_loss/denom):.4f}",
                pol=f"{(pol_loss_acc/denom):.4f}",
                value=f"{(value_loss_acc/denom):.4f}"
            )

        # Calculate train metrics (averaged)
        train_denominator = max(1, total_examples)
        train_acc = (correct / train_denominator)
        train_total_avg = (total_loss / train_denominator)
        train_policy_avg = (pol_loss_acc / train_denominator)
        train_value_avg = (value_loss_acc / train_denominator)

        # Calculate validation accuracy if available
        validation_acc = None
        if dl_val is not None:
            model.eval()
            validation_total = validation_pol = validation_value = 0.0
            validation_count = 0
            validation_correct = 0
            with torch.no_grad():
                for vb in dl_val:
                    (
                        hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages,
                        returns,
                    ) = _prepare_batch_tensors(vb, dev)
                    loss_v, policy_loss_v, value_loss_v, _, _ = _compute_losses(
                        model,
                        hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages,
                        returns,
                        value_coeff=value_coeff,
                        bc_fallback_ratio=bc_fallback_ratio,
                        bc_weight=1.0, policy_weight=0.0,
                    )
                    bsz = int(gsv.size(0))
                    validation_total += float(loss_v.item()) * bsz
                    validation_pol += float(policy_loss_v.item()) * bsz
                    validation_value += float(value_loss_v.item()) * bsz
                    validation_count += bsz
                    a_pp, t_pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                    pred_a = torch.argmax(a_pp, dim=1)
                    pred_t = torch.argmax(t_pp, dim=1)
                    both = (pred_a == action_idx) & (pred_t == tile_idx)
                    validation_correct += int(both.sum().item())
            validation_denominator = max(1, validation_count)
            validation_acc = (validation_correct / max(1, validation_count))
            print(f"\nWarmUp {epoch+1} [validation] - total: {validation_total/validation_denominator:.4f} | policy: {validation_pol/validation_denominator:.4f} | value: {validation_value/validation_denominator:.4f} | acc: {validation_acc:.4f}")
            model.train()

        # Display both train and validation metrics
        if validation_acc is not None:
            print(f"WarmUp {epoch+1} [train] - total: {train_total_avg:.4f} | policy: {train_policy_avg:.4f} | value: {train_value_avg:.4f} | acc: {train_acc:.4f}")
            epoch_acc = validation_acc  # Use validation accuracy for threshold checking
        else:
            print(f"WarmUp {epoch+1} [train] - total: {train_total_avg:.4f} | policy: {train_policy_avg:.4f} | value: {train_value_avg:.4f} | acc: {train_acc:.4f}")
            epoch_acc = train_acc

        if epoch_acc >= threshold:
            reached = True
            print(f"Warm-up threshold met: accuracy {epoch_acc:.4f} >= {threshold:.4f}. Switching to PPO.")
        initial_acc = epoch_acc
        epoch += 1


def _run_ppo_training(
    model: torch.nn.Module,
    dl: DataLoader,
    dl_val: DataLoader | None,
    dev: torch.device,
    opt: torch.optim.Optimizer,
    epochs: int,
    epsilon: float,
    value_coeff: float,
    entropy_coeff: float,
    kl_threshold: float | None,
    patience: int,
    bc_fallback_ratio: float,
) -> str:
    """Run main PPO training loop and return model path."""
    import copy
    import math
    import time
    
    best_loss = math.inf
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    epochs_no_improve = 0
    # Low-memory alternative: keep a single frozen copy of previous epoch's model on CPU
    prev_epoch_model_cpu: torch.nn.Module | None = None
    # Track consecutive epochs where joint KL(prev||curr) <= threshold (when enabled)
    kl_below_count: int = 0
    
    # Track overall BC fallback usage across all training epochs
    overall_bc_batches = 0
    overall_total_batches = 0
    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss_main = 0.0
        progress = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        # Epoch-level debug accumulators
        dbg_ratio_sum = 0.0
        dbg_ratio_sumsq = 0.0
        dbg_adv_sum = 0.0
        dbg_adv_sumsq = 0.0
        dbg_clipped = 0
        dbg_count = 0
        # Epoch-level BC fallback counters
        epoch_bc_batches = 0
        epoch_total_batches = 0
        for bi, batch in enumerate(progress):
            tensors = _prepare_batch_tensors(batch, dev)
            (
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns,
            ) = tensors

            # Print PPO debugging once per epoch at the last training step
            last_step = (bi == (len(dl) - 1))
            total, policy_loss, value_loss, batch_size_curr, dbg = _compute_losses(
                model, hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns,
                epsilon=epsilon,
                value_coeff=value_coeff,
                entropy_coeff=entropy_coeff,
                bc_fallback_ratio=bc_fallback_ratio,
                bc_weight=0.0, policy_weight=1.0,
                print_debug=False,
            )

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(total.item()) * batch_size_curr
            total_policy_loss += float(policy_loss.item()) * batch_size_curr
            total_value_loss += float(value_loss.item()) * batch_size_curr
            total_policy_loss_main += float(policy_loss.item()) * batch_size_curr
            
            total_examples += batch_size_curr
            # Accumulate epoch-level debug stats
            dbg_ratio_sum += dbg['ratio_sum']
            dbg_ratio_sumsq += dbg['ratio_sumsq']
            dbg_adv_sum += dbg['adv_sum']
            dbg_adv_sumsq += dbg['adv_sumsq']
            dbg_clipped += int(dbg['clipped_cnt'])
            dbg_count += int(dbg['bsz'])
            # Count BC fallback usage at batch granularity
            epoch_total_batches += 1
            overall_total_batches += 1
            if bool(dbg.get('bc_fallback_used', False)):
                epoch_bc_batches += 1
                overall_bc_batches += 1
            # Show running averages to reduce noise
            denom = max(1, total_examples)
            progress.set_postfix(
                loss=f"{(total_loss/denom):.4f}",
                pol=f"{(total_policy_loss/denom):.4f}",
                value=f"{(total_value_loss/denom):.4f}"
            )

        # End-of-epoch debug summary (averages over training epoch)
        if dbg_count > 0:
            r_mean = dbg_ratio_sum / dbg_count
            r_var = max(0.0, (dbg_ratio_sumsq / dbg_count) - r_mean * r_mean)
            r_std = math.sqrt(r_var)
            a_mean = dbg_adv_sum / dbg_count
            a_var = max(0.0, (dbg_adv_sumsq / dbg_count) - a_mean * a_mean)
            a_std = math.sqrt(a_var)
            clip_rate = dbg_clipped / max(1, dbg_count)
            print(f"Epoch {epoch + 1}/{epochs} [train dbg] - ratio_mean={r_mean:.4f} | ratio_std={r_std:.4f} | clipped={clip_rate:.3f}")
        # Report BC fallback fraction for this epoch
        if epoch_total_batches > 0:
            frac_bc = epoch_bc_batches / float(epoch_total_batches)
            print(f"Epoch {epoch + 1}/{epochs} [train dbg] - BC fallback batches: {epoch_bc_batches}/{epoch_total_batches} ({frac_bc:.3f})")

        # Evaluate on holdout set
        validation_avg_joint_kl_prev_curr = None
        if dl_val is not None:
            model.eval()
            validation_total = validation_pol = validation_value = 0.0
            validation_pol_main = 0.0
            validation_count = 0
            # Track value loss separately for non-BC batches only
            validation_value_non_bc = 0.0
            validation_count_non_bc = 0
            # Joint KL(prev || curr) across validation set
            validation_joint_kl_prev_curr_sum = 0.0
            with torch.no_grad():
                for vb in dl_val:
                    tensors = _prepare_batch_tensors(vb, dev)
                    (
                        hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                    ) = tensors
                    total_v, policy_loss_v, value_loss_v, bsz, _ = _compute_losses(
                        model, hand, calls, disc, gsv,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                        epsilon=epsilon,
                        value_coeff=value_coeff,
                        entropy_coeff=entropy_coeff,
                        bc_fallback_ratio=bc_fallback_ratio,
                        bc_weight=0.0, policy_weight=1.0,
                    )

                    # Compute current probs
                    a_curr, t_curr, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                    
                    validation_total += float(total_v.item()) * bsz
                    validation_pol += float(policy_loss_v.item()) * bsz
                    validation_value += float(value_loss_v.item()) * bsz
                    validation_pol_main += float(policy_loss_v.item()) * bsz
                    
                    # Check if this is a BC batch (ratio > bc_fallback_ratio) - if not, include in value loss tracking
                    ratio_joint = torch.exp(joint_old_log_probs - (a_curr.log_softmax(dim=1).gather(1, action_idx.unsqueeze(1)).squeeze(1) + 
                                                                   t_curr.log_softmax(dim=1).gather(1, tile_idx.unsqueeze(1)).squeeze(1)))
                    if ratio_joint.max().item() <= bc_fallback_ratio:
                        # Non-BC batch: include in value loss measurement
                        validation_value_non_bc += float(value_loss_v.item()) * bsz
                        validation_count_non_bc += bsz
                    # Low memory mode: use a frozen previous-epoch model on CPU if available
                    if prev_epoch_model_cpu is not None:
                        # Move inputs to CPU for prev model
                        hand_cpu = hand.float().to('cpu')
                        calls_cpu = calls.float().to('cpu')
                        disc_cpu = disc.float().to('cpu')
                        gsv_cpu = gsv.float().to('cpu')
                        a_prev, t_prev, _ = prev_epoch_model_cpu(hand_cpu, calls_cpu, disc_cpu, gsv_cpu)
                        a_prev_safe = a_prev.clamp_min(1e-8)
                        t_prev_safe = t_prev.clamp_min(1e-8)
                        a_curr_safe = a_curr.clamp_min(1e-8).detach().cpu()
                        t_curr_safe = t_curr.clamp_min(1e-8).detach().cpu()
                        kl_a = (a_prev_safe * (a_prev_safe.log() - a_curr_safe.log())).sum(dim=1)
                        kl_t = (t_prev_safe * (t_prev_safe.log() - t_curr_safe.log())).sum(dim=1)
                        kl_joint = kl_a + kl_t
                        validation_joint_kl_prev_curr_sum += float(kl_joint.mean().item()) * bsz

                    validation_count += bsz
            validation_denominator = max(1, validation_count)
            validation_avg = validation_total / validation_denominator
            validation_avg_pol = validation_pol / validation_denominator
            validation_avg_value = validation_value / validation_denominator
            # Calculate value loss average for non-BC batches only
            validation_avg_value_non_bc = validation_value_non_bc / max(1, validation_count_non_bc) if validation_count_non_bc > 0 else None
            # In low memory mode, KL is computed on-the-fly if prev model exists, else N/A for first epoch
            validation_avg_joint_kl_prev_curr = (validation_joint_kl_prev_curr_sum / validation_denominator) if (prev_epoch_model_cpu is not None and validation_denominator > 0) else None
            
            # Print validation results with separate value loss for non-BC batches
            value_loss_display = f"{validation_avg_value:.4f}"
            if validation_avg_value_non_bc is not None:
                value_loss_display += f" (non-BC: {validation_avg_value_non_bc:.4f})"
            
            print(
                f"\nEpoch {epoch + 1}/{epochs} [validation] - total: {validation_avg:.4f} | policy: {validation_avg_pol:.4f} | value: {value_loss_display}"
                + (
                    f" | joint_kl(prev||curr): {validation_avg_joint_kl_prev_curr:.4f}" if validation_avg_joint_kl_prev_curr is not None else " | joint_kl(prev||curr): N/A")
            )
            # Update previous references for next epoch (low memory mode)
            # Keep a frozen copy of the current model on CPU for next epoch KL
            prev_epoch_model_cpu = copy.deepcopy(model).to('cpu').eval()
            model.train()

        # Early stopping
        if dl_val is not None:
            # KL-based early stopping with patience: require joint KL(prev||curr) <= threshold for `patience` consecutive epochs
            if kl_threshold is not None and validation_avg_joint_kl_prev_curr is not None:
                if validation_avg_joint_kl_prev_curr <= float(kl_threshold):
                    kl_below_count += 1
                else:
                    kl_below_count = 0
                if patience and kl_below_count >= int(max(1, patience)):
                    print(
                        f"Early stopping triggered: joint KL(prev||curr) <= {float(kl_threshold):.6f} for {kl_below_count} consecutive epoch(s) (patience={int(patience)})"
                    )
                    break

    # Final overall BC fallback usage summary
    if overall_total_batches > 0:
        overall_frac = overall_bc_batches / float(overall_total_batches)
        print(f"Overall BC fallback batches: {overall_bc_batches}/{overall_total_batches} ({overall_frac:.3f})")
    return timestamp


def train_ppo(
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    value_lr: float | None = None,
    epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    bc_fallback_ratio: float = 5.0,
    device: str | None = None,
    min_delta: float = 1e-4,
    val_split: float = 0.1,
    init_model: str | None = None,
    warm_up_acc: float = 0.0,
    warm_up_max_epochs: int = 50,
    warm_up_value: bool = False,
    *,
    hidden_size: int = 128,
    embedding_dim: int = 16,
    kl_threshold: float | None = 0.008,
    patience: int = 0,
    # low_mem_mode is now always enabled for better memory efficiency
    # Runtime/loader tuning
    os_hint: str | None = None,
    start_method: str | None = None,
    torch_threads: int | None = None,
    interop_threads: int | None = None,
    dl_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
) -> str:
    # Choose device: prefer CUDA, then macOS MPS, else CPU
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dev = torch.device('mps')
        else:
            dev = torch.device('cpu')
    else:
        dev = torch.device(device)

    # Configure multiprocessing and thread settings for platform
    _apply_runtime_config(
        os_hint=os_hint,
        start_method=start_method,
        torch_threads=torch_threads,
        interop_threads=interop_threads,
    )


    if init_model:
        try:
            player = ACPlayer.from_directory(init_model)
            net = player.network.to(dev)
            model = net.torch_module
            ds = ACDataset(
                dataset_path,
                net,
                fit_scaler=False,
                precompute_features=False,
                mmap=True,
            )  # in the future we can always fit scaler to fit new distribution
            print(f"Loaded initial weights from {init_model}")
            '''
            # Evaluate loaded model performance on dataset
            print("\n" + "="*80)
            print("LOADED MODEL EVALUATION")
            print("="*80)
            dl_temp, dl_val_temp = _create_dataloaders(ds, batch_size, val_split, dl_workers, prefetch_factor)
            _evaluate_model_on_combined_dataset(model, dl_temp, dl_val_temp, dev, bc_fallback_ratio, "POST-LOAD")
            print("="*80)
            '''
            
        except Exception as e:
            raise ValueError(f"Warning: failed to load init model '{init_model}': {e}")
    else:
        gsv_scaler = StandardScaler()
        # Initialize AC network first so we can use its feature extraction table and standard scaler for preprocessing
        net = ACNetwork(gsv_scaler=gsv_scaler, hidden_size=hidden_size, embedding_dim=embedding_dim, temperature=0.05)
        net = net.to(dev)
        # Ensure the scaler we persist is the exact one used by the network during preprocessing
        player = ACPlayer(gsv_scaler=gsv_scaler, network=net)

        model = net.torch_module
        ds = ACDataset(
            dataset_path,
            net,
            fit_scaler=True,
            precompute_features=False,
            mmap=True,
        )
    # Build train/validation split by game_ids to avoid leakage
    ds_train, ds_val = ds.train_val_subsets_by_game(val_split)

    # Determine DataLoader worker defaults based on platform if not provided
    resolved = _resolve_loader_defaults(
        os_hint=os_hint,
        dl_workers=dl_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    # Always use conservative settings to minimize RAM (low_mem_mode always enabled)
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
        prefetch_factor=None,
        persistent_workers=False,
    )
    dl_val = (
        DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=resolved['dl_workers'],
            pin_memory=resolved['pin_memory'] and (dev.type in ('cuda', 'mps')),
            prefetch_factor=None,
            persistent_workers=False,
        ) if ds_val is not None else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional value pretraining: optimize value-only for a few epochs before warm-up/PPO
    _run_value_pretraining(model, dl, dl_val, dev, lr, value_lr, bc_fallback_ratio)

    # Optional warm-up: behavior cloning on flat action index + value regression until accuracy threshold
    _run_warmup_bc(model, dl, dl_val, dev, opt, warm_up_acc, warm_up_max_epochs, value_coeff, bc_fallback_ratio, lr, value_lr, warm_up_value)

    # Main PPO training loop
    timestamp = _run_ppo_training(model, dl, dl_val, dev, opt, epochs, epsilon, value_coeff, entropy_coeff, kl_threshold, patience, bc_fallback_ratio)
    '''
    # Evaluate final model performance on combined dataset before saving
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION BEFORE SAVING")
    print("="*80)
    _evaluate_model_on_combined_dataset(model, dl, dl_val, dev, bc_fallback_ratio, "PRE-SAVE")
    '''
    # Save trained weights and scaler
    out_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(out_dir, exist_ok=True)

    # Create a directory for this model
    model_dir = os.path.join(out_dir, f'ac_ppo_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights (state_dict) for portability
    model_path = os.path.join(model_dir, 'model.pt')
    net.save_model(model_path, save_entire_module=False)

    # Save scaler
    assert player.gsv_scaler.mean_[0] > 2 # sanity check that the scaler is fitted correctly
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(player.gsv_scaler, f)

    print(f"Saved PPO model and scaler to {model_dir}")
    return model_path


def main():
    ap = argparse.ArgumentParser(description='Train AC network using PPO on AC dataset (.npz)')
    ap.add_argument('--data', type=str, required=True, help='Path to .npz built by create_dataset.py')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--value_lr', type=float, default=None, help='Learning rate for value pretraining (defaults to --lr if not specified)')
    ap.add_argument('--bc_fallback_ratio', type=float, default=5.0, help='Ratio threshold for falling back to BC mode during PPO training')
    ap.add_argument('--epsilon', type=float, default=0.2)
    ap.add_argument('--value_coeff', type=float, default=0.5)
    ap.add_argument('--entropy_coeff', type=float, default=0.01)
    ap.add_argument('--value_epochs', type=int, default=0, help='Run N epochs of value-only pretraining before warm-up/PPO')
    ap.add_argument('--patience', type=int, default=0, help='Early stopping patience (number of consecutive epochs validation KL(prev||curr) must be <= --kl_threshold to stop; 0 disables)')
    ap.add_argument('--min_delta', type=float, default=1e-4, help='(Legacy/unused) Reserved for potential val-loss patience; currently ignored when using KL-based early stopping')
    ap.add_argument('--kl_threshold', type=float, default=None, help='Early stop when KL(prev||curr) on validation <= this threshold (epoch >= 1)')
    ap.add_argument('--init', type=str, default=None, help='Path to initial AC model weights/module to load')
    ap.add_argument('--warm_up_acc', type=float, default=0.0, help='Accuracy threshold to reach with behavior cloning before switching to PPO (0 disables)')
    ap.add_argument('--warm_up_max_epochs', type=int, default=50, help='Maximum warm-up epochs before switching even if threshold not reached')
    ap.add_argument('--warm_up_value', action='store_true', help='Enable value network training during warm-up BC phase')
    ap.add_argument('--hidden_size', type=int, default=128, help='Hidden size for ACNetwork')
    ap.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for ACNetwork')
    # DataLoader tuning
    ap.add_argument('--dl_workers', type=int, default=None, help='Number of DataLoader workers (overrides platform defaults)')
    ap.add_argument('--prefetch_factor', type=int, default=None, help='DataLoader prefetch_factor when workers>0 (overrides platform defaults)')
    args = ap.parse_args()

    # Pass value_epochs via a temporary attribute to avoid altering the function signature in many places
    setattr(train_ppo, '_value_epochs_override', int(args.value_epochs))
    train_ppo(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_lr=args.value_lr,
        epsilon=args.epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        bc_fallback_ratio=args.bc_fallback_ratio,
        min_delta=float(args.min_delta),
        init_model=args.init,
        warm_up_acc=args.warm_up_acc,
        warm_up_max_epochs=args.warm_up_max_epochs,
        warm_up_value=args.warm_up_value,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        kl_threshold=args.kl_threshold,
        patience=args.patience,
        dl_workers=args.dl_workers,
        prefetch_factor=args.prefetch_factor,
    )


if __name__ == '__main__':
    main()


