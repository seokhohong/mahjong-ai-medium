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
from core.learn.ac_constants import ACTION_HEAD_SIZE, GAME_STATE_VEC_LEN
from core.tile import UNIQUE_TILE_COUNT
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
    def __init__(self, npz_path: str, net: ACNetwork, fit_scaler: bool = True, *, mmap: bool = True):
        """Dataset backed by a memory-mapped .npz created by build_ac_dataset.

        Only stores a reference to the data file handle and computes features on demand.
        """
        # Keep only the handle to the npz to avoid loading everything in memory
        self._data = np.load(npz_path, allow_pickle=True, mmap_mode=('r' if mmap else None))

        # Cache references to arrays we access in __getitem__ (still memmap-backed if mmap=True)
        d = self._data
        # Feature inputs
        self.hand_idx = d['hand_idx']
        self.called_idx = d['called_idx']
        self.disc_idx = d['disc_idx']
        self.remaining_tiles = d['remaining_tiles']
        self.owner_of_reactable_tile = d['owner_of_reactable_tile']
        self.seat_winds = d['seat_winds']
        self.riichi_declarations = d['riichi_declarations']
        self.legal_action_mask = d['legal_action_mask']
        self.dora_indicator_tiles = d['dora_indicator_tiles']
        self.reactable_tile = d['reactable_tile']
        # Targets and aux
        self.action_idx = d['action_idx']
        self.tile_idx = d['tile_idx']
        self.returns = d['returns']
        self.advantages = d['advantages']
        self.joint_log_probs = d['joint_log_probs']
        self.game_ids = d['game_ids']
        self.wall_count = d['wall_count']
        self.deal_in_tiles = d['deal_in_tiles']

        # Optionally fit the scaler on remaining_tiles (column 0 of GSV)
        if fit_scaler:
            remaining_tiles_arr = np.asarray(self.remaining_tiles, dtype=np.float32).reshape(-1, 1)
            net.fit_scaler(remaining_tiles_arr)

        self._net = net

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

        # Use cached game_ids
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
        ds_validation = Subset(self, val_indices) if len(val_indices) > 0 else None
        return ds_train, ds_validation

    def __len__(self) -> int:
        # Use length of the primary arrays (action_idx)
        return int(len(self.action_idx))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Compute features on demand to save RAM
        # Build game_tile_indicators_idx = [react, d1, d2, d3, d4]
        react_idx = int(self.reactable_tile[idx]) if self.reactable_tile is not None else -1
        dora = np.asarray(self.dora_indicator_tiles[idx], dtype=np.int32)
        dora_vals = dora.tolist()
        while len(dora_vals) < 4:
            dora_vals.append(-1)
        game_tile_indicators_idx = np.asarray([react_idx] + dora_vals[:4], dtype=np.int32)

        (
            player_hand_idx_fixed,
            player_called_idx,
            opp_called_idx,
            disc_idx,
        ) = self._net.extract_features_from_indexed(
            hand_idx=np.asarray(self.hand_idx[idx], dtype=np.int32),
            called_idx=np.asarray(self.called_idx[idx], dtype=np.int32),
            disc_idx=np.asarray(self.disc_idx[idx], dtype=np.int32),
            game_tile_indicators=game_tile_indicators_idx,
        )

        # Construct game_state_vec to length GAME_STATE_VEC_LEN
        gsv = np.zeros((int(GAME_STATE_VEC_LEN),), dtype=np.float32)
        # [0] remaining_tiles (to be standardized later)
        gsv[0] = float(int(self.remaining_tiles[idx]))
        # [1] owner_of_reactable_tile
        gsv[1] = float(int(self.owner_of_reactable_tile[idx]))
        # [2..5] seat_winds[4]
        sw = np.asarray(self.seat_winds[idx], dtype=np.int32)
        for i in range(4):
            gsv[2 + i] = float(int(sw[i]) if i < sw.shape[0] else 0)
        # [6..9] riichi_declarations[4]
        rd = np.asarray(self.riichi_declarations[idx], dtype=np.int32)
        for i in range(4):
            gsv[6 + i] = float(int(rd[i]) if i < rd.shape[0] else -1)
        # [10..] legal_action_mask[ACTION_HEAD_SIZE]
        lam = np.asarray(self.legal_action_mask[idx], dtype=np.int32)
        lam_fixed = np.zeros((int(ACTION_HEAD_SIZE),), dtype=np.float32)
        k = min(int(ACTION_HEAD_SIZE), int(lam.shape[0]))
        lam_fixed[:k] = lam[:k]
        gsv[10:10 + int(ACTION_HEAD_SIZE)] = lam_fixed

        # Standardize remaining_tiles column only
        gsv_std = gsv.copy()
        try:
            col0_scaled = self._net._gsv_scaler.transform(gsv_std[0:1].reshape(-1, 1)).astype(np.float32)[0, 0]
            gsv_std[0] = float(col0_scaled)
        except Exception:
            gsv_std[0] = 0.0

        # Build deal_in_mask (37-dim multi-hot from deal_in_tiles), and passthrough wall_count
        if self.deal_in_tiles is not None:
            tiles = np.asarray(self.deal_in_tiles[idx], dtype=np.int32).ravel()
            deal_in_mask = np.zeros((int(UNIQUE_TILE_COUNT),), dtype=np.float32)
            for t in tiles:
                ti = int(t)
                if 0 <= ti < int(UNIQUE_TILE_COUNT):
                    deal_in_mask[ti] = 1.0
        else:
            deal_in_mask = np.zeros((int(UNIQUE_TILE_COUNT),), dtype=np.float32)
        wall_count = self.wall_count[idx]

        return {
            'player_hand_idx': np.asarray(player_hand_idx_fixed, dtype=np.int32),
            'player_called_idx': np.asarray(player_called_idx, dtype=np.int32),
            'opp_called_idx': opp_called_idx.astype(np.int32),
            'disc_idx': disc_idx.astype(np.int32),
            'game_tile_indicators_idx': game_tile_indicators_idx.astype(np.int32),
            'game_state_vec': gsv_std.astype(np.float32),
            'action_idx': int(self.action_idx[idx]),
            'tile_idx': int(self.tile_idx[idx]),
            'return': float(self.returns[idx]),
            'advantage': float(self.advantages[idx]),
            'joint_old_log_prob': float(self.joint_log_probs[idx]),
            # Aux targets
            'wall_count': wall_count,
            'deal_in_mask': deal_in_mask,
        }


 

def _prepare_batch_tensors(batch: Dict[str, Any], dev: torch.device):
    def _to_dev(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    player_hand_idx = torch.from_numpy(np.stack(batch['player_hand_idx'])).to(dev)
    player_called_idx = torch.from_numpy(np.stack(batch['player_called_idx'])).to(dev)
    opp_called_idx = torch.from_numpy(np.stack(batch['opp_called_idx'])).to(dev)
    disc_idx = torch.from_numpy(np.stack(batch['disc_idx'])).to(dev)
    game_tile_indicators_idx = torch.from_numpy(np.stack(batch['game_tile_indicators_idx'])).to(dev)
    game_state_vec = torch.from_numpy(np.stack(batch['game_state_vec'])).to(dev)

    action_idx = _to_dev(batch['action_idx'], torch.long)
    tile_idx = _to_dev(batch['tile_idx'], torch.long)
    joint_old_log_probs = _to_dev(batch['joint_old_log_prob'], torch.float32)
    advantages = _to_dev(batch['advantage'], torch.float32)
    returns = _to_dev(batch['return'], torch.float32)
    # Aux targets
    wall_count = _to_dev(batch['wall_count'], torch.float32)
    deal_in_mask = _to_dev(batch['deal_in_mask'], torch.float32)

    return (
        player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
        action_idx, tile_idx,
        joint_old_log_probs,
        advantages, returns,
        wall_count, deal_in_mask,
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

def _evaluate_model_on_combined_dataset(model: torch.nn.Module, dl_train, dl_validation, dev: torch.device, bc_fallback_ratio: float, description: str = ""):
    """Evaluate model performance on the combined train+validation dataset with weighted averaging."""
    model.eval()
    
    # Combine datasets with proper weighting
    datasets = []
    weights = []
    
    if dl_train is not None:
        datasets.append(('train', dl_train))
        weights.append(len(dl_train.dataset))
    
    if dl_validation is not None:
        datasets.append(('validation', dl_validation))
        weights.append(len(dl_validation.dataset))
    
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
                (player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                 action_idx, tile_idx, joint_old_log_probs, advantages, returns, wall_count, deal_in_mask) = _prepare_batch_tensors(batch, dev)
                
                # Compute losses
                total_v, policy_loss_v, value_loss_v, bsz, _ = _compute_losses(
                    model,
                    player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                    action_idx, tile_idx, joint_old_log_probs, advantages, returns,
                    wall_count, deal_in_mask,
                    bc_fallback_ratio=bc_fallback_ratio,
                    bc_weight=0.0, policy_weight=1.0,
                )
                
                # Compute accuracy
                a_logits, t_logits, _, _, _ = model(
                    player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
                )
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
        player_hand_idx: torch.Tensor,
        player_called_idx: torch.Tensor,
        opp_called_idx: torch.Tensor,
        disc_idx: torch.Tensor,
        game_tile_indicators_idx: torch.Tensor,
        game_state_vec: torch.Tensor,
        action_idx: torch.Tensor,
        tile_idx: torch.Tensor,
        joint_old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        wall_count: torch.Tensor,
        deal_in_mask: torch.Tensor,
        *,
        epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        bc_fallback_ratio: float = 5.0,
        bc_weight: float = 0.0,
        policy_weight: float = 1.0,
        aux_wall_coeff: float = 0.1,
        aux_deal_in_coeff: float = 0.1,
        print_debug: bool = False,
):
    a_pp, t_pp, value_pred, wall_counts_pred, deal_in_pred = model(
        player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
    )
    value_pred = value_pred.squeeze(1)
    batch_size_curr = int(player_hand_idx.size(0))

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
        policy_loss = torch.zeros((), device=value_pred.device, dtype=value_pred.dtype)
        value_loss = F.smooth_l1_loss(value_pred, returns)
        # Aux losses also contribute during value-only pretraining
        wall_loss = F.mse_loss(wall_counts_pred, wall_count)
        deal_in_loss = F.binary_cross_entropy(deal_in_pred.clamp(1e-6, 1-1e-6), deal_in_mask)
        total = value_loss + aux_wall_coeff * wall_loss + aux_deal_in_coeff * deal_in_loss
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
        return total, policy_loss, value_loss, wall_loss, deal_in_loss, batch_size_curr, dbg

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
    value_loss = F.smooth_l1_loss(value_pred, returns)
    # Aux terms
    wall_loss = F.smooth_l1_loss(wall_counts_pred, wall_count)
    deal_in_loss = F.binary_cross_entropy(deal_in_pred.clamp(1e-6, 1-1e-6), deal_in_mask)
    entropy_a = safe_entropy_calculation(a_pp)
    entropy_t = safe_entropy_calculation(t_pp)
    entropy = entropy_a + entropy_t

    total = policy_loss + value_coeff * value_loss + aux_wall_coeff * wall_loss + aux_deal_in_coeff * deal_in_loss - entropy_coeff * entropy

    return total, policy_loss, value_loss, wall_loss, deal_in_loss, batch_size_curr, dbg


def _run_warmup_bc(
    model: torch.nn.Module,
    dl: DataLoader,
    dl_validation: DataLoader | None,
    dev: torch.device,
    opt: torch.optim.Optimizer,
    warm_up_acc: float,
    warm_up_max_epochs: int,
    value_coeff: float,
    bc_fallback_ratio: float,
    lr: float,
) -> None:
    """Run behavior cloning warm-up until accuracy threshold is met."""
    if not warm_up_acc or warm_up_acc <= 0.0:
        return
    
    # Use main lr for warm-up
    warmup_lr = lr
    warmup_opt = torch.optim.Adam(model.parameters(), lr=warmup_lr)
    print(f"Warm-up BC using learning rate: {warmup_lr}")
    
    def _eval_policy_accuracy(dloader: DataLoader) -> float:
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in dloader:
                (
                    player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                    action_idx, tile_idx,
                    _joint_old_log_probs,
                    _advantages,
                    _returns,
                    _wall_count, _deal_in_mask,
                ) = _prepare_batch_tensors(batch, dev)
                a_pp, t_pp, _, _, _ = model(
                    player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
                )
                pred_a = torch.argmax(a_pp, dim=1)
                pred_t = torch.argmax(t_pp, dim=1)
                both = (pred_a == action_idx) & (pred_t == tile_idx)
                correct += int(both.sum().item())
                count += int(player_hand_idx.size(0))
        model.train()
        return float(correct) / float(max(1, count))
    
    threshold = float(max(0.0, min(1.0, warm_up_acc)))
    # Pre-check: if current model already meets threshold on validation (or train) accuracy, skip warm-up
    initial_acc = _eval_policy_accuracy(dl_validation if dl_validation is not None else dl)
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
        wall_loss_acc = 0.0
        deal_in_loss_acc = 0.0
        correct = 0
        progress = tqdm(dl, desc=f"WarmUp {epoch+1}", leave=False)
        for batch in progress:
            (
                player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages,
                returns,
                wall_count, deal_in_mask,
            ) = _prepare_batch_tensors(batch, dev)

            loss, policy_loss, value_loss, wall_loss, deal_in_loss, _, _dbg = _compute_losses(
                model,
                player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages,
                returns,
                wall_count, deal_in_mask,
                value_coeff=value_coeff,
                entropy_coeff=0.0,  # no entropy regularization during warm-up
                bc_fallback_ratio=bc_fallback_ratio,
                bc_weight=1.0, policy_weight=0.0,
            )

            warmup_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            warmup_opt.step()

            bsz = int(player_hand_idx.size(0))
            total_examples += bsz
            total_loss += float(loss.item()) * bsz
            pol_loss_acc += float(policy_loss.item()) * bsz
            value_loss_acc += float(value_loss.item()) * bsz
            wall_loss_acc += float(wall_loss.item()) * bsz
            deal_in_loss_acc += float(deal_in_loss.item()) * bsz
            # Compute training accuracy in eval mode for parity with validation accuracy
            with torch.no_grad():
                was_training = model.training
                model.eval()
                a_pp, t_pp, _, _, _ = model(
                    player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
                )
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
                value=f"{(value_loss_acc/denom):.4f}",
                wall=f"{(wall_loss_acc/denom):.4f}",
                deal_in=f"{(deal_in_loss_acc/denom):.4f}",
            )

        # Calculate train metrics (averaged)
        train_denominator = max(1, total_examples)
        train_acc = (correct / train_denominator)
        train_total_avg = (total_loss / train_denominator)
        train_policy_avg = (pol_loss_acc / train_denominator)
        train_value_avg = (value_loss_acc / train_denominator)
        train_wall_avg = (wall_loss_acc / train_denominator)
        train_deal_in_avg = (deal_in_loss_acc / train_denominator)

        # Calculate validation accuracy if available
        validation_acc = None
        if dl_validation is not None:
            model.eval()
            validation_total = validation_pol = validation_value = 0.0
            validation_wall = validation_deal_in = 0.0
            validation_count = 0
            validation_correct = 0
            with torch.no_grad():
                for vb in dl_validation:
                    (
                        player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages,
                        returns,
                        wall_count, deal_in_mask,
                    ) = _prepare_batch_tensors(vb, dev)
                    loss_v, policy_loss_v, value_loss_v, wall_loss_v, deal_in_loss_v, _, _ = _compute_losses(
                        model,
                        player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages,
                        returns,
                        wall_count, deal_in_mask,
                        value_coeff=value_coeff,
                        entropy_coeff=0.0,  # no entropy regularization during warm-up validation
                        bc_fallback_ratio=bc_fallback_ratio,
                        bc_weight=1.0, policy_weight=0.0,
                    )
                    bsz = int(player_hand_idx.size(0))
                    validation_total += float(loss_v.item()) * bsz
                    validation_pol += float(policy_loss_v.item()) * bsz
                    validation_value += float(value_loss_v.item()) * bsz
                    validation_count += bsz
                    validation_wall += float(wall_loss_v.item()) * bsz
                    validation_deal_in += float(deal_in_loss_v.item()) * bsz
                    a_pp, t_pp, _, _, _ = model(
                        player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
                    )
                    pred_a = torch.argmax(a_pp, dim=1)
                    pred_t = torch.argmax(t_pp, dim=1)
                    both = (pred_a == action_idx) & (pred_t == tile_idx)
                    validation_correct += int(both.sum().item())
            validation_denominator = max(1, validation_count)
            validation_acc = (validation_correct / max(1, validation_count))
            print(f"\nWarmUp {epoch+1} [validation] - total: {validation_total/validation_denominator:.4f} | policy: {validation_pol/validation_denominator:.4f} | value: {validation_value/validation_denominator:.4f} | wall: {validation_wall/validation_denominator:.4f} | deal_in: {validation_deal_in/validation_denominator:.4f} | acc: {validation_acc:.4f}")
            model.train()

        # Display both train and validation metrics
        if validation_acc is not None:
            print(f"WarmUp {epoch+1} [train] - total: {train_total_avg:.4f} | policy: {train_policy_avg:.4f} | value: {train_value_avg:.4f} | wall: {train_wall_avg:.4f} | deal_in: {train_deal_in_avg:.4f} | acc: {train_acc:.4f}")
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
    dl_validation: DataLoader | None,
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
            (
                player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns, wall_count, deal_in_mask
            ) = _prepare_batch_tensors(batch, dev)

            # Print PPO debugging once per epoch at the last training step
            last_step = (bi == (len(dl) - 1))
            total, policy_loss, value_loss, wall_loss, deal_in_loss, batch_size_curr, dbg = _compute_losses(
                model,
                player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                action_idx, tile_idx,
                joint_old_log_probs,
                advantages, returns,
                wall_count, deal_in_mask,
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
            total_wall_loss = locals().get('total_wall_loss', 0.0) + float(wall_loss.item()) * batch_size_curr
            total_deal_in_loss = locals().get('total_deal_in_loss', 0.0) + float(deal_in_loss.item()) * batch_size_curr
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
                value=f"{(total_value_loss/denom):.4f}",
                wall=f"{(locals().get('total_wall_loss',0.0)/denom):.4f}",
                deal_in=f"{(locals().get('total_deal_in_loss',0.0)/denom):.4f}",
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
        if dl_validation is not None:
            model.eval()
            validation_total = validation_pol = validation_value = 0.0
            validation_wall = validation_deal_in = 0.0
            validation_pol_main = 0.0
            validation_count = 0
            # Track value loss separately for non-BC batches only
            validation_value_non_bc = 0.0
            validation_count_non_bc = 0
            # Joint KL(prev || curr) across validation set
            validation_joint_kl_prev_curr_sum = 0.0
            with torch.no_grad():
                for vb in dl_validation:
                    (
                        player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                        wall_count, deal_in_mask,
                    ) = _prepare_batch_tensors(vb, dev)
                    total_v, policy_loss_v, value_loss_v, wall_loss_v, deal_in_loss_v, bsz, _ = _compute_losses(
                        model,
                        player_hand_idx, player_called_idx, opp_called_idx, disc_idx, game_tile_indicators_idx, game_state_vec,
                        action_idx, tile_idx,
                        joint_old_log_probs,
                        advantages, returns,
                        wall_count, deal_in_mask,
                        epsilon=epsilon,
                        value_coeff=value_coeff,
                        entropy_coeff=entropy_coeff,
                        bc_fallback_ratio=bc_fallback_ratio,
                        bc_weight=0.0, policy_weight=1.0,
                    )

                    # Compute current probs
                    a_curr, t_curr, _, _, _ = model(
                        player_hand_idx.long(), player_called_idx.long(), opp_called_idx.long(), disc_idx.long(), game_tile_indicators_idx.long(), game_state_vec.float(),
                    )
                    
                    validation_total += float(total_v.item()) * bsz
                    validation_pol += float(policy_loss_v.item()) * bsz
                    validation_value += float(value_loss_v.item()) * bsz
                    validation_pol_main += float(policy_loss_v.item()) * bsz
                    validation_wall += float(wall_loss_v.item()) * bsz
                    validation_deal_in += float(deal_in_loss_v.item()) * bsz
                    
                    # Check if this is a BC batch (ratio > bc_fallback_ratio) - if not, include in value loss tracking
                    # joint_old_log_probs are from dataset; compute current joint log-prob from probabilities
                    curr_logp_a = a_curr.clamp_min(1e-8).log().gather(1, action_idx.view(-1,1)).squeeze(1)
                    curr_logp_t = t_curr.clamp_min(1e-8).log().gather(1, tile_idx.view(-1,1)).squeeze(1)
                    ratio_joint = torch.exp(curr_logp_a + curr_logp_t - joint_old_log_probs)
                    if ratio_joint.max().item() <= bc_fallback_ratio:
                        # Non-BC batch: include in value loss measurement
                        validation_value_non_bc += float(value_loss_v.item()) * bsz
                        validation_count_non_bc += bsz
                    # Low memory mode: use a frozen previous-epoch model on CPU if available
                    if prev_epoch_model_cpu is not None:
                        # Move inputs to CPU for prev model
                        player_hand_idx_cpu = player_hand_idx.long().to('cpu')
                        player_called_idx_cpu = player_called_idx.long().to('cpu')
                        opp_called_idx_cpu = opp_called_idx.long().to('cpu')
                        disc_idx_cpu = disc_idx.long().to('cpu')
                        gti_cpu = game_tile_indicators_idx.long().to('cpu')
                        gsv_cpu = game_state_vec.float().to('cpu')
                        a_prev, t_prev, _, _, _ = prev_epoch_model_cpu(
                            player_hand_idx_cpu, player_called_idx_cpu, opp_called_idx_cpu, disc_idx_cpu, gti_cpu, gsv_cpu,
                        )
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
            validation_avg_wall = validation_wall / validation_denominator
            validation_avg_deal_in = validation_deal_in / validation_denominator
            # Calculate value loss average for non-BC batches only
            validation_avg_value_non_bc = validation_value_non_bc / max(1, validation_count_non_bc) if validation_count_non_bc > 0 else None
            # In low memory mode, KL is computed on-the-fly if prev model exists, else N/A for first epoch
            validation_avg_joint_kl_prev_curr = (validation_joint_kl_prev_curr_sum / validation_denominator) if (prev_epoch_model_cpu is not None and validation_denominator > 0) else None
            
            # Print validation results with separate value loss for non-BC batches
            value_loss_display = f"{validation_avg_value:.4f}"
            if validation_avg_value_non_bc is not None:
                value_loss_display += f" (non-BC: {validation_avg_value_non_bc:.4f})"
            
            print(
                f"\nEpoch {epoch + 1}/{epochs} [validation] - total: {validation_avg:.4f} | policy: {validation_avg_pol:.4f} | value: {value_loss_display} | wall: {validation_avg_wall:.4f} | deal_in: {validation_avg_deal_in:.4f}"
                + (
                    f" | joint_kl(prev||curr): {validation_avg_joint_kl_prev_curr:.4f}" if validation_avg_joint_kl_prev_curr is not None else " | joint_kl(prev||curr): N/A")
            )
            # Update previous references for next epoch (low memory mode)
            # Keep a frozen copy of the current model on CPU for next epoch KL
            prev_epoch_model_cpu = copy.deepcopy(model).to('cpu').eval()
            model.train()

        # Early stopping
        if dl_validation is not None:
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
            model = net.torch_module()
            ds = ACDataset(
                dataset_path,
                net,
                fit_scaler=False,
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
        net = ACNetwork(gsv_scaler=gsv_scaler, hidden_size=hidden_size, embedding_dim=embedding_dim)
        net = net.to(dev)
        # Ensure the scaler we persist is the exact one used by the network during preprocessing
        player = ACPlayer(gsv_scaler=gsv_scaler, network=net)

        model = net.torch_module()
        ds = ACDataset(
            dataset_path,
            net,
            fit_scaler=True,
            mmap=True,
        )
    # Build train/validation split by game_ids to avoid leakage
    ds_train, ds_validation = ds.train_val_subsets_by_game(val_split)

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
    dl_validation = (
        DataLoader(
            ds_validation,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=resolved['dl_workers'],
            pin_memory=resolved['pin_memory'] and (dev.type in ('cuda', 'mps')),
            prefetch_factor=None,
            persistent_workers=False,
        ) if ds_validation is not None else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional warm-up: behavior cloning on flat action index + value regression until accuracy threshold
    _run_warmup_bc(model, dl, dl_validation, dev, opt, warm_up_acc, warm_up_max_epochs, value_coeff, bc_fallback_ratio, lr)

    # Main PPO training loop
    timestamp = _run_ppo_training(model, dl, dl_validation, dev, opt, epochs, epsilon, value_coeff, entropy_coeff, kl_threshold, patience, bc_fallback_ratio)
    '''
    # Evaluate final model performance on combined dataset before saving
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION BEFORE SAVING")
    print("="*80)
    _evaluate_model_on_combined_dataset(model, dl, dl_validation, dev, bc_fallback_ratio, "PRE-SAVE")
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
    ap.add_argument('--bc_fallback_ratio', type=float, default=5.0, help='Ratio threshold for falling back to BC mode during PPO training')
    ap.add_argument('--epsilon', type=float, default=0.2)
    ap.add_argument('--value_coeff', type=float, default=0.5)
    ap.add_argument('--entropy_coeff', type=float, default=0.01)
    ap.add_argument('--patience', type=int, default=0, help='Early stopping patience (number of consecutive epochs validation KL(prev||curr) must be <= --kl_threshold to stop; 0 disables)')
    ap.add_argument('--min_delta', type=float, default=1e-4, help='(Legacy/unused) Reserved for potential val-loss patience; currently ignored when using KL-based early stopping')
    ap.add_argument('--kl_threshold', type=float, default=None, help='Early stop when KL(prev||curr) on validation <= this threshold (epoch >= 1)')
    ap.add_argument('--init', type=str, default=None, help='Path to initial AC model weights/module to load')
    ap.add_argument('--warm_up_acc', type=float, default=0.0, help='Accuracy threshold to reach with behavior cloning before switching to PPO (0 disables)')
    ap.add_argument('--warm_up_max_epochs', type=int, default=50, help='Maximum warm-up epochs before switching even if threshold not reached')
    ap.add_argument('--hidden_size', type=int, default=128, help='Hidden size for ACNetwork')
    ap.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for ACNetwork')
    # DataLoader tuning
    ap.add_argument('--dl_workers', type=int, default=None, help='Number of DataLoader workers (overrides platform defaults)')
    ap.add_argument('--prefetch_factor', type=int, default=None, help='DataLoader prefetch_factor when workers>0 (overrides platform defaults)')
    args = ap.parse_args()

    train_ppo(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        bc_fallback_ratio=args.bc_fallback_ratio,
        min_delta=float(args.min_delta),
        init_model=args.init,
        warm_up_acc=args.warm_up_acc,
        warm_up_max_epochs=args.warm_up_max_epochs,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        kl_threshold=args.kl_threshold,
        patience=args.patience,
        dl_workers=args.dl_workers,
        prefetch_factor=args.prefetch_factor,
    )


if __name__ == '__main__':
    main()


