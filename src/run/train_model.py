#!/usr/bin/env python3
from __future__ import annotations

import os
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

# not using called discards yet
class ACDataset(Dataset):
    def __init__(self, npz_path: str, net: ACNetwork, fit_scaler = True):
        data = np.load(npz_path, allow_pickle=True)
        # New compact format fields
        hand_arr = data['hand_idx']
        disc_arr = data['disc_idx']
        called_arr = data['called_idx']
        gsv_arr = data['game_state']
        self.action_idx = data['action_idx']
        self.tile_idx = data['tile_idx']
        # Load returns/advantages
        returns = data['returns'].astype(np.float32)
        advantages = data['advantages'].astype(np.float32)
        self.returns = returns
        self.advantages = advantages
        if 'joint_log_probs' in data.files:
            self.joint_old_log_probs = data['joint_log_probs'].astype(np.float32)
        else:
            # Backward compatibility with datasets that stored separate head log-probs
            a_lp = data['action_log_probs'].astype(np.float32)
            t_lp = data['tile_log_probs'].astype(np.float32)
            self.joint_old_log_probs = (a_lp + t_lp).astype(np.float32)

        # fit the standard scaler so we can properly use extract_features
        if fit_scaler:
            net.fit_scaler(gsv_arr)

        # Pre-extract indexed states and transform to embedded sequences once
        hand_list = []
        calls_list = []
        disc_list = []
        gsv_list = []
        N = len(hand_arr)
        for i in range(N):
            h, c, d, g = net.extract_features_from_indexed(
                np.asarray(hand_arr[i], dtype=np.int32),
                np.asarray(disc_arr[i], dtype=np.int32),
                np.asarray(called_arr[i], dtype=np.int32),
                np.asarray(gsv_arr[i], dtype=np.float32),
            )
            hand_list.append(h.astype(np.float32))
            calls_list.append(c.astype(np.float32))
            disc_list.append(d.astype(np.float32))
            gsv_list.append(g.astype(np.float32))
        self.hand = np.asarray(hand_list, dtype=np.float32)
        self.calls = np.asarray(calls_list, dtype=np.float32)
        self.disc = np.asarray(disc_list, dtype=np.float32)
        self.gsv = np.asarray(gsv_list, dtype=np.float32)

    def __len__(self) -> int:
        return self.hand.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'hand': self.hand[idx],
            'calls': self.calls[idx],
            'disc': self.disc[idx],
            'gsv': self.gsv[idx],
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
    hand = torch.from_numpy(np.stack(batch['hand'])).to(dev)
    calls = torch.from_numpy(np.stack(batch['calls'])).to(dev)
    disc = torch.from_numpy(np.stack(batch['disc'])).to(dev)
    gsv = torch.from_numpy(np.stack(batch['gsv'])).to(dev)
    action_idx = torch.tensor(batch['action_idx'], dtype=torch.long, device=dev)
    tile_idx = torch.tensor(batch['tile_idx'], dtype=torch.long, device=dev)
    joint_old_log_probs = torch.tensor(batch['joint_old_log_prob'], dtype=torch.float32, device=dev)
    advantages = torch.tensor(batch['advantage'], dtype=torch.float32, device=dev)
    returns = torch.tensor(batch['return'], dtype=torch.float32, device=dev)
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
        mode: str = 'ppo',
        epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
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
    dbg = {
        'bsz': batch_size_curr,
        'ratio_sum': float(ratio_joint.sum().item()),
        'ratio_sumsq': float((ratio_joint * ratio_joint).sum().item()),
        'adv_sum': float(advantages.sum().item()),
        'adv_sumsq': float((advantages * advantages).sum().item()),
        'clipped_cnt': int(((ratio_joint < (1 - epsilon)) | (ratio_joint > (1 + epsilon))).sum().item()),
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

    # Check for divergence earlier and more strictly based on joint ratio
    if mode == 'bc' or ratio_joint.max().item() > 5.0:
        if mode == 'ppo' and print_debug:
            print(f"Switching to BC mode due to high joint ratio: {ratio_joint.max().item():.2f}")

        log_a = (a_pp.clamp_min(1e-8)).log()
        log_t = (t_pp.clamp_min(1e-8)).log()
        policy_loss_a = F.nll_loss(log_a, action_idx.to(dtype=torch.long, device=a_pp.device))
        policy_loss_t = F.nll_loss(log_t, tile_idx.to(dtype=torch.long, device=t_pp.device))
        policy_loss = policy_loss_a + policy_loss_t

        # Add entropy even in BC mode to maintain exploration
        entropy_a = safe_entropy_calculation(a_pp)
        entropy_t = safe_entropy_calculation(t_pp)
        entropy = entropy_a + entropy_t

        total = policy_loss - entropy_coeff * entropy
        return total, policy_loss, torch.zeros_like(policy_loss), batch_size_curr, dbg

    clipped_ratio = torch.clamp(ratio_joint, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(
        ratio_joint * advantages,
        clipped_ratio * advantages
    ).mean()

    value_loss = F.mse_loss(val, returns)
    entropy_a = safe_entropy_calculation(a_pp)
    entropy_t = safe_entropy_calculation(t_pp)
    entropy = entropy_a + entropy_t

    total = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    return total, policy_loss, value_loss, batch_size_curr, dbg


def train_ppo(
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    device: str | None = None,
    min_delta: float = 1e-4,
    val_split: float = 0.1,
    init_model: str | None = None,
    warm_up_acc: float = 0.0,
    warm_up_max_epochs: int = 50,
    *,
    hidden_size: int = 128,
    embedding_dim: int = 16,
    kl_threshold: float | None = 0.008,
    patience: int = 0,
) -> str:
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


    if init_model:
        try:
            player = ACPlayer.from_directory(init_model)
            net = player.network
            model = net.torch_module
            ds = ACDataset(dataset_path, net, fit_scaler=False) # in the future we can always fit scaler to fit new distribution
            print(f"Loaded initial weights from {init_model}")
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
        ds = ACDataset(dataset_path, net, fit_scaler=True)
    # Build train/validation split
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
    dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False) if ds_val is not None else None
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Helper: compute top-1 policy accuracy on a given dataloader
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

    # Optional warm-up: behavior cloning on flat action index + value regression until accuracy threshold
    if warm_up_acc and warm_up_acc > 0.0:
        threshold = float(max(0.0, min(1.0, warm_up_acc)))
        # Pre-check: if current model already meets threshold on validation (or train) accuracy, skip warm-up
        initial_acc = _eval_policy_accuracy(dl_val if dl_val is not None else dl)
        if initial_acc >= threshold:
            print(f"Skipping warm-up: initial accuracy {initial_acc:.4f} >= {threshold:.4f}")
        else:
            print(f"Starting warm-up until accuracy >= {threshold:.2f} (behavior cloning + value regression)...")
        epoch = 0
        reached = False
        while initial_acc < threshold and epoch < int(max(1, warm_up_max_epochs)) and not reached:
            total_loss = 0.0
            total_examples = 0
            pol_loss_acc = 0.0
            val_loss_acc = 0.0
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
                    mode='bc',
                    value_coeff=value_coeff,
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                bsz = int(gsv.size(0))
                total_examples += bsz
                total_loss += float(loss.item()) * bsz
                pol_loss_acc += float(policy_loss.item()) * bsz
                val_loss_acc += float(value_loss.item()) * bsz
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
                progress.set_postfix(loss=f"{float(loss.item()):.4f}", pol=f"{float(policy_loss.item()):.4f}", val=f"{float(value_loss.item()):.4f}")

            # Calculate train accuracy
            train_acc = (correct / max(1, total_examples))

            # Calculate validation accuracy if available
            val_acc = None
            if dl_val is not None:
                model.eval()
                v_total = v_pol = v_val = 0.0
                v_count = 0
                v_correct = 0
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
                            mode='bc',
                            value_coeff=value_coeff,
                        )
                        bsz = int(gsv.size(0))
                        v_total += float(loss_v.item()) * bsz
                        v_pol += float(policy_loss_v.item()) * bsz
                        v_val += float(value_loss_v.item()) * bsz
                        v_count += bsz
                        a_pp, t_pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                        pred_a = torch.argmax(a_pp, dim=1)
                        pred_t = torch.argmax(t_pp, dim=1)
                        both = (pred_a == action_idx) & (pred_t == tile_idx)
                        v_correct += int(both.sum().item())
                vden = max(1, v_count)
                val_acc = (v_correct / max(1, v_count))
                print(f"\nWarmUp {epoch+1} [val] - total: {v_total/vden:.4f} | policy: {v_pol/vden:.4f} | value: {v_val/vden:.4f} | acc: {val_acc:.4f}")
                model.train()

            # Display both train and validation accuracy
            if val_acc is not None:
                print(f"WarmUp {epoch+1} [train] - acc: {train_acc:.4f}")
                epoch_acc = val_acc  # Use validation accuracy for threshold checking
            else:
                print(f"WarmUp {epoch+1} [train] - acc: {train_acc:.4f}")
                epoch_acc = train_acc

            if epoch_acc >= threshold:
                reached = True
                print(f"Warm-up threshold met: accuracy {epoch_acc:.4f} >= {threshold:.4f}. Switching to PPO.")
            initial_acc = epoch_acc
            epoch += 1

    best_loss = math.inf
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    epochs_no_improve = 0
    # Store previous epoch's validation policy probabilities for BOTH heads to compute joint KL(prev || curr)
    prev_val_action_probs: torch.Tensor | None = None
    prev_val_tile_probs: torch.Tensor | None = None
    # Track consecutive epochs where joint KL(prev||curr) <= threshold (when enabled)
    kl_below_count: int = 0
    # Latest computed validation joint KL (prev||curr) average for logging/ES
    v_avg_joint_kl_prev_curr: float | None = None
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
                advantages, returns, mode='ppo',
                epsilon=epsilon,
                value_coeff=value_coeff,
                entropy_coeff=entropy_coeff,
                print_debug=False,
            )

            opt.zero_grad()
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
            # Per-batch progress like Keras
            progress.set_postfix(
                loss=f"{float(total.item()):.4f}",
                pol=f"{float(policy_loss.item()):.4f}",
                val=f"{float(value_loss.item()):.4f}"
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

        # Evaluate on holdout set
        if dl_val is not None:
            model.eval()
            val_total = val_pol = val_val = 0.0
            val_pol_main = 0.0
            val_count = 0
            # Joint KL(prev || curr) across validation set (available from epoch >= 1)
            val_joint_kl_prev_curr_sum = 0.0
            offset = 0
            curr_action_probs_chunks: list[torch.Tensor] = []
            curr_tile_probs_chunks: list[torch.Tensor] = []
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
                    )

                    val_total += float(total_v.item()) * bsz
                    val_pol += float(policy_loss_v.item()) * bsz
                    val_val += float(value_loss_v.item()) * bsz
                    val_pol_main += float(policy_loss_v.item()) * bsz
                    # Collect current policy probabilities for this batch for BOTH heads
                    a_curr, t_curr, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                    curr_action_probs_chunks.append(a_curr.detach().cpu())
                    curr_tile_probs_chunks.append(t_curr.detach().cpu())
                    # If previous epoch's probs are available, compute joint KL(prev || curr) = KL_a + KL_t
                    if prev_val_action_probs is not None and prev_val_tile_probs is not None:
                        prev_slice_a = prev_val_action_probs[offset:offset + bsz, :]
                        prev_slice_t = prev_val_tile_probs[offset:offset + bsz, :]
                        # Numerical stability
                        prev_a_safe = prev_slice_a.clamp_min(1e-8)
                        curr_a_safe = a_curr.clamp_min(1e-8)
                        prev_t_safe = prev_slice_t.clamp_min(1e-8)
                        curr_t_safe = t_curr.clamp_min(1e-8)
                        # KL(prev||curr) per sample for both heads
                        kl_a = (prev_a_safe * (prev_a_safe.log() - curr_a_safe.detach().cpu().log())).sum(dim=1)
                        kl_t = (prev_t_safe * (prev_t_safe.log() - curr_t_safe.detach().cpu().log())).sum(dim=1)
                        kl_joint = kl_a + kl_t
                        val_joint_kl_prev_curr_sum += float(kl_joint.mean().item()) * bsz
                    offset += bsz

                    val_count += bsz
            vden = max(1, val_count)
            v_avg = val_total / vden
            v_avg_pol = val_pol / vden
            v_avg_val = val_val / vden
            # Finalize current epoch's concatenated validation probs for next epoch
            curr_val_action_probs = torch.cat(curr_action_probs_chunks, dim=0) if curr_action_probs_chunks else None
            curr_val_tile_probs = torch.cat(curr_tile_probs_chunks, dim=0) if curr_tile_probs_chunks else None
            # Compute joint KL(prev||curr) average if prev exists; otherwise mark as None
            v_avg_joint_kl_prev_curr = (val_joint_kl_prev_curr_sum / vden) if (
                        prev_val_action_probs is not None and prev_val_tile_probs is not None and vden > 0) else None
            print(
                f"\nEpoch {epoch + 1}/{epochs} [val] - total: {v_avg:.4f} | policy: {v_avg_pol:.4f} | value: {v_avg_val:.4f}"
                + (
                    f" | joint_kl(prev||curr): {v_avg_joint_kl_prev_curr:.4f}" if v_avg_joint_kl_prev_curr is not None else " | joint_kl(prev||curr): N/A")
            )
            # Update previous epoch validation policy probabilities for next epoch
            if curr_val_action_probs is not None and curr_val_tile_probs is not None:
                prev_val_action_probs = curr_val_action_probs
                prev_val_tile_probs = curr_val_tile_probs
            model.train()

        # Early stopping
        if dl_val is not None:
            # KL-based early stopping with patience: require joint KL(prev||curr) <= threshold for `patience` consecutive epochs
            if kl_threshold is not None and v_avg_joint_kl_prev_curr is not None:
                if v_avg_joint_kl_prev_curr <= float(kl_threshold):
                    kl_below_count += 1
                else:
                    kl_below_count = 0
                if patience and kl_below_count >= int(max(1, patience)):
                    print(
                        f"Early stopping triggered: joint KL(prev||curr) <= {float(kl_threshold):.6f} for {kl_below_count} consecutive epoch(s) (patience={int(patience)})"
                    )
                    break

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
    args = ap.parse_args()

    train_ppo(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        min_delta=float(args.min_delta),
        init_model=args.init,
        warm_up_acc=args.warm_up_acc,
        warm_up_max_epochs=args.warm_up_max_epochs,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        kl_threshold=args.kl_threshold,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()


