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

from core.learn import ACPlayer
from core.learn.ac_network import ACNetwork
from sklearn.preprocessing import StandardScaler  # type: ignore

# not using called discards yet
class ACDataset(Dataset):
    def __init__(self, npz_path: str, net: ACNetwork, *, negative_reward_weight: float = 1.0):
        data = np.load(npz_path, allow_pickle=True)
        # New compact format fields
        hand_arr = data['hand_idx']
        disc_arr = data['disc_idx']
        called_arr = data['called_idx']
        gsv_arr = data['game_state']
        self.flat_idx = data['flat_idx']
        # Load returns/advantages and apply optional negative reward weight transform
        returns = data['returns'].astype(np.float32)
        advantages = data['advantages'].astype(np.float32)
        nrw = float(max(0.0, negative_reward_weight))
        if nrw != 1.0:
            # Scale only negative returns; positives unchanged
            neg_mask = returns < 0
            returns[neg_mask] = returns[neg_mask] * nrw
            advantages[neg_mask] = advantages[neg_mask] * nrw
        self.returns = returns
        self.advantages = advantages
        self.old_log_probs = data['old_log_probs'].astype(np.float32)

        # fit the standard scaler so we can properly use extract_features
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
            'flat_idx': int(self.flat_idx[idx]),
            'return': float(self.returns[idx]),
            'advantage': float(self.advantages[idx]),
            'old_log_prob': float(self.old_log_probs[idx]),
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
    flat_idx = torch.tensor(batch['flat_idx'], dtype=torch.long, device=dev)
    old_log_probs = torch.tensor(batch['old_log_prob'], dtype=torch.float32, device=dev)
    advantages = torch.tensor(batch['advantage'], dtype=torch.float32, device=dev)
    returns = torch.tensor(batch['return'], dtype=torch.float32, device=dev)
    return (
        hand, calls, disc, gsv,
        flat_idx,
        old_log_probs,
        advantages, returns,
    )


def _compute_losses(
    model: torch.nn.Module,
    hand: torch.Tensor,
    calls: torch.Tensor,
    disc: torch.Tensor,
    gsv: torch.Tensor,
    flat_idx: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    mode: str = 'ppo',
    epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
):
    pp, val = model(hand.float(), calls.float(), disc.float(), gsv.float())
    val = val.squeeze(1)
    batch_size_curr = int(gsv.size(0))
    if mode == 'bc':
        log_pp = (pp.clamp_min(1e-8)).log()
        policy_loss = F.nll_loss(log_pp, flat_idx.to(dtype=torch.long, device=pp.device))
        value_loss = F.mse_loss(val.view_as(returns), returns)
        total = policy_loss + value_coeff * value_loss
        return total, policy_loss, value_loss, batch_size_curr
    # PPO path
    idx = flat_idx.to(device=pp.device, dtype=torch.long).view(-1, 1)
    chosen = torch.gather(pp.clamp_min(1e-8), 1, idx).squeeze(1)
    logp = chosen.log()
    ratio = torch.exp(logp - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = F.mse_loss(val.view_as(returns), returns)
    entropy_loss = -(pp * (pp.clamp_min(1e-8).log())).sum(dim=1).mean()
    total = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
    return total, policy_loss, value_loss, batch_size_curr


def train_ppo(
    dataset_path: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    device: str | None = None,
    patience: int = 0,
    min_delta: float = 1e-4,
    val_split: float = 0.1,
    init_model: str | None = None,
    warm_up_acc: float = 0.0,
    warm_up_max_epochs: int = 50,
    *,
    hidden_size: int = 128,
    embedding_dim: int = 16,
    negative_reward_weight: float = 1.0,
) -> str:
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


    if init_model:
        try:
            player = ACPlayer.from_directory(init_model)
            net = player.network

            # reset standard scaler since we're retraining it
            net._gsv_scaler = StandardScaler()

            print(f"Loaded initial weights from {init_model}")
        except Exception as e:
            raise ValueError(f"Warning: failed to load init model '{init_model}': {e}")
    else:
        gsv_scaler = StandardScaler()
        # Initialize AC network first so we can use its feature extraction table and standard scaler for preprocessing
        net = ACNetwork(gsv_scaler=gsv_scaler, hidden_size=hidden_size, embedding_dim=embedding_dim, temperature=0.05)
        net = net.to(dev)
        # Ensure the scaler we persist is the exact one used by the network during preprocessing
        player = ACPlayer(player_id=0, gsv_scaler=gsv_scaler, network=net)

    model = net.torch_module
    ds = ACDataset(dataset_path, net, negative_reward_weight=negative_reward_weight)
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
                    flat_idx,
                    _old_log_probs,
                    _advantages,
                    _returns,
                ) = _prepare_batch_tensors(batch, dev)
                pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                pred = torch.argmax(pp, dim=1)
                correct += int((pred == flat_idx).sum().item())
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
                    flat_idx,
                    old_log_probs,
                    advantages,
                    returns,
                ) = _prepare_batch_tensors(batch, dev)

                loss, policy_loss, value_loss, _ = _compute_losses(
                    model,
                    hand, calls, disc, gsv,
                    flat_idx,
                    old_log_probs,
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
                    pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                    pred = torch.argmax(pp, dim=1)
                    correct += int((pred == flat_idx).sum().item())
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
                            flat_idx,
                            old_log_probs,
                            advantages,
                            returns,
                        ) = _prepare_batch_tensors(vb, dev)
                        loss_v, policy_loss_v, value_loss_v, _ = _compute_losses(
                            model,
                            hand, calls, disc, gsv,
                            flat_idx,
                            old_log_probs,
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
                        pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
                        pred = torch.argmax(pp, dim=1)
                        v_correct += int((pred == flat_idx).sum().item())
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
    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss_main = 0.0
        progress = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress:
            tensors = _prepare_batch_tensors(batch, dev)
            (
                hand, calls, disc, gsv,
                flat_idx,
                old_log_probs,
                advantages, returns,
            ) = tensors

            total, policy_loss, value_loss, batch_size_curr = _compute_losses(
                model, hand, calls, disc, gsv,
                flat_idx,
                old_log_probs,
                advantages, returns, mode='ppo',
                epsilon=epsilon,
                value_coeff=value_coeff,
                entropy_coeff=entropy_coeff,
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
            # Per-batch progress like Keras
            progress.set_postfix(
                loss=f"{float(total.item()):.4f}",
                pol=f"{float(policy_loss.item()):.4f}",
                val=f"{float(value_loss.item()):.4f}",
            )

        # Evaluate on holdout set
        if dl_val is not None:
            model.eval()
            val_total = val_pol = val_val = 0.0
            val_pol_main = 0.0
            val_count = 0
            with torch.no_grad():
                for vb in dl_val:
                    tensors = _prepare_batch_tensors(vb, dev)
                    (
                        hand, calls, disc, gsv,
                        flat_idx,
                        old_log_probs,
                        advantages, returns,
                    ) = tensors
                    total_v, policy_loss_v, value_loss_v, bsz = _compute_losses(
                        model, hand, calls, disc, gsv,
                        flat_idx,
                        old_log_probs,
                        advantages, returns,
                        epsilon=epsilon,
                        value_coeff=value_coeff,
                        entropy_coeff=entropy_coeff,
                    )

                    val_total += float(total_v.item()) * bsz
                    val_pol += float(policy_loss_v.item()) * bsz
                    val_val += float(value_loss_v.item()) * bsz
                    val_pol_main += float(policy_loss_v.item()) * bsz
                    
                    val_count += bsz
            vden = max(1, val_count)
            v_avg = val_total / vden
            v_avg_pol = val_pol / vden
            v_avg_val = val_val / vden
            v_avg_pol_main = val_pol_main / vden
            print(
                f"\nEpoch {epoch+1}/{epochs} [val] - total: {v_avg:.4f} | policy: {v_avg_pol:.4f} | value: {v_avg_val:.4f}"
            )
            model.train()

        # Early stopping on total loss (only when validation split is used)
        if dl_val is not None:
            if v_avg < best_loss - float(min_delta):
                best_loss = v_avg
                epochs_no_improve = 0
                # Save best checkpoint
                out_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
                os.makedirs(out_dir, exist_ok=True)

                # Create best model directory
                best_dir = os.path.join(out_dir, f'ac_ppo_best_{timestamp}')
                os.makedirs(best_dir, exist_ok=True)

                # Save model weights (state_dict) for portability
                net.save_model(os.path.join(best_dir, 'model.pt'), save_entire_module=False)

            else:
                epochs_no_improve += 1
                if 0 < patience <= epochs_no_improve:
                    print(f"Early stopping triggered (patience={patience}, min_delta={min_delta})")
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
    print(f"Saved StandardScaler to {scaler_path}")

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
    ap.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    ap.add_argument('--min_delta', type=float, default=1e-4, help='Minimum improvement in val loss to reset patience')
    ap.add_argument('--init', type=str, default=None, help='Path to initial AC model weights/module to load')
    ap.add_argument('--warm_up_acc', type=float, default=0.0, help='Accuracy threshold to reach with behavior cloning before switching to PPO (0 disables)')
    ap.add_argument('--warm_up_max_epochs', type=int, default=50, help='Maximum warm-up epochs before switching even if threshold not reached')
    ap.add_argument('--hidden_size', type=int, default=128, help='Hidden size for ACNetwork')
    ap.add_argument('--embedding_dim', type=int, default=16, help='Embedding dimension for ACNetwork')
    ap.add_argument('--negative_reward_weight', type=float, default=1.0, help='Scale factor applied to negative returns/advantages after loading dataset (>=0). 1.0 means no change')
    args = ap.parse_args()

    train_ppo(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        patience=args.patience,
        min_delta=float(args.min_delta),
        init_model=args.init,
        warm_up_acc=args.warm_up_acc,
        warm_up_max_epochs=args.warm_up_max_epochs,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        negative_reward_weight=float(max(0.0, args.negative_reward_weight)),
    )


if __name__ == '__main__':
    main()


