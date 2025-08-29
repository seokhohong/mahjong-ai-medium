#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# Local imports
from core.learn.ac_network import ACNetwork
from core.learn.ac_constants import MAX_CONCEALED_TILES
from core.constants import NUM_PLAYERS, MAX_CALLED_TILES_PER_PLAYER, MAX_DISCARDS_PER_PLAYER
from run.train_model import ACDataset  # reuse dataset and game-based split
from core.tile import UNIQUE_TILE_COUNT


class WallConvNet(nn.Module):
    """Wall counting model with AC-like hand/calls processing and discard counts.

    Inputs:
      - player_hand_idx: (B, MAX_CONCEALED_TILES)
      - player_called_idx: (B, MAX_CALLED_TILES_PER_PLAYER)
      - opp_called_idx: (B, 3, MAX_CALLED_TILES_PER_PLAYER)
      - disc_counts: (B, 37)  visible counts from discards only
    Output:
      - wall_counts_pred: (B, 37)
    """
    def __init__(self, embedding_dim: int = 16, hidden: int = 128):
        super().__init__()
        emb = int(embedding_dim)
        self.pad_index = int(UNIQUE_TILE_COUNT)  # 37 used as PAD
        self.tile_emb = nn.Embedding(int(UNIQUE_TILE_COUNT) + 1, emb, padding_idx=self.pad_index)
        self.disc_seq_len = int(MAX_DISCARDS_PER_PLAYER)

        # Hand conv path: (B,1,MAX_CONCEALED_TILES,emb) -> GAP -> 64
        self.hand_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.hand_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.hand_gap = nn.AdaptiveAvgPool2d((1, 1))

        # Calls shared linear encoder: per player pooled emb -> 64 -> reduce to 16 per player
        self.call_linear = nn.Linear(emb, 64)
        self.call_reduce = nn.Linear(64, 16)

        # Discard attention modules (keys/values from discards; queries from hand)
        self.disc_pos_emb = nn.Embedding(self.disc_seq_len, 8)
        self.disc_player_emb = nn.Embedding(int(NUM_PLAYERS), 4)
        # emb -> Dkv (= emb + 8 + 4)
        self.query_proj = nn.Linear(emb, emb + 8 + 4)
        # Reduce flattened (Q * Dkv) -> 64; bias-free like AC
        self.query_reduce = nn.Linear(int(MAX_CONCEALED_TILES) * (emb + 8 + 4), 64, bias=False)

        # Head: concat(hand64, calls64, discAttn64) -> hidden -> 37
        in_dim = 64 + 64 + 64
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(UNIQUE_TILE_COUNT)),
        )

    def _padify(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, torch.full_like(x, self.pad_index), x)

    def forward(
        self,
        player_hand_idx: torch.Tensor,
        player_called_idx: torch.Tensor,
        opp_called_idx: torch.Tensor,
        disc_idx: torch.Tensor,
    ) -> torch.Tensor:
        B = player_hand_idx.shape[0]
        # Hand features
        hand_idx = self._padify(player_hand_idx).long()
        hand_emb = self.tile_emb(hand_idx)                     # (B, MAX_CONCEALED_TILES, emb)
        hand_2d = hand_emb.unsqueeze(1)                        # (B,1,MAX_CONCEALED_TILES,emb)
        hand_feat = self.hand_gap(F.relu(self.hand_conv2(F.relu(self.hand_conv1(hand_2d))))).view(B, 64)

        # Calls features (player + 3 opp = 4)
        all_calls = torch.cat([
            player_called_idx.unsqueeze(1),
            opp_called_idx,
        ], dim=1)  # (B,4,MAX_CALLED_TILES_PER_PLAYER)
        all_calls_padded = self._padify(all_calls)
        all_calls_emb = self.tile_emb(all_calls_padded)        # (B,4,MAX_CALLED_TILES_PER_PLAYER,emb)
        mask = (all_calls_padded != self.pad_index).float().unsqueeze(-1)
        pooled = (all_calls_emb * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)   # (B,4,emb)
        call_features = F.dropout(F.relu(self.call_linear(pooled)), p=0.3, training=self.training)  # (B,4,64)
        call_features_reduced = self.call_reduce(call_features)  # (B,4,16)
        call_feat = call_features_reduced.view(B, 4 * 16)        # (B,64)

        # Discard attention features
        d_idx = self._padify(disc_idx).long()                                   # (B,NUM_PLAYERS,disc_seq_len)
        d_tile = self.tile_emb(d_idx)                                           # (B,NUM_PLAYERS,disc_seq_len,emb)
        pos_ids = torch.arange(self.disc_seq_len, device=d_idx.device).view(1, 1, -1).expand(B, int(NUM_PLAYERS), -1)
        d_pos = self.disc_pos_emb(pos_ids)                                      # (B,NUM_PLAYERS,disc_seq_len,8)
        ply_ids = torch.arange(int(NUM_PLAYERS), device=d_idx.device).view(1, -1, 1).expand(B, -1, self.disc_seq_len)
        d_ply = self.disc_player_emb(ply_ids)                                   # (B,NUM_PLAYERS,disc_seq_len,4)
        d_kv = torch.cat([d_tile, d_pos, d_ply], dim=-1)                        # (B,NUM_PLAYERS,disc_seq_len,Dkv)
        Dkv = d_kv.shape[-1]
        N = int(NUM_PLAYERS) * self.disc_seq_len
        d_kv = d_kv.view(B, N, Dkv)                                             # (B,N,Dkv)
        d_mask = (d_idx.view(B, N) != self.pad_index)                           # (B,N)

        q_idx = self._padify(player_hand_idx).long()                            # (B,MAX_CONCEALED_TILES)
        q_emb = self.tile_emb(q_idx)                                            # (B,MAX_CONCEALED_TILES,emb)
        q_proj = self.query_proj(q_emb)                                         # (B,MAX_CONCEALED_TILES,Dkv)
        attn_logits = torch.matmul(q_proj, d_kv.transpose(1, 2)) / (Dkv ** 0.5) # (B,Q,N)
        mask = d_mask.unsqueeze(1)                                              # (B,1,N)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_w = F.softmax(attn_logits, dim=-1)
        attn_w = attn_w * mask
        denom = attn_w.sum(dim=-1, keepdim=True)
        has_valid = denom > 0
        attn_w = torch.where(has_valid, attn_w / torch.clamp_min(denom, 1e-12), torch.zeros_like(attn_w))
        d_per_query = torch.matmul(attn_w, d_kv)                                 # (B,MAX_CONCEALED_TILES,Dkv)
        q_flat = d_per_query.reshape(B, -1)                                      # (B,Q*Dkv)
        d_feat = self.query_reduce(q_flat)                                       # (B,64)

        x = torch.cat([hand_feat, call_feat, d_feat], dim=1)
        out = self.head(x)
        return out


def _resolve_loader_defaults(os_hint: str | None) -> Dict[str, Any]:
    # Simple heuristic similar to train_model._resolve_loader_defaults
    # On macOS/spawn, avoid multiprocessing due to pickling issues with memmaps.
    plat = os.sys.platform if os_hint is None else os_hint
    cpu_count = max(1, (os.cpu_count() or 1))
    if plat.startswith('win') or plat == 'windows':
        dl_workers = min(8, max(0, cpu_count // 2))
        pin = True
    elif plat == 'darwin' or plat == 'mac':
        dl_workers = 0  # critical: avoid pickling dataset state on macOS
        pin = False
    else:
        dl_workers = min(16, max(0, cpu_count - 2))
        pin = True

    cfg: Dict[str, Any] = {
        'num_workers': int(max(0, dl_workers)),
        'pin_memory': bool(pin),
    }
    # Only valid when using workers > 0
    if cfg['num_workers'] > 0:
        cfg['prefetch_factor'] = 2
        cfg['persistent_workers'] = True
    else:
        cfg['persistent_workers'] = False
    return cfg


def _build_visible_counts(
    player_hand_idx: torch.Tensor,
    player_called_idx: torch.Tensor,
    opp_called_idx: torch.Tensor,
    disc_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute 37-d visible tile counts per sample from indices (>=0 valid).

    Inputs:
      - player_hand_idx: (B, H)
      - player_called_idx: (B, C)
      - opp_called_idx: (B, 3, C)
      - disc_idx: (B, P, D)
    Returns:
      - counts: (B, 37)
    """
    B = player_hand_idx.shape[0]
    U = int(UNIQUE_TILE_COUNT)
    dev = player_hand_idx.device

    hand = player_hand_idx.view(B, -1)
    pc = player_called_idx.view(B, -1)
    oc = opp_called_idx.view(B, -1)
    dc = disc_idx.view(B, -1)
    all_idx = torch.cat([hand, pc, oc, dc], dim=1).long()  # (B, N)
    # mask valid tile ids
    mask = (all_idx >= 0) & (all_idx < U)
    # Flatten with per-batch offsets so we can bincount once
    offsets = (torch.arange(B, device=dev).unsqueeze(1) * U).long()  # (B,1)
    flat_idx = (all_idx.clamp_min(0) + offsets).view(-1)  # invalid will be corrected by mask weights
    weights = mask.view(-1).to(dtype=torch.float32, device=dev)
    counts = torch.bincount(flat_idx, weights=weights, minlength=B * U).view(B, U)
    return counts


def _build_discard_counts(
    disc_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute 37-d counts only from discards across all players."""
    B = disc_idx.shape[0]
    U = int(UNIQUE_TILE_COUNT)
    dev = disc_idx.device
    dc = disc_idx.view(B, -1).long()
    mask = (dc >= 0) & (dc < U)
    offsets = (torch.arange(B, device=dev).unsqueeze(1) * U).long()
    flat_idx = (dc.clamp_min(0) + offsets).view(-1)
    weights = mask.view(-1).to(dtype=torch.float32, device=dev)
    counts = torch.bincount(flat_idx, weights=weights, minlength=B * U).view(B, U)
    return counts


def _prepare_batch_tensors(batch: Dict[str, Any], dev: torch.device):
    def _to_dev(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    player_hand_idx = _to_dev(batch['player_hand_idx'], torch.long)
    player_called_idx = _to_dev(batch['player_called_idx'], torch.long)
    opp_called_idx = _to_dev(batch['opp_called_idx'], torch.long)
    disc_idx = _to_dev(batch['disc_idx'], torch.long)
    wall_count = _to_dev(batch['wall_count'], torch.float32)
    return player_hand_idx, player_called_idx, opp_called_idx, disc_idx, wall_count


def train(
    *,
    dataset_path: str,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    os_hint: str | None = None,
    save_dir: str | None = None,
) -> str | None:
    dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Minimal ACNetwork instance only for dataset feature extraction (we do not use its torch module)
    net = ACNetwork(gsv_scaler=None)
    ds = ACDataset(dataset_path, net, fit_scaler=False, mmap=True)
    ds_train, ds_val = ds.train_val_subsets_by_game(val_frac=float(max(0.0, min(1.0, val_frac))), seed=42)

    ld_cfg = _resolve_loader_defaults(os_hint)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, **ld_cfg)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, **ld_cfg) if isinstance(ds_val, Subset) else None

    model = WallConvNet(embedding_dim=16, hidden=128).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae = None
    best_path = None

    for epoch in range(int(max(1, epochs))):
        model.train()
        tr_loss = 0.0
        tr_cnt = 0
        tr_abs_err = 0.0
        tr_sq_err = 0.0
        for batch in tqdm(dl_train, desc=f"Train e{epoch+1}/{int(max(1, epochs))}", leave=False):
            ph, pc, oc, dc, wc = _prepare_batch_tensors(batch, dev)
            y = wc  # (B,37)

            pred = model(ph, pc, oc, dc).clamp_(0.0, 4.0)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = int(ph.size(0))
            tr_loss += float(loss.item()) * bsz
            tr_cnt += bsz
            tr_abs_err += torch.abs(pred - y).sum().item()
            tr_sq_err += torch.square(pred - y).sum().item()

        tr_mae = tr_abs_err / max(1, tr_cnt * int(UNIQUE_TILE_COUNT))
        tr_rmse = (tr_sq_err / max(1, tr_cnt * int(UNIQUE_TILE_COUNT))) ** 0.5
        print(f"Epoch {epoch+1} [train] - loss={tr_loss/max(1,tr_cnt):.6f} | mae={tr_mae:.6f} | rmse={tr_rmse:.6f}")

        # Validation
        if dl_val is not None:
            model.eval()
            va_loss = 0.0
            va_cnt = 0
            va_abs_err = 0.0
            va_sq_err = 0.0
            with torch.no_grad():
                for batch in tqdm(dl_val, desc="Val", leave=False):
                    ph, pc, oc, dc, wc = _prepare_batch_tensors(batch, dev)
                    y = wc
                    pred = model(ph, pc, oc, dc).clamp_(0.0, 4.0)
                    loss = F.mse_loss(pred, y)

                    bsz = int(ph.size(0))
                    va_loss += float(loss.item()) * bsz
                    va_cnt += bsz
                    va_abs_err += torch.abs(pred - y).sum().item()
                    va_sq_err += torch.square(pred - y).sum().item()

            va_mae = va_abs_err / max(1, va_cnt * int(UNIQUE_TILE_COUNT))
            va_rmse = (va_sq_err / max(1, va_cnt * int(UNIQUE_TILE_COUNT))) ** 0.5
            print(f"Epoch {epoch+1} [val]   - loss={va_loss/max(1,va_cnt):.6f} | mae={va_mae:.6f} | rmse={va_rmse:.6f}")

            # Save best by MAE
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                if best_val_mae is None or va_mae < best_val_mae:
                    best_val_mae = va_mae
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    best_path = os.path.join(save_dir, f'wall_limit_{ts}_mae{va_mae:.4f}.pt')
                    torch.save({'state_dict': model.state_dict()}, best_path)
                    print(f"Saved checkpoint: {best_path}")

    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a wall-only predictor from visible tile counts (37->128->37)')
    ap.add_argument('--dataset', type=str, required=True, help='Path to the AC dataset .npz')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--os-hint', type=str, default=None, choices=[None, 'mac', 'linux', 'windows'])
    ap.add_argument('--save-dir', type=str, default=None)
    args = ap.parse_args()

    best_path = train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        os_hint=args.os_hint,
        save_dir=args.save_dir,
    )
    if best_path:
        print(f"Best checkpoint: {best_path}")


if __name__ == '__main__':
    main()
