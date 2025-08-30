#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import time
from typing import Dict, Any

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


class DealInNet(nn.Module):
    """Deal-in predictor using AC-like hand/calls encoders and discard attention.

    Inputs:
      - player_hand_idx: (B, MAX_CONCEALED_TILES)
      - player_called_idx: (B, MAX_CALLED_TILES_PER_PLAYER)
      - opp_called_idx: (B, 3, MAX_CALLED_TILES_PER_PLAYER)
      - disc_idx: (B, NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER)
    Output:
      - logits: (B, 37)  (use BCE-with-logits)
    """
    def __init__(self, embedding_dim: int = 16, hidden: int = 128):
        super().__init__()
        emb = int(embedding_dim)
        self.pad_index = int(UNIQUE_TILE_COUNT)  # 37 used as PAD
        self.tile_emb = nn.Embedding(int(UNIQUE_TILE_COUNT) + 1, emb, padding_idx=self.pad_index)
        self.disc_seq_len = int(MAX_DISCARDS_PER_PLAYER)

        # Hand conv path
        self.hand_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.hand_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.hand_gap = nn.AdaptiveAvgPool2d((1, 1))

        # Calls shared linear encoder
        self.call_linear = nn.Linear(emb, 64)
        self.call_reduce = nn.Linear(64, 16)

        # Discard attention (keys/values from discards; queries from hand)
        self.disc_pos_emb = nn.Embedding(self.disc_seq_len, 8)
        self.disc_player_emb = nn.Embedding(int(NUM_PLAYERS), 4)
        # emb -> Dkv (= emb + 8 + 4)
        self.query_proj = nn.Linear(emb, emb + 8 + 4)
        # Reduce flattened (Q * Dkv) -> 64; bias-free similar to AC
        self.query_reduce = nn.Linear(int(MAX_CONCEALED_TILES) * (emb + 8 + 4), 64, bias=False)

        # Head: concat(hand64, calls64, discAttn64) -> hidden -> 37 logits
        in_dim = 64 + 64 + 64
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(UNIQUE_TILE_COUNT)),
        )

    def _padify(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, torch.full_like(x, self.pad_index), x)

    def forward(self, player_hand_idx: torch.Tensor, player_called_idx: torch.Tensor, opp_called_idx: torch.Tensor, disc_idx: torch.Tensor) -> torch.Tensor:
        B = player_hand_idx.shape[0]
        # Hand features
        hand_idx = self._padify(player_hand_idx).long()
        hand_emb = self.tile_emb(hand_idx)                     # (B, H, emb)
        hand_2d = hand_emb.unsqueeze(1)                        # (B,1,H,emb)
        hand_feat = self.hand_gap(F.relu(self.hand_conv2(F.relu(self.hand_conv1(hand_2d))))).view(B, 64)

        # Calls features (player + 3 opp = 4)
        all_calls = torch.cat([
            player_called_idx.unsqueeze(1),
            opp_called_idx,
        ], dim=1)  # (B,4,C)
        all_calls_padded = self._padify(all_calls)
        all_calls_emb = self.tile_emb(all_calls_padded)        # (B,4,C,emb)
        mask = (all_calls_padded != self.pad_index).float().unsqueeze(-1)
        pooled = (all_calls_emb * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)   # (B,4,emb)
        call_features = F.dropout(F.relu(self.call_linear(pooled)), p=0.3, training=self.training)  # (B,4,64)
        call_features_reduced = self.call_reduce(call_features)  # (B,4,16)
        call_feat = call_features_reduced.view(B, 4 * 16)        # (B,64)

        # Discard attention features
        d_idx = self._padify(disc_idx).long()                                   # (B,P,D)
        d_tile = self.tile_emb(d_idx)                                           # (B,P,D,emb)
        pos_ids = torch.arange(self.disc_seq_len, device=d_idx.device).view(1, 1, -1).expand(B, int(NUM_PLAYERS), -1)
        d_pos = self.disc_pos_emb(pos_ids)                                      # (B,P,D,8)
        ply_ids = torch.arange(int(NUM_PLAYERS), device=d_idx.device).view(1, -1, 1).expand(B, -1, self.disc_seq_len)
        d_ply = self.disc_player_emb(ply_ids)                                   # (B,P,D,4)
        d_kv = torch.cat([d_tile, d_pos, d_ply], dim=-1)                        # (B,P,D,Dkv)
        Dkv = d_kv.shape[-1]
        N = int(NUM_PLAYERS) * self.disc_seq_len
        d_kv = d_kv.view(B, N, Dkv)                                             # (B,N,Dkv)
        d_mask = (d_idx.view(B, N) != self.pad_index)                           # (B,N)

        q_idx = self._padify(player_hand_idx).long()                            # (B,H)
        q_emb = self.tile_emb(q_idx)                                            # (B,H,emb)
        q_proj = self.query_proj(q_emb)                                         # (B,H,Dkv)
        attn_logits = torch.matmul(q_proj, d_kv.transpose(1, 2)) / (Dkv ** 0.5) # (B,H,N)
        mask = d_mask.unsqueeze(1)                                              # (B,1,N)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_w = F.softmax(attn_logits, dim=-1)
        attn_w = attn_w * mask
        denom = attn_w.sum(dim=-1, keepdim=True)
        has_valid = denom > 0
        attn_w = torch.where(has_valid, attn_w / torch.clamp_min(denom, 1e-12), torch.zeros_like(attn_w))
        d_per_query = torch.matmul(attn_w, d_kv)                                 # (B,H,Dkv)
        q_flat = d_per_query.reshape(B, -1)                                      # (B,H*Dkv)
        d_feat = self.query_reduce(q_flat)                                       # (B,64)

        x = torch.cat([hand_feat, call_feat, d_feat], dim=1)
        logits = self.head(x)
        return logits  # BCE-with-logits outside


def _resolve_loader_defaults(os_hint: str | None) -> Dict[str, Any]:
    plat = os.sys.platform if os_hint is None else os_hint
    cpu_count = max(1, (os.cpu_count() or 1))
    if plat.startswith('win') or plat == 'windows':
        dl_workers = min(8, max(0, cpu_count // 2))
        pin = True
    elif plat == 'darwin' or plat == 'mac':
        dl_workers = 0  # avoid pickling problems on macOS
        pin = False
    else:
        dl_workers = min(16, max(0, cpu_count - 2))
        pin = True

    cfg: Dict[str, Any] = {
        'num_workers': int(max(0, dl_workers)),
        'pin_memory': bool(pin),
    }
    if cfg['num_workers'] > 0:
        cfg['prefetch_factor'] = 2
        cfg['persistent_workers'] = True
    else:
        cfg['persistent_workers'] = False
    return cfg


def _prepare_batch_tensors(batch: Dict[str, Any], dev: torch.device):
    def _to_dev(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    player_hand_idx = _to_dev(batch['player_hand_idx'], torch.long)
    player_called_idx = _to_dev(batch['player_called_idx'], torch.long)
    opp_called_idx = _to_dev(batch['opp_called_idx'], torch.long)
    disc_idx = _to_dev(batch['disc_idx'], torch.long)
    deal_in_mask = _to_dev(batch['deal_in_mask'], torch.float32)
    return player_hand_idx, player_called_idx, opp_called_idx, disc_idx, deal_in_mask


def _hand_presence_mask(player_hand_idx: torch.Tensor) -> torch.Tensor:
    """Build a (B,37) binary mask of which tiles are present in hand (ignoring pads).
    """
    B = player_hand_idx.size(0)
    U = int(UNIQUE_TILE_COUNT)
    idx = player_hand_idx.long().clamp_min(-1)
    valid = (idx >= 0) & (idx < U)
    # Map invalid to 0, will be removed by valid mask anyway
    idx_clipped = torch.where(valid, idx, torch.zeros_like(idx))
    # Build bincount per batch
    offsets = (torch.arange(B, device=idx.device).unsqueeze(1) * U).long()
    flat = (idx_clipped + offsets).view(-1)
    w = valid.view(-1).to(torch.float32)
    counts = torch.bincount(flat, weights=w, minlength=B * U).view(B, U)
    presence = (counts > 0).to(torch.float32)
    return presence


class _SubsetWithDiscards(torch.utils.data.Dataset):
    """Wrap a Subset[ACDataset] to add raw disc_idx into each sample dict.

    We rely on the underlying ACDataset's memmap handle (`_data['disc_idx']`).
    """
    def __init__(self, subset: Subset | None):
        self.subset = subset
        if subset is not None:
            self.base = subset.dataset  # ACDataset
            self.indices = subset.indices  # List[int]
        else:
            self.base = None
            self.indices = None

    def __len__(self):
        return 0 if (self.subset is None) else len(self.indices)  # type: ignore[arg-type]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        assert self.subset is not None and self.base is not None and self.indices is not None
        orig_idx = int(self.indices[i])
        item = self.base[orig_idx]
        # Inject raw disc_idx from npz for attention path
        try:
            disc_idx = self.base._data['disc_idx'][orig_idx]  # numpy array
        except Exception:
            # Fallback: no discards available; synthesize all -1s
            P = int(NUM_PLAYERS)
            D = int(MAX_DISCARDS_PER_PLAYER)
            import numpy as _np
            disc_idx = _np.full((P, D), -1, dtype=_np.int32)
        item['disc_idx'] = disc_idx
        return item


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
    # Wrap subsets to ensure 'disc_idx' is present in each sample
    wrapped_train = _SubsetWithDiscards(ds_train)
    wrapped_val = _SubsetWithDiscards(ds_val) if isinstance(ds_val, Subset) else None

    dl_train = DataLoader(wrapped_train, batch_size=batch_size, shuffle=True, drop_last=False, **ld_cfg)
    dl_val = DataLoader(wrapped_val, batch_size=batch_size, shuffle=False, drop_last=False, **ld_cfg) if wrapped_val is not None else None

    model = DealInNet(embedding_dim=16, hidden=128).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    best_path = None

    bce = nn.BCEWithLogitsLoss(reduction='none')

    def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
        # logits, target, hand_mask: (B,37)
        per_elem = bce(logits, target)
        # zero out non-hand tiles
        per_elem = per_elem * hand_mask
        denom = hand_mask.sum(dim=1).clamp_min(1.0)  # avoid div by zero
        per_sample = per_elem.sum(dim=1) / denom
        return per_sample.mean()

    for epoch in range(int(max(1, epochs))):
        model.train()
        tr_loss = 0.0
        tr_cnt = 0
        for batch in tqdm(dl_train, desc=f"Train e{epoch+1}/{int(max(1, epochs))}", leave=False):
            ph, pc, oc, dc, di = _prepare_batch_tensors(batch, dev)
            logits = model(ph, pc, oc, dc)
            hand_mask = _hand_presence_mask(ph)
            loss = masked_bce_loss(logits, di, hand_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = int(ph.size(0))
            tr_loss += float(loss.item()) * bsz
            tr_cnt += bsz

        print(f"Epoch {epoch+1} [train] - masked_bce={tr_loss/max(1,tr_cnt):.6f}")

        # Validation
        if dl_val is not None:
            model.eval()
            va_loss = 0.0
            va_cnt = 0
            with torch.no_grad():
                for batch in tqdm(dl_val, desc="Val", leave=False):
                    ph, pc, oc, dc, di = _prepare_batch_tensors(batch, dev)
                    logits = model(ph, pc, oc, dc)
                    hand_mask = _hand_presence_mask(ph)
                    loss = masked_bce_loss(logits, di, hand_mask)
                    bsz = int(ph.size(0))
                    va_loss += float(loss.item()) * bsz
                    va_cnt += bsz
            va = va_loss / max(1, va_cnt)
            print(f"Epoch {epoch+1} [val]   - masked_bce={va:.6f}")

            # Save best by val masked BCE
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                if best_val is None or va < best_val:
                    best_val = va
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    best_path = os.path.join(save_dir, f'deals_limit_{ts}_bce{va:.4f}.pt')
                    torch.save({'state_dict': model.state_dict()}, best_path)
                    print(f"Saved checkpoint: {best_path}")

    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a deal-in predictor from AC features (hand/calls/attn -> 37) with hand-masked BCE')
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
