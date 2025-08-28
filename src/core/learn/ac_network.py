from __future__ import annotations

from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

import numpy as np

# Torch is installed in the project's Python environment (.venv312). Import directly.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game import GamePerspective, Tile, TileType, Suit
from ..tile import UNIQUE_TILE_COUNT
from core.constants import NUM_PLAYERS, MAX_CALLS, MAX_CALLED_SET_SIZE, MAX_CALLED_TILES_PER_PLAYER, MAX_DISCARDS_PER_PLAYER
from .feature_engineering import encode_game_perspective
from .ac_constants import (
    ACTION_HEAD_SIZE,
    TILE_HEAD_SIZE,
    MAX_CONCEALED_TILES
)


class ACNetwork:
    """
    Actor-Critic network for MediumJong with shared trunk and two policy heads:
    - Action head over ACTION_HEAD_SIZE
    - Tile head over TILE_HEAD_SIZE (0=no-op, 1..37=tiles)
    - Value head: scalar

    All heads share the same feature extractor from inputs to the pre-head layer.
    """

    def __init__(
        self,
        gsv_scaler: StandardScaler | None,
        hidden_size: int = 128,
        embedding_dim: int = 16,
        discard_projection_size: int = 64,
    ):
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.disc_proj_size = int(discard_projection_size)
        if gsv_scaler is None:
            print("Warning, instantiating ACNetwork with null StandardScaler")
            self._gsv_scaler = StandardScaler()
        else:
            # Respect provided scaler (may already be fit from dataset)
            self._gsv_scaler = gsv_scaler

        from ..constants import TOTAL_TILES
        from .ac_constants import GAME_STATE_VEC_LEN as GSV
        dealt = 13 * int(NUM_PLAYERS)
        self._max_discards_per_player = max(1, (int(TOTAL_TILES) - dealt) // int(NUM_PLAYERS))
        self._max_called_sets = int(MAX_CALLS)
        self._max_tiles_per_called_set = int(MAX_CALLED_SET_SIZE)
        # Fixed lengths derived from constants
        # 18 = 2 (pair) + MAX_CALLED_TILES_PER_PLAYER
        self.seq_len_total = 2 + int(MAX_CALLED_TILES_PER_PLAYER)
        # Discards per player use the global cap (typically 21)
        self.disc_seq_len = int(MAX_DISCARDS_PER_PLAYER)

        class _ACModule(nn.Module):
            def __init__(self, outer: 'ACNetwork') -> None:
                super().__init__()
                self.outer = outer
                # Embedding for tiles (37 unique + 1 pad)
                self.pad_index = int(UNIQUE_TILE_COUNT)  # 37 used as PAD
                self.tile_emb = nn.Embedding(int(UNIQUE_TILE_COUNT) + 1, outer.embedding_dim, padding_idx=self.pad_index)

                # Hand+calls conv over (B, 1, seq_len_total, emb+1)
                hc_in_channels = 1
                hc_width = outer.embedding_dim + 1
                self.hc_conv1 = nn.Conv2d(hc_in_channels, 32, kernel_size=3, padding=1)
                self.hc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.hc_gap = nn.AdaptiveAvgPool2d((1, 1))

                # Opponents conv shared for each of 3 stacks over (B*3,1,MAX_CALLED_TILES_PER_PLAYER,emb)
                self.opp_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.opp_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.opp_gap = nn.AdaptiveAvgPool2d((1, 1))
                # Reduce each opponent's feature to 16-d
                self.opp_reduce = nn.Linear(64, 16)

                # Discards attention (query-based):
                # - Keys/values built from discard tokens: tile_emb + pos_emb(8) + player_emb(4) -> Dkv
                # - Queries from player's concealed hand tiles (MAX_CONCEALED_TILES)
                # - Output: per-query attended vectors (B,MAX_CONCEALED_TILES,Dkv)
                #   Reduced via a single Linear over flattened (Q*Dkv) -> 32: (B, Q*Dkv) -> (B, 32)
                self.disc_pos_emb = nn.Embedding(outer.disc_seq_len, 8)
                self.disc_player_emb = nn.Embedding(int(NUM_PLAYERS), 4)
                self.query_proj = nn.Linear(outer.embedding_dim, outer.embedding_dim + 8 + 4)  # emb -> Dkv
                # Flattened reduction from (Q * Dkv) -> 32 (bias-free so zero attention -> zero features)
                self.query_reduce = nn.Linear(int(MAX_CONCEALED_TILES) * (outer.embedding_dim + 8 + 4), 32, bias=False)

                # Player concealed hand conv over (B,1,MAX_CONCEALED_TILES,emb)
                self.hand_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.hand_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.hand_gap = nn.AdaptiveAvgPool2d((1, 1))
                # Separate reducer for player's calls for clarity (64 -> 16)
                self.player_calls_reduce = nn.Linear(64, 16)

                # Separate trunks for policy and value; input is
                #   hand(64) + player_calls(16) + opp(3*16) + disc_attn(32) + react(emb) + gsv(GSV)
                from .ac_constants import GAME_STATE_VEC_LEN as GSV
                input_dim = 64 + 16 + (3 * 16) + 32 + outer.embedding_dim + int(GSV)
                # Shared trunk for policy, value, and auxiliary heads
                self.trunk = nn.Sequential(
                    nn.Linear(input_dim, outer.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(outer.hidden_size, outer.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                trunk_dim = outer.hidden_size // 2
                # Heads: policy (action, tile), value, wall_counts, and deal_in
                self.head_action = nn.Linear(trunk_dim, int(ACTION_HEAD_SIZE))
                self.head_tile = nn.Linear(trunk_dim, int(TILE_HEAD_SIZE))
                self.head_value = nn.Linear(trunk_dim, 1)
                self.head_wall_counts = nn.Linear(trunk_dim, int(UNIQUE_TILE_COUNT))
                self.head_deal_in = nn.Linear(trunk_dim, int(UNIQUE_TILE_COUNT))

            def forward(
                self,
                player_hand_idx: torch.Tensor,              # (B, MAX_CONCEALED_TILES)
                player_called_idx: torch.Tensor,            # (B, MAX_CALLED_TILES_PER_PLAYER)
                opp_called_idx: torch.Tensor,               # (B, 3, MAX_CALLED_TILES_PER_PLAYER)
                disc_idx: torch.Tensor,                     # (B, NUM_PLAYERS, disc_seq_len)
                game_tile_indicator_idx: torch.Tensor,      # (B, 5) [react, d1..d4] as indices or -1
                game_state_vec: torch.Tensor,               # (B, GSV) standardized numeric vector
            ):
                B = player_hand_idx.shape[0]

                # Helper: map negative indices to pad_index
                def padify(x: torch.Tensor) -> torch.Tensor:
                    return torch.where(x < 0, torch.full_like(x, self.pad_index), x)

                # Player concealed hand conv path
                hand_idx = padify(player_hand_idx).long()                               # (B, MAX_CONCEALED_TILES)
                hand_emb = self.tile_emb(hand_idx)                                      # (B, MAX_CONCEALED_TILES, emb)
                hand_emb_2d = hand_emb.unsqueeze(1)                                     # (B,1,MAX_CONCEALED_TILES,emb)
                hand_feat = self.hand_gap(F.relu(self.hand_conv2(F.relu(self.hand_conv1(hand_emb_2d))))).view(B, 64)

                # React vector: use only index at position 0, direct embedding lookup (no processing)
                gri0 = padify(game_tile_indicator_idx[:, 0]).long()                    # (B,)
                react_vec = self.tile_emb(gri0)                                        # (B,emb)

                # Player called tiles: conv path (same style as opponents) -> reduce to 16
                pc_idx = padify(player_called_idx).long()                               # (B,MAX_CALLED_TILES_PER_PLAYER)
                pc_emb = self.tile_emb(pc_idx).unsqueeze(1)                             # (B,1,MAX_CALLED_TILES_PER_PLAYER,emb)
                pc_feat_agg = self.opp_gap(F.relu(self.opp_conv2(F.relu(self.opp_conv1(pc_emb)))))  # (B,64,1,1)
                pc_feat = pc_feat_agg.view(B, 64)
                pc_feat_small = self.player_calls_reduce(pc_feat)                       # (B,16)

                # Opponents called tiles: (B,3,MAX_CALLED_TILES_PER_PLAYER) -> embed -> conv per opponent (shared weights) -> (B, 3*64)
                o_idx = padify(opp_called_idx).long()                                  # (B,3,MAX_CALLED_TILES_PER_PLAYER)
                o_emb = self.tile_emb(o_idx)                                           # (B,3,MAX_CALLED_TILES_PER_PLAYER,emb)
                o_emb = o_emb.view(B * 3, int(MAX_CALLED_TILES_PER_PLAYER), -1).unsqueeze(1)  # (B*3,1,MAX_CALLED_TILES_PER_PLAYER,emb)
                o_feat_agg = self.opp_gap(F.relu(self.opp_conv2(F.relu(self.opp_conv1(o_emb)))))  # (B*3,64,1,1)
                o_feat_agg = o_feat_agg.view(B, 3, 64)                                  # (B,3,64)
                o_feat_small = self.opp_reduce(o_feat_agg)                              # (B,3,16)
                o_feat = o_feat_small.view(B, 3 * 16)                                   # (B,48)

                # Discards: query-based attention with MAX_CONCEALED_TILES queries
                d_idx = padify(disc_idx).long()                                        # (B,NUM_PLAYERS,disc_seq_len)
                d_tile = self.tile_emb(d_idx)                                          # (B,NUM_PLAYERS,disc_seq_len,emb)
                pos_ids = torch.arange(self.outer.disc_seq_len, device=d_idx.device).view(1, 1, -1).expand(B, int(NUM_PLAYERS), -1)
                d_pos = self.disc_pos_emb(pos_ids)                                     # (B,NUM_PLAYERS,disc_seq_len,8)
                ply_ids = torch.arange(int(NUM_PLAYERS), device=d_idx.device).view(1, -1, 1).expand(B, -1, self.outer.disc_seq_len)
                d_ply = self.disc_player_emb(ply_ids)                                  # (B,NUM_PLAYERS,disc_seq_len,4)
                d_kv = torch.cat([d_tile, d_pos, d_ply], dim=-1)                       # (B,NUM_PLAYERS,disc_seq_len,Dkv)
                Dkv = d_kv.shape[-1]
                N = int(NUM_PLAYERS) * self.outer.disc_seq_len
                d_kv = d_kv.view(B, N, Dkv)                                            # (B,N,Dkv)
                d_mask = (d_idx.view(B, N) != self.pad_index)                          # (B,N)

                # Queries from player's hand (concealed draw+13):
                q_idx = padify(player_hand_idx).long()                                 # (B,MAX_CONCEALED_TILES)
                q_emb = self.tile_emb(q_idx)                                           # (B,MAX_CONCEALED_TILES,emb)
                q_proj = self.query_proj(q_emb)                                        # (B,MAX_CONCEALED_TILES,Dkv)

                # Attention: per-query weights over discard keys
                # attn_logits: (B, Q, N) where Q = MAX_CONCEALED_TILES
                attn_logits = torch.matmul(q_proj, d_kv.transpose(1, 2)) / (Dkv ** 0.5)
                # mask PAD tokens in discards
                mask = d_mask.unsqueeze(1)                                             # (B,1,N)
                attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
                # softmax, then zero out masked, and renormalize only if there is at least one valid token
                attn_w = F.softmax(attn_logits, dim=-1)                                # (B,Q,N)
                attn_w = attn_w * mask                                                # zero out masked positions
                denom = attn_w.sum(dim=-1, keepdim=True)                               # (B,Q,1)
                has_valid = denom > 0
                # avoid division by zero: only normalize where sum>0
                attn_w = torch.where(has_valid, attn_w / torch.clamp_min(denom, 1e-12), torch.zeros_like(attn_w))
                # Weighted sum over values -> (B,Q,Dkv), then flatten and reduce to 32-d
                d_per_query = torch.matmul(attn_w, d_kv)                               # (B,MAX_CONCEALED_TILES,Dkv)
                q_flat = d_per_query.reshape(B, -1)                                    # (B, Q*Dkv)
                d_feat = self.query_reduce(q_flat)                                     # (B,32)

                # Game-state features: use the standardized vector directly
                gsv_feat = game_state_vec.float()                                       # (B,GSV)

                # Concatenate all features
                x = torch.cat([hand_feat, pc_feat_small, o_feat, d_feat, react_vec, gsv_feat], dim=1)

                # Process through shared trunk
                trunk_features = self.trunk(x)

                action_logits = self.head_action(trunk_features)
                tile_logits = self.head_tile(trunk_features)
                action_pp = F.softmax(action_logits, dim=-1)
                tile_pp = F.softmax(tile_logits, dim=-1)
                val = self.head_value(trunk_features)
                wall_counts = self.head_wall_counts(trunk_features)
                deal_in = torch.sigmoid(self.head_deal_in(trunk_features))
                return action_pp, tile_pp, val, wall_counts, deal_in

        self._net = _ACModule(self)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net.to(self._device)
        self._net.eval()

        # Note: model now consumes index sequences + flags and performs its own embeddings.

    def _is_fit(self):
        return hasattr(self._gsv_scaler, 'scale_')

    def fit_scaler(self, global_state_arr):
        """Fit scaler on the remaining-tiles feature only (index 0)."""
        assert not self._is_fit(), "gsv_scaler has already been fitted; refusing to refit in ACNetwork"
        g = np.asarray(global_state_arr, dtype=np.float32)
        if g.ndim == 2 and g.shape[1] > 1:
            col0 = g[:, 0:1]
        else:
            col0 = g.reshape(-1, 1)
        assert col0.shape[0] > 0
        self._gsv_scaler.fit(col0)


    def extract_features_from_indexed(
        self,
        *,
        hand_idx: np.ndarray,
        disc_idx: np.ndarray,
        called_idx: np.ndarray,
        game_tile_indicators: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build index-based features for the new architecture.
        Returns:
          - player_hand_idx_fixed: (MAX_CONCEALED_TILES,)
          - player_called_idx: (MAX_CALLED_TILES_PER_PLAYER,)
          - opp_called_idx: (3, MAX_CALLED_TILES_PER_PLAYER)
          - disc_idx_seq: (NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER) most recent discards (pad with -1)
        """
        PAD = -1
        # Flatten called tiles per player
        called = np.asarray(called_idx, dtype=np.int32)
        if called.ndim == 2:
            called_flat = called
        else:
            called_flat = called.reshape(called.shape[0], -1)

        # Player hand (may include padding); keep non-negative
        hand_arr = np.asarray(hand_idx, dtype=np.int32)
        hand_tiles = [int(v) for v in hand_arr.tolist() if int(v) >= 0]
        # Fixed-length concealed hand vector for queries (length MAX_CONCEALED_TILES)
        if len(hand_tiles) < int(MAX_CONCEALED_TILES):
            player_hand_idx_fixed = hand_tiles + ([-1] * (int(MAX_CONCEALED_TILES) - len(hand_tiles)))
        else:
            player_hand_idx_fixed = hand_tiles[: int(MAX_CONCEALED_TILES)]
        # Player called tiles
        p_called_tiles = [int(v) for v in called_flat[0].tolist() if int(v) >= 0]
        # Fixed-length player called vector
        target_len_pc = int(MAX_CALLED_TILES_PER_PLAYER)
        if len(p_called_tiles) < target_len_pc:
            player_called_idx = p_called_tiles + ([PAD] * (target_len_pc - len(p_called_tiles)))
        else:
            player_called_idx = p_called_tiles[: target_len_pc]
        # removed compact hand+calls sequence and flags (not used by the network anymore)

        # Opponents called tiles sequences up to MAX_CALLED_TILES_PER_PLAYER each
        opp_called_list: List[np.ndarray] = []
        for i in range(1, 4):
            tiles = [int(v) for v in called_flat[i].tolist() if int(v) >= 0]
            target_len = int(MAX_CALLED_TILES_PER_PLAYER)
            if len(tiles) < target_len:
                tiles = tiles + [PAD] * (target_len - len(tiles))
            else:
                tiles = tiles[: target_len]
            opp_called_list.append(np.asarray(tiles, dtype=np.int32))
        opp_called_idx = np.stack(opp_called_list, axis=0)

        # Discards: take most recent up to MAX_DISCARDS_PER_PLAYER per player
        disc = np.asarray(disc_idx, dtype=np.int32)
        disc_seq_list: List[np.ndarray] = []
        for i in range(int(NUM_PLAYERS)):
            tiles = [int(v) for v in disc[i].tolist() if int(v) >= 0]
            # take last disc_seq_len
            tiles = tiles[-self.disc_seq_len:]
            if len(tiles) < self.disc_seq_len:
                tiles = ([PAD] * (self.disc_seq_len - len(tiles))) + tiles
            disc_seq_list.append(np.asarray(tiles, dtype=np.int32))
        disc_idx_seq = np.stack(disc_seq_list, axis=0)

        # game_tile_indicators are handled in evaluate() and passed directly to forward()

        return (
            np.asarray(player_hand_idx_fixed, dtype=np.int32),
            np.asarray(player_called_idx, dtype=np.int32),
            opp_called_idx.astype(np.int32),
            disc_idx_seq.astype(np.int32),
        )

    def evaluate(self, game_state: GamePerspective) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (action_head_probs, tile_head_probs, value) for two-head policy.
        """
        self._net.eval()
        feat = encode_game_perspective(game_state)
        # Unpack encoded features
        hand_idx = np.asarray(feat['hand_idx'], dtype=np.int32)
        called_idx = np.asarray(feat['called_idx'], dtype=np.int32)
        disc_idx = np.asarray(feat['disc_idx'], dtype=np.int32)
        # Construct game_tile_indicator_idx = [react, d1, d2, d3, d4]
        react_idx = int(feat.get('reactable_tile', -1))
        dora_vals = [int(v) for v in np.asarray(feat.get('dora_indicator_tiles', []), dtype=np.int32).tolist()]
        while len(dora_vals) < 4:
            dora_vals.append(-1)
        game_tile_indicators = np.asarray([react_idx] + dora_vals[:4], dtype=np.int32)

        # Build fixed-length index features
        (
            player_hand_idx,
            player_called_idx,
            opp_called_idx,
            disc_idx_seq,
        ) = self.extract_features_from_indexed(
            hand_idx=hand_idx,
            disc_idx=disc_idx,
            called_idx=called_idx,
            game_tile_indicators=game_tile_indicators,
        )

        gsv_std = self._game_state_vector(feat)

        with torch.no_grad():
            action_pp, tile_pp, val, _wall_counts, _deal_in = self._net(
                torch.from_numpy(player_hand_idx[None, ...]).to(self._device),
                torch.from_numpy(player_called_idx[None, ...]).to(self._device),
                torch.from_numpy(opp_called_idx[None, ...]).to(self._device),
                torch.from_numpy(disc_idx_seq[None, ...]).to(self._device),
                torch.from_numpy(game_tile_indicators[None, ...]).to(self._device),
                torch.from_numpy(gsv_std[None, ...]).to(self._device),
            )
        return (
            action_pp.cpu().numpy()[0],
            tile_pp.cpu().numpy()[0],
            float(val.cpu().numpy()[0][0]),
        )

    def _game_state_vector(self, feat: Dict[str, Any]) -> np.ndarray:
        # Game-state vector: explicit layout per ac_constants.GAME_STATE_VEC_LEN doc
        from .ac_constants import GAME_STATE_VEC_LEN as GSV
        gsv = np.zeros((int(GSV),), dtype=np.float32)
        # [0] remaining_tiles
        gsv[0] = float(int(feat.get('remaining_tiles', 0)))
        # [1] last_discard_player (owner_of_reactable_tile), -1 if none
        gsv[1] = float(int(feat.get('owner_of_reactable_tile', -1)))
        # [2..5] seat_winds[4] (Honor.value)
        seat_winds = [int(v) for v in np.asarray(feat.get('seat_winds', [0, 0, 0, 0]), dtype=np.int32).tolist()]
        for i in range(4):
            gsv[2 + i] = float(seat_winds[i] if i < len(seat_winds) else 0)
        # [6..9] riichi_decl_idxs[4]
        riichi_decl = [int(v) for v in np.asarray(feat.get('riichi_declarations', [-1, -1, -1, -1]), dtype=np.int32).tolist()]
        for i in range(4):
            gsv[6 + i] = float(riichi_decl[i] if i < len(riichi_decl) else -1)
        # [10..] legal_action_mask[ACTION_HEAD_SIZE]
        lam = np.asarray(feat.get('legal_action_mask', []), dtype=np.int32)
        from .ac_constants import ACTION_HEAD_SIZE
        lam_fixed = np.zeros((int(ACTION_HEAD_SIZE),), dtype=np.float32)
        k = min(int(ACTION_HEAD_SIZE), int(lam.shape[0]))
        lam_fixed[:k] = lam[:k]
        gsv[10:10 + int(ACTION_HEAD_SIZE)] = lam_fixed
        gsv_std = gsv.copy()
        try:
            col0_scaled = self._gsv_scaler.transform(gsv_std[0:1].reshape(-1, 1)).astype(np.float32)[0, 0]
            gsv_std[0] = float(col0_scaled)
        except Exception:
            gsv_std[0] = 0.0
        return gsv_std

    def torch_module(self) -> nn.Module:
        return self._net

    def to(self, device: torch.device) -> 'ACNetwork':
        self._device = device
        self._net.to(device)
        return self

    def load_model(
        self,
        path: str,
        *,
        load_entire_module: bool = False,
        strict: bool = True,
        map_location: Any | None = None,
    ) -> None:
        """Load model weights or an entire serialized module.

        - When load_entire_module is False (default):
          Expects a state_dict (or a dict with key 'state_dict') and loads into the current architecture.
        - When load_entire_module is True: if the file contains a serialized nn.Module, replace the internal
          module with it. This allows swapping in networks with different parameterizations.
        """
        obj = torch.load(path, map_location=(map_location or self._device))
        # Case 1: replace entire module
        if load_entire_module and isinstance(obj, torch.nn.Module):
            self._net = obj
            self._net.to(self._device)
            self._net.eval()
            return
        # Case 2: dict wrapper with state_dict
        state_dict = None
        if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            state_dict = obj['state_dict']
        elif isinstance(obj, dict):
            # Assume it's directly a state_dict
            state_dict = obj
        elif isinstance(obj, torch.nn.Module):
            # Entire module saved but caller did not request replacement; extract its state_dict
            state_dict = obj.state_dict()
        else:
            raise ValueError("Unsupported model file format for ACNetwork.load_model")
        # Load into current architecture
        missing, unexpected = self._net.load_state_dict(state_dict, strict=strict)
        # Move to device and eval
        self._net.to(self._device)
        self._net.eval()
        return

    def save_model(self, path: str, *, save_entire_module: bool = False) -> None:
        """Save the network to a file.

        - When save_entire_module is False (default): saves only state_dict for portability.
        - When True: saves the entire nn.Module (architecture + weights).
        """
        import torch
        if save_entire_module:
            torch.save(self._net, path)
        else:
            torch.save(self._net.state_dict(), path)


