from __future__ import annotations

from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

import numpy as np

# Torch is installed in the project's Python environment (.venv312). Import directly.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game import GamePerspective, Tile, TileType, Suit
from .feature_engineering import encode_game_perspective
from .ac_constants import (
    NUM_PLAYERS as AC_NUM_PLAYERS,
    FLAT_POLICY_SIZE,
    TILE_INDEX_SIZE,
    MAX_CALLS,
    MAX_CALLED_SET_SIZE,
)


class ACNetwork:
    """
    Actor-Critic network for MediumJong with shared trunk and multiple heads:
    - Policy main-head over actions: [Chi, Pon, Ron, Tsumo, Discard, Pass] (size=6)
    - Policy tile-head: one-hot over 18 tile indices (size=18)
    - Policy chi-range-head: 3-class (low, medium, high) (size=3)
    - Value head: real-valued state value (R)

    All heads share the same feature extractor from inputs to the pre-head layer.
    """

    def __init__(self, gsv_scaler: StandardScaler | None, hidden_size: int = 128, embedding_dim: int = 4,
                 temperature: float = 1.0):
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.temperature = float(max(0.0, temperature))
        if gsv_scaler is None:
            print("Warning, instantiating ACNetwork with null StandardScaler")
            self._gsv_scaler = StandardScaler()
        else:
            # Respect provided scaler (may already be fit from dataset)
            self._gsv_scaler = gsv_scaler

        from ..constants import TOTAL_TILES
        from .ac_constants import GAME_STATE_VEC_LEN as GSV
        dealt = 13 * int(AC_NUM_PLAYERS)
        self._max_discards_per_player = max(1, (int(TOTAL_TILES) - dealt) // int(AC_NUM_PLAYERS))
        self._max_called_sets = int(MAX_CALLS)
        self._max_tiles_per_called_set = int(MAX_CALLED_SET_SIZE)

        conv_ch1, conv_ch2 = 32, 64

        class _ACModule(nn.Module):
            def __init__(self, outer: 'ACNetwork') -> None:
                super().__init__()
                self.outer = outer
                num_p = int(AC_NUM_PLAYERS)
                # Convolutional towers
                self.hand_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.calls_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.disc_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                # Shared trunk
                input_dim = (conv_ch2 * (1 + 2 * num_p)) + GSV
                self.trunk = nn.Sequential(
                    nn.Linear(input_dim, outer.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(outer.hidden_size, outer.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                # Heads: single flat policy sized per constants
                self.head_policy = nn.Linear(outer.hidden_size // 2, int(FLAT_POLICY_SIZE))
                self.head_value = nn.Linear(outer.hidden_size // 2, 1)

            def forward(self, hand_seq: torch.Tensor, calls_seq: torch.Tensor, disc_seq: torch.Tensor, gsv: torch.Tensor):
                # hand_seq: (batch_size, embedding_dim, hand_len)
                hand_features = self.hand_conv(hand_seq).squeeze(-1)
                # calls_seq: (batch_size, num_players, embedding_dim, flat_calls_len)
                batch_size, num_players, embedding_dim, flat_calls_len = calls_seq.shape
                calls_flat = calls_seq.reshape(batch_size * num_players, embedding_dim, flat_calls_len)
                calls_features_per_player = self.calls_conv(calls_flat).squeeze(-1)  # (batch_size*num_players, conv_ch2)
                calls_features = calls_features_per_player.reshape(batch_size, num_players * calls_features_per_player.shape[1])
                # disc_seq: (batch_size, num_players, embedding_dim, max_discards_per_player)
                _, _, embedding_dim_disc, max_discards_per_player = disc_seq.shape
                disc_flat = disc_seq.reshape(batch_size * num_players, embedding_dim_disc, max_discards_per_player)
                disc_features_per_player = self.disc_conv(disc_flat).squeeze(-1)
                disc_features = disc_features_per_player.reshape(batch_size, num_players * disc_features_per_player.shape[1])
                x = torch.cat([hand_features, calls_features, disc_features, gsv], dim=1)
                z = self.trunk(x)
                pp = F.softmax(self.head_policy(z), dim=-1)
                val = self.head_value(z)
                return pp, val

        self._net = _ACModule(self)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net.to(self._device)
        self._net.eval()

        # Precomputed embedding table [0..TILE_INDEX_SIZE-1];
        # index < 0 (PAD) will be masked to zeros after lookup
        self._embedding_table = np.zeros((int(TILE_INDEX_SIZE), self.embedding_dim), dtype=np.float32)
        for idx in range(0, int(TILE_INDEX_SIZE)):
            rng = np.random.RandomState(seed=idx + 1337)
            self._embedding_table[idx] = (rng.randn(self.embedding_dim) * 0.1).astype(np.float32)

    def _get_tile_index(self, tile: Tile) -> int:
        return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)

    def _is_fit(self):
        return hasattr(self._gsv_scaler, 'scale_')

    def fit_scaler(self, game_state_arr):
        assert not self._is_fit(), "gsv_scaler has already been fitted; refusing to refit in ACNetwork"
        # Ensure StandardScaler sees a consistent feature size matching AC constants
        from core.learn.ac_constants import GAME_STATE_VEC_LEN as AC_GSV_LEN  # type: ignore
        gsv_arr = np.asarray(game_state_arr, dtype=np.float32)
        assert gsv_arr.shape[0] > 0
        if gsv_arr.ndim == 1:
            gsv_arr = gsv_arr[None, :]
        pad_width = int(AC_GSV_LEN) - gsv_arr.shape[1]
        if pad_width > 0:
            gsv_arr = np.pad(gsv_arr, ((0, 0), (0, pad_width)), mode='constant')
        self._gsv_scaler.fit(gsv_arr)


    def extract_features_from_indexed(self, hand_idx: np.ndarray, disc_idx: np.ndarray, called_idx: np.ndarray, game_state_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._is_fit, "StandardScaler is not fit"
        from .ac_constants import GAME_STATE_VEC_LEN as GSV
        # Hand embeddings (index < 0 is padding and will be zeroed). We assume inputs are already fixed-size.
        hand_idx_safe = np.asarray(hand_idx, dtype=np.int32)
        valid_mask = (hand_idx_safe >= 0) #need the >=0 to avoid padding
        hand_emb = np.zeros((hand_idx_safe.shape[0], self.embedding_dim), dtype=np.float32)
        hand_emb[valid_mask] = self._embedding_table[hand_idx_safe[valid_mask]]
        hand_seq = np.transpose(hand_emb, (1, 0))  # (embed, hand_len)
        # Called per player now structured as (num_players, max_calls, tiles_per_set)
        called = np.asarray(called_idx, dtype=np.int32)
        if called.ndim == 2:
            # Legacy: flatten called tiles per player -> reshape to (max_calls, tiles_per_set)
            num_players, flat_len = called.shape
            mc = int(self._max_called_sets)
            ts = int(self._max_tiles_per_called_set)
            pad = mc * ts
            if flat_len < pad:
                pad_width = pad - flat_len
                called = np.pad(called, ((0,0),(0,pad_width)), constant_values=-1)
            called = called.reshape(num_players, mc, ts)
        # Embed called tiles
        valid_called = (called >= 0)
        calls_emb = np.zeros((called.shape[0], called.shape[1], called.shape[2], self.embedding_dim), dtype=np.float32)
        calls_emb[valid_called] = self._embedding_table[called[valid_called]]
        # Rearrange to (num_players, embed, max_calls * tiles_per_set)
        calls_seq = calls_emb.reshape(called.shape[0], called.shape[1] * called.shape[2], self.embedding_dim)
        calls_seq = np.transpose(calls_seq, (0, 2, 1))  # (4, embed, max_calls*tiles_per_set)
        # Discards per player (shape (4, max_discards)) already aligned from serialization
        discs = np.asarray(disc_idx, dtype=np.int32)
        valid_disc = (discs >= 0)
        disc_emb = np.zeros((discs.shape[0], discs.shape[1], self.embedding_dim), dtype=np.float32)
        disc_emb[valid_disc] = self._embedding_table[discs[valid_disc]]
        disc_seq = np.transpose(disc_emb, (0, 2, 1))  # (4, embed, max_discards)
        # Game state vec (already a flat float vector from serialization pass)
        gs = np.asarray(game_state_vec, dtype=np.float32)
        if gs.shape[0] < GSV:
            gs = np.pad(gs, (0, GSV - gs.shape[0]))
        gs = self._gsv_scaler.transform([gs])[0]
        return hand_seq.astype(np.float32), calls_seq.astype(np.float32), disc_seq.astype(np.float32), gs.astype(np.float32)

    def evaluate(self, game_state: GamePerspective) -> Tuple[Dict[str, np.ndarray], float]:
        # not sure if this is slow, but we need to set it to eval mode before inference
        self._net.eval()
        # Use the feature engineering exporter directly; ignore called_discards for now.
        feat = encode_game_perspective(game_state)
        hand_idx = feat['hand_idx']            # (14,)
        called_idx = feat['called_idx']        # (4, max_called)
        disc_idx = feat['disc_idx']            # (4, max_discards)
        game_state_vec = feat['game_state']    # (GSV',)

        h, c, d, g = self.extract_features_from_indexed(hand_idx, disc_idx, called_idx, game_state_vec)

        with torch.no_grad():
            pp, val = self._net(
                torch.from_numpy(h[None, ...]).to(self._device),
                torch.from_numpy(c[None, ...]).to(self._device),
                torch.from_numpy(d[None, ...]).to(self._device),
                torch.from_numpy(g[None, ...]).to(self._device),
            )
        policy = pp.cpu().numpy()[0]
        value = float(val.cpu().numpy()[0][0])
        return policy, value

    @property
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


