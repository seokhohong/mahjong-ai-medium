from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
	TORCH_AVAILABLE = False

from ..game import GamePerspective, Tile, TileType, Suit
from .ac_constants import (
	MAX_TURNS,
	NUM_PLAYERS as AC_NUM_PLAYERS,
	FLAT_POLICY_SIZE,
	TILE_INDEX_SIZE,
	AC_MAX_CALLED_TILES_PER_PLAYER,
	CALLED_SETS_DEFAULT_SHAPE,
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

	def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = MAX_TURNS, temperature: float = 1.0):
		if not TORCH_AVAILABLE:
			raise ImportError("PyTorch is required for ACNetwork. Please install torch.")
		self.hidden_size = hidden_size
		self.embedding_dim = embedding_dim
		self.max_turns = max_turns
		self.temperature = float(max(0.0, temperature))

		from ..constants import TOTAL_TILES, GAME_STATE_VEC_LEN as GSV
		dealt = 13 * int(AC_NUM_PLAYERS)
		self._max_discards_per_player = max(1, (int(TOTAL_TILES) - dealt) // int(AC_NUM_PLAYERS))
		self._max_called_tiles_per_player = int(AC_MAX_CALLED_TILES_PER_PLAYER)

		conv_ch1, conv_ch2 = 32, 64

		class _ACModule(nn.Module):
			def __init__(self, outer: 'ACNetwork') -> None:
				super().__init__()
				self.outer = outer
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
				self.trunk = nn.Sequential(
					nn.Linear((conv_ch2 * 3) + GSV, outer.hidden_size),
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
				h = self.hand_conv(hand_seq).squeeze(-1)
				c = self.calls_conv(calls_seq).squeeze(-1)
				d = self.disc_conv(disc_seq).squeeze(-1)
				x = torch.cat([h, c, d, gsv], dim=1)
				z = self.trunk(x)
				pp = F.softmax(self.head_policy(z), dim=-1)
				val = self.head_value(z)
				return pp, val

		self._net = _ACModule(self)
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._net.to(self._device)
		self._net.eval()

		# Precomputed embedding table [0..TILE_INDEX_SIZE-1] (0 padding)
		self._embedding_table = np.zeros((int(TILE_INDEX_SIZE), self.embedding_dim), dtype=np.float32)
		for idx in range(1, int(TILE_INDEX_SIZE)):
			rng = np.random.RandomState(seed=idx)
			self._embedding_table[idx] = (rng.randn(self.embedding_dim) * 0.1).astype(np.float32)

	def _get_tile_index(self, tile: Tile) -> int:
		return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)

	def _extract_features_from_indexed(self, hand_idx: np.ndarray, disc_idx: np.ndarray, called_idx: np.ndarray, game_state_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		from ..constants import GAME_STATE_VEC_LEN as GSV
		# Hand embeddings
		hand_idx_safe = np.asarray(hand_idx, dtype=np.int32)
		hand_emb = self._embedding_table[np.clip(hand_idx_safe, 0, int(TILE_INDEX_SIZE) - 1)]
		hand_seq = np.transpose(hand_emb, (1, 0))  # (embed, 12)
		# Called per player -> flatten top N
		called = np.asarray(called_idx, dtype=np.int32)
		if called.ndim != 3:
			called = np.zeros(CALLED_SETS_DEFAULT_SHAPE, dtype=np.int32)
		called_flat = called.reshape(4, -1)
		zero_first = (called_flat == 0).astype(np.int32)
		order = np.argsort(zero_first, axis=1, kind='stable')
		called_reordered = np.take_along_axis(called_flat, order, axis=1)
		called_top = called_reordered[:, : self._max_called_tiles_per_player]
		calls_emb = self._embedding_table[np.clip(called_top, 0, int(TILE_INDEX_SIZE) - 1)]  # (4,N,embed)
		calls_seq = np.transpose(calls_emb.reshape(-1, calls_emb.shape[-1]), (1, 0))  # (embed,4*N)
		# Discards per player -> K and concat
		discs = np.asarray(disc_idx, dtype=np.int32)
		K = self._max_discards_per_player
		maxT = discs.shape[1] if discs.ndim >= 2 else 0
		if maxT >= K:
			disc_slice = discs[:, :K]
		else:
			pad = np.zeros((4, K - maxT), dtype=np.int32)
			disc_slice = np.concatenate([discs, pad], axis=1)
		disc_emb = self._embedding_table[np.clip(disc_slice, 0, int(TILE_INDEX_SIZE) - 1)]  # (4,K,embed)
		disc_seq = np.transpose(disc_emb.reshape(-1, disc_emb.shape[-1]), (1, 0))  # (embed,4*K)
		# Game state vec
		gs = np.asarray(game_state_vec, dtype=np.float32)
		if gs.shape[0] < GSV:
			gs = np.pad(gs, (0, GSV - gs.shape[0]))
		elif gs.shape[0] > GSV:
			gs = gs[:GSV]
		return hand_seq.astype(np.float32), calls_seq.astype(np.float32), disc_seq.astype(np.float32), gs.astype(np.float32)

	def evaluate(self, game_state: GamePerspective) -> Tuple[Dict[str, np.ndarray], float]:
		from ..learn.pure_policy_dataset import serialize_state, extract_indexed_state  # type: ignore
		sd = serialize_state(game_state)
		idx = extract_indexed_state(sd)
		hand_idx = idx['hand_idx']
		disc_idx = idx['disc_idx']
		called_idx = idx.get('called_sets_idx', np.zeros(CALLED_SETS_DEFAULT_SHAPE, dtype=np.int32))
		game_state_vec = idx['game_state']
		h, c, d, g = self._extract_features_from_indexed(hand_idx, disc_idx, called_idx, game_state_vec)
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
		if not TORCH_AVAILABLE:
			raise ImportError("PyTorch is required for ACNetwork.load_model. Please install torch.")
		import torch  # local import for consistency with environment
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
		if not TORCH_AVAILABLE:
			raise ImportError("PyTorch is required for ACNetwork.save_model. Please install torch.")
		import torch
		if save_entire_module:
			torch.save(self._net, path)
		else:
			torch.save(self._net.state_dict(), path)


