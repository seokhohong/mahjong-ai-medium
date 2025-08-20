from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Self
from sklearn.preprocessing import StandardScaler
import os
import pickle

import numpy as np

from ..game import (
	Player,
	GamePerspective,
	Tile,
	Tsumo,
	Ron,
	Discard,
	Pon,
	Chi,
	Reaction,
	PassCall,
)
from .ac_network import ACNetwork
from ..game import MediumJong
from .ac_constants import chi_variant_index
from .policy_utils import build_move_from_flat


class ACPlayer(Player):
	"""
	Actor-Critic player using `ACNetwork` outputs.
	Selection is hierarchical:
	1) Choose main action from policy main-head with temperature.
	2) If action needs tile parameter, select from tile-head.
	3) If action is chi, select chi range from chi-head.

	No decision needs both a tile and a chi-range simultaneously.
	"""

	def __init__(self, player_id: int, network: Any, gsv_scaler: StandardScaler | None = None,
                 temperature: float = 1.0):
		super().__init__(player_id)
		self.network = network
		self.temperature = max(1e-6, float(temperature))
		self.gsv_scaler = gsv_scaler

	def _mask_to_indices(self, mask: np.ndarray) -> List[int]:
		return [i for i, ok in enumerate(mask) if ok]

	# --- Temperature helper (single-head) ---
	def _temper_and_mask(self, probs: np.ndarray, mask01: np.ndarray) -> np.ndarray:
		"""Apply temperature to entries where mask01==1 and normalize over those; zeros elsewhere.

        - probs: 1D array of raw probabilities
        - mask01: 1D array of zeros/ones (same length)

        Temperature behavior (standard convention):
        - temperature < 1.0: sharper distribution (more deterministic)
        - temperature = 1.0: no change
        - temperature > 1.0: flatter distribution (more exploratory)
        """
		base = np.asarray(probs, dtype=np.float64)
		m = np.asarray(mask01, dtype=np.float64)
		eff = np.zeros_like(base, dtype=np.float64)
		allowed_indices = np.where(m > 0.0)[0]

		if allowed_indices.size > 0:
			# Convert probabilities to log probabilities
			log_probs = np.log(np.clip(base[allowed_indices], 1e-10, None))

			# Apply temperature scaling (standard convention)
			temp = max(1e-6, float(self.temperature))
			scaled_log_probs = log_probs / temp

			# Convert back to probabilities via softmax
			# Subtract max for numerical stability
			scaled_log_probs = scaled_log_probs - np.max(scaled_log_probs)
			exp_vals = np.exp(scaled_log_probs)
			sumv = float(exp_vals.sum())
			vals = exp_vals / sumv

			eff[allowed_indices] = vals.astype(np.float64)

		return eff


	# --- Core selection ---
	def compute_play(self, gs: GamePerspective) -> Tuple[Any, float, np.ndarray]:
		"""Evaluate once and return (move, value, log_policy) over 25 actions."""
		policy, value = self.network.evaluate(gs)
		flat = np.asarray(policy, dtype=np.float64)
		mask = gs.legal_flat_mask_np()
		eff_policy = self._temper_and_mask(flat, mask)
		move = np.argmax(eff_policy)
		# Avoid log(0) by clipping for logging
		log_policy = np.log(np.clip(eff_policy, 1e-12, None))
		return build_move_from_flat(gs, int(move)), float(value), log_policy

	def _select_action(self, gs: GamePerspective) -> Optional[Any]:
		move, _, _ = self.compute_play(gs)
		return move

	@classmethod
	def from_directory(cls, model_dir: str, player_id: int = 0, temperature: float = 1.0) -> Self:
		"""Load an ACPlayer from a directory containing model.pt and scaler.pkl files.

		Args:
			model_dir: Path to directory containing model.pt and scaler.pkl
			player_id: Player ID for the ACPlayer
			temperature: Temperature for action selection

		Returns:
			ACPlayer instance with loaded model and scaler

		Raises:
			FileNotFoundError: If required files are missing
			ValueError: If files are corrupted or incompatible
		"""
		# Look for model files in the directory
		model_path = None
		scaler_path = None

		if os.path.isfile(model_dir):
			# If model_dir is actually a file, treat it as the model path
			model_path = model_dir
			# Look for scaler in the same directory
			scaler_path = os.path.join(os.path.dirname(model_dir), 'scaler.pkl')
		else:
			# model_dir is a directory
			for filename in os.listdir(model_dir):
				if filename.endswith('.pt'):
					model_path = os.path.join(model_dir, filename)
				elif filename.endswith('.pkl'):
					scaler_path = os.path.join(model_dir, filename)

		if model_path is None:
			raise FileNotFoundError(f"No .pt model file found in {model_dir}")

		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model file not found: {model_path}")

		# Load the scaler (optional, will use default if not found)
		gsv_scaler = None
		if scaler_path and os.path.exists(scaler_path):
			try:
				with open(scaler_path, 'rb') as f:
					gsv_scaler = pickle.load(f)
				print(f"Loaded StandardScaler from {scaler_path}")
			except Exception as e:
				print(f"Warning: Could not load scaler from {scaler_path}: {e}")
				print("Using default StandardScaler")
				gsv_scaler = StandardScaler()
		else:
			print(f"Warning: No scaler.pkl found in {model_dir}, using default StandardScaler")
			gsv_scaler = StandardScaler()

		# First try to load the entire model (preserves original architecture)
		try:
			# Create a minimal ACNetwork to use as a loader
			temp_network = ACNetwork(gsv_scaler=gsv_scaler)
			temp_network.load_model(model_path, load_entire_module=True)
			network = temp_network
			print(f"Loaded model with original architecture from {model_path}")
		except Exception as e:
			print(f"Warning: Could not load as complete module ({e}), trying to load weights into matching architecture if possible...")
			# Fall back: read state_dict and infer architecture
			try:
				obj = __import__('torch').load(model_path, map_location='cpu')
				if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
					state_dict = obj['state_dict']
				elif isinstance(obj, dict):
					state_dict = obj
				elif isinstance(obj, __import__('torch').nn.Module):
					state_dict = obj.state_dict()
				else:
					raise ValueError('Unsupported model file format')
				# Infer embedding_dim from first conv in-channels; infer hidden_size from first trunk layer out_features
				embed_dim = int(state_dict.get('hand_conv.0.weight').shape[1]) if 'hand_conv.0.weight' in state_dict else 4
				hidden_size = int(state_dict.get('trunk.0.weight').shape[0]) if 'trunk.0.weight' in state_dict else 128
				# Build network with inferred sizes and load weights non-strictly
				network = ACNetwork(gsv_scaler=gsv_scaler, hidden_size=hidden_size, embedding_dim=embed_dim)
				missing, unexpected = network.torch_module.load_state_dict(state_dict, strict=False)
				# Move to device and eval
				network.to(network._device)
				network.torch_module.eval()
				print(f"Loaded model weights into inferred architecture (hidden_size={hidden_size}, embedding_dim={embed_dim}) from {model_path}")
			except Exception as e2:
				raise ValueError(f"Failed to load model from {model_path}: {e2}")

		# Create and return ACPlayer; attach loaded scaler so downstream save routines can persist it
		return cls(player_id=player_id, network=network, gsv_scaler=gsv_scaler, temperature=temperature)

	# Example usage:
	# # Load a trained ACPlayer from a model directory
	# player = ACPlayer.from_directory('models/ac_ppo_20241220_143022', player_id=0, temperature=0.1)
	#
	# # Use the player in a game
	# move = player.play(game_state)

	# --- Overrides ---
	def play(self, game_state: GamePerspective):
		return self.compute_play(game_state)[0]

	def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
		return self.compute_play(game_state)[0]


