from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

	def __init__(self, player_id: int, network: Any, temperature: float = 1.0):
		super().__init__(player_id)
		self.network = network
		self.temperature = max(1e-6, float(temperature))

	def _mask_to_indices(self, mask: np.ndarray) -> List[int]:
		return [i for i, ok in enumerate(mask) if ok]

	# --- Temperature helper (single-head) ---
	def _temper_and_mask(self, probs: np.ndarray, mask01: np.ndarray) -> np.ndarray:
		"""Apply temperature to entries where mask01==1 and normalize over those; zeros elsewhere.

		- probs: 1D array of raw probabilities
		- mask01: 1D array of zeros/ones (same length)
		"""
		base = np.asarray(probs, dtype=np.float64)
		m = np.asarray(mask01, dtype=np.float64)
		eff = np.zeros_like(base, dtype=np.float64)
		allowed_indices = np.where(m > 0.0)[0]
		if allowed_indices.size > 0:
			inv_temp = 1.0 / max(1e-6, float(self.temperature))
			vals = np.power(np.clip(base[allowed_indices], 0.0, None), inv_temp)
			sumv = float(vals.sum())
			if sumv > 0.0:
				vals = vals / sumv
			else:
				# Fallback to uniform over legal actions to avoid zero-mass distribution
				vals = np.full(allowed_indices.shape[0], 1.0 / float(allowed_indices.size), dtype=np.float64)
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

	# --- Overrides ---
	def play(self, game_state: GamePerspective):
		return self.compute_play(game_state)[0]

	def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
		return self.compute_play(game_state)[0]


