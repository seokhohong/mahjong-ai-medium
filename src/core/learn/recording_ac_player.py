from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
from .feature_engineering import encode_game_perspective
from .data_utils import DebugSnapshot

from ..game import (
    Player,
    GamePerspective,
    Tile,
)
from ..action import (
    Tsumo,
    Ron,
    Discard,
    Riichi,
    Pon,
    Chi,
    Reaction,
    PassCall, KanDaimin,
)
from .ac_player import ACPlayer
from ..heuristics_player import MediumHeuristicsPlayer

# Intermediate Rewards
TENPAI_REWARD = 0.1
YAKUHAI_REWARD = 0.05
# Reward for making any call (Chi, Pon, Daiminkan)
CALL_REWARD = 0.02

class ExperienceBuffer:
    """Simple experience buffer for AC training: (encoded_features, action, reward).

    encoded_features is the dict returned by encode_game_perspective.
    """
    def __init__(self) -> None:
        self.states: List[Dict[str, Any]] = []
        # Store two-head indices as tuples (action_idx, tile_idx)
        self.actions: List[Tuple[int, int]] = []
        self.rewards: List[float] = []
        # Optional value baseline per step (e.g., V(s_t) from a network)
        self.values: List[float] = []
        # Stored joint log-prob for chosen move
        self.joint_log_probs: List[float] = []
        # Per-step list of deal-in tiles (as Tile objects) captured from encoded features
        self.deal_in_tiles: List[List[Tile]] = []

    def add(self, state_features: Dict[str, Any], action_indices: Tuple[int, int], reward: float, value: float,
            joint_logp: float,
            raw_state: Optional[Any] = None,
            action_obj: Optional[Any] = None) -> None:
        ai, ti = int(action_indices[0]), int(action_indices[1])
        if ai < 0 or ti < 0:
            # Persist illegal move context for debugging via centralized utility
            DebugSnapshot.save_illegal_move(
                action_index=(ai, ti),
                game_perspective=raw_state,
                action_obj=action_obj,
                encoded_state=state_features,
                value=float(value),
                action_logp=float(joint_logp),
                tile_logp=0.0,
                reason='illegal_action_index',
            )
            return
        self.states.append(state_features)
        self.actions.append((ai, ti))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.joint_log_probs.append(float(joint_logp))
        # Capture deal_in_tiles from encoded features when available
        dit = state_features.get('deal_in_tiles', [])
        if isinstance(dit, list):
            # Ensure we store a shallow-copied list of Tile objects (or empty list)
            self.deal_in_tiles.append(list(dit))
        else:
            self.deal_in_tiles.append([])

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.joint_log_probs.clear()
        self.deal_in_tiles.clear()

    def __len__(self) -> int:
        return len(self.states)


class RecordingACPlayer(ACPlayer):
    """
    Extends ACPlayer to record (state, action, reward) tuples.

    Unlike the pure policy dataset path, we do not store legality masks here since
    they can be recomputed from each stored GamePerspective as needed.
    """

    def __init__(self, network: Any, temperature: float = 1.0, zero_network_reward: bool = False,
                 gsv_scaler: Any = None,
                 exploration_consumption_factor: float = 1.0,
                 low_prob_threshold: float = 0.05,
                 min_temperature: float = 0.1, # gets nans otherwise
                 identifier: Optional[int] = None):
        super().__init__(network=network, gsv_scaler=gsv_scaler, temperature=temperature, identifier=identifier)
        self.experience = ExperienceBuffer()
        self._terminal_reward: Optional[float] = None
        self._zero_network_reward = bool(zero_network_reward)
        self.last_decision_reward: Optional[float] = None
        # Exploration consumption: multiply temperature by this factor when a low-prob move is taken
        self.exploration_consumption_factor = float(max(0.0, min(1.0, exploration_consumption_factor)))
        self.low_prob_threshold = float(max(0.0, low_prob_threshold))
        self.min_temperature = float(max(1e-6, min_temperature))

    def _consume_exploration(self, logp_joint: float) -> None:
        """Reduce temperature after choosing a low-probability move.

        - If `exploration_consumption_factor` < 1.0 and selected prob <= low_prob_threshold,
          set: temperature = max(min_temperature, temperature * exploration_consumption_factor)
        """
        if self.exploration_consumption_factor >= 1.0:
            return
        try:
            prob = float(np.exp(float(logp_joint)))
        except Exception:
            return
        if prob <= self.low_prob_threshold:
            self.temperature = max(self.min_temperature, float(self.temperature) * self.exploration_consumption_factor)

    # Hooks for the engine or controller to assign final rewards when the round ends
    def set_final_reward(self, reward: float) -> None:
        self._terminal_reward = float(reward)

    def finalize_episode(self, terminal_value: float) -> None:
        """Assign a provided terminal reward value to the last recorded step.

        The value should be computed externally (e.g., from points delta or
        other outcome-based heuristic) and passed in.
        """
        if len(self.experience) == 0:
            return
        # Ensure intermediate positive rewards do not exceed the positive terminal value.
        # We rescale only the positive intermediate rewards proportionally so that
        # their sum is clamped to max(0, terminal_value). Negative rewards are kept as-is.
        term_pos_cap = max(0.0, float(terminal_value))
        if len(self.experience.rewards) >= 2:
            interm = self.experience.rewards[:-1]
            sum_pos = sum(r for r in interm if r > 0)
            if sum_pos > term_pos_cap:
                factor = term_pos_cap / sum_pos if sum_pos > 0 else 0.0
                self.experience.rewards[:-1] = [
                    (r * factor) if r > 0 else r for r in interm
                ]
        # Assign terminal reward to the last step
        self.experience.rewards[-1] = float(terminal_value)

    # Record decisions along with value estimates from the network
    def act(self, game_state: GamePerspective):  # type: ignore[override]
        move, value, a_idx, t_idx, logp_joint = self.compute_play(game_state)
        # Consume exploration if the sampled move was low-probability
        self._consume_exploration(logp_joint)
        # Aggregate intermediate rewards (transition to tenpai, yakuhai acquisition, etc.)
        reward = _compute_intermediate_reward(move, game_state)
        self.experience.add(
            encode_game_perspective(game_state),
            (a_idx, t_idx),
            reward,
            float(value if not self._zero_network_reward else 0.0),
            joint_logp=logp_joint,
            raw_state=game_state,
            action_obj=move,
        )
        return move

    def react(self, game_state: GamePerspective, options: List[Reaction]):  # type: ignore[override]
        move, value, a_idx, t_idx, logp_joint = self.compute_play(game_state)
        # Consume exploration if the sampled move was low-probability
        self._consume_exploration(logp_joint)
        reward = _compute_intermediate_reward(move, game_state)
        self.experience.add(
            encode_game_perspective(game_state),
            (a_idx, t_idx),
            reward,
            float(value if not self._zero_network_reward else 0.0),
            joint_logp=logp_joint,
            raw_state=game_state,
            action_obj=move,
        )
        return move


class RecordingHeuristicACPlayer(MediumHeuristicsPlayer):
    """
    Generation-0 recorder using base heuristic policy (no network), recording rewards as 0
    for all decisions except the terminal step, which is set via finalize_episode.
    """

    def __init__(self, random_exploration: float = 0.0, gsv_scaler: Any = None, identifier: Optional[int] = None) -> None:
        super().__init__(identifier=identifier)
        self.random_exploration = max(0.0, float(random_exploration))
        self.experience = ExperienceBuffer()
        # Note: gsv_scaler is not used by heuristic players but kept for API consistency

    def finalize_episode(self, terminal_value: float) -> None:
        if len(self.experience) == 0:
            return
        self.experience.rewards[-1] = float(terminal_value)
        # Keep heuristic value baseline at 0.0 for all steps (including terminal)

    def act(self, game_state: GamePerspective):  # type: ignore[override]
        # With probability = random_exploration, pick a random legal move (prefer discards); otherwise use heuristic strategy
        legal = game_state.legal_moves()
        if self.random_exploration > 0.0 and legal and random.random() < self.random_exploration:
            discards = [m for m in legal if isinstance(m, Discard)]
            move = random.choice(discards or legal)
        else:
            # Delegate to heuristic strategy from MediumHeuristicsPlayer
            move = super().act(game_state)
        assert game_state.is_legal(move)
        # Record encoded state with zero value/logp for heuristic policy (two-head indices)
        from .policy_utils import encode_two_head_action
        ai, ti = encode_two_head_action(move)
        self.experience.add(
            encode_game_perspective(game_state),
            (int(ai), int(ti)),
            0.0,
            0.0,
            joint_logp=0.0,
            raw_state=game_state,
            action_obj=move,
        )
        return move

    def react(self, game_state: GamePerspective, options: List[Reaction]):  # type: ignore[override]
        # With probability = random_exploration, pick a random legal reaction from options
        if self.random_exploration > 0.0:
            legal_reacts: List[Reaction] = []
            if game_state.can_ron():
                legal_reacts.append(Ron())
            legal_reacts.extend([r for r in options if isinstance(r, (Pon, Chi, KanDaimin))])
            legal_reacts.append(PassCall())
            if legal_reacts and random.random() < self.random_exploration:
                move = random.choice(legal_reacts)
                from .policy_utils import encode_two_head_action
                ai, ti = encode_two_head_action(move)
                self.experience.add(
                    encode_game_perspective(game_state),
                    (int(ai), int(ti)),
                    0.0,
                    0.0,
                    joint_logp=0.0,
                )
                return move
        # Otherwise, delegate to heuristic strategy
        move = super().react(game_state, options)
        from .policy_utils import encode_two_head_action
        ai, ti = encode_two_head_action(move)
        self.experience.add(
            encode_game_perspective(game_state),
            (int(ai), int(ti)),
            0.0,
            0.0,
            joint_logp=0.0,
        )
        return move

def _transitions_into_tenpai(move: Any, game_state: GamePerspective) -> bool:
    """Reward only when action transitions the hand into tenpai.

    Conditions:
    - move must be Discard or Riichi (tile-parametrized action)
    - prior_concealed_13 = current concealed hand minus newly_drawn_tile must NOT be tenpai
    - post_concealed_13 = current concealed hand minus move.tile must be tenpai

    Notes:
    - If `newly_drawn_tile` is missing or either removal fails, return False (no reward)
    - For open hands, include `called_sets` in tenpai checks
    """
    if not isinstance(move, (Discard, Riichi)):
        return False
    try:
        from ..tenpai import hand_is_tenpai_for_tiles as _tenpai_closed
        from ..tenpai import hand_is_tenpai as _tenpai_any
    except Exception:
        return False

    hand = list(getattr(game_state, 'player_hand', []))
    if not hand:
        return False
    called_sets = getattr(game_state, 'called_sets', {})
    my_calls = called_sets.get(0, []) if isinstance(called_sets, dict) else []

    def _is_tenpai(tiles_13: List[Tile]) -> bool:
        return _tenpai_any(tiles_13, my_calls) if my_calls else _tenpai_closed(tiles_13)

    # Build prior concealed set: remove newly_drawn_tile (works for closed and open hands)
    last = getattr(game_state, 'newly_drawn_tile', None)
    removed_prior = False
    prior_13: List[Tile] = []
    if last is None:
        # No draw (e.g., after a call); use current concealed set as the prior state
        prior_13 = list(hand)
        removed_prior = True
    else:
        for t in hand:
            if (not removed_prior) and t.exactly_equal(last):
                removed_prior = True
                continue
            prior_13.append(t)
        if not removed_prior:
            return False
    if _is_tenpai(prior_13):
        # Already in tenpai before the action; do not reward holding
        return False

    # Build post concealed set: remove the tile in the chosen move
    target = move.tile
    removed_post = False
    post_13: List[Tile] = []
    for t in hand:
        if (not removed_post) and t.exactly_equal(target):
            removed_post = True
            continue
        post_13.append(t)
    if not removed_post:
        return False
    return _is_tenpai(post_13)


def _is_yakuhai_tile(tile: Tile, game_state: GamePerspective) -> bool:
    from ..tile import Suit, Honor as H
    if tile.suit != Suit.HONORS:
        return False
    # Dragons are always yakuhai
    if tile.tile_type in (H.WHITE, H.GREEN, H.RED):
        return True
    # Seat and round winds are yakuhai
    seat_winds = getattr(game_state, 'seat_winds', {}) or {}
    round_wind = getattr(game_state, 'round_wind', None)
    my_seat = seat_winds.get(0, None) if isinstance(seat_winds, dict) else None
    return tile.tile_type == my_seat or tile.tile_type == round_wind


def _yakuhai_acquired_by_draw(game_state: GamePerspective) -> bool:
    # Reward if newly drawn tile forms (at least) a triplet of yakuhai in concealed hand
    last = getattr(game_state, 'newly_drawn_tile', None)
    if last is None:
        return False
    if not _is_yakuhai_tile(last, game_state):
        return False
    hand = list(getattr(game_state, 'player_hand', []))
    cnt = sum(1 for t in hand if t.functionally_equal(last))
    # If we drew one and now have 3+, we must have had >=2 before draw; treat as acquisition
    return cnt >= 3


def _yakuhai_acquired_by_pon(move: Any, game_state: GamePerspective) -> bool:
    # Reward if a Pon creates a yakuhai triplet
    if not isinstance(move, Pon):
        return False
    tiles = getattr(move, 'tiles', [])
    if len(tiles) != 3:
        return False
    a = tiles[0]
    # Ensure identical tiles
    if not all(t.functionally_equal(a) for t in tiles):
        return False
    return _is_yakuhai_tile(a, game_state)


def _compute_intermediate_reward(move: Any, game_state: GamePerspective) -> float:
    reward = 0.0
    # Tenpai transition on action (discard/riichi), and also valid after calls due to prior-state handling
    if _transitions_into_tenpai(move, game_state):
        reward += float(TENPAI_REWARD)
    # Yakuhai acquisition by draw or by Pon
    if _yakuhai_acquired_by_draw(game_state) or _yakuhai_acquired_by_pon(move, game_state):
        reward += float(YAKUHAI_REWARD)
    # Any call: Chi, Pon, or open Kan
    try:
        from ..action import Chi, Pon, KanDaimin
        if isinstance(move, (Chi, Pon, KanDaimin)):
            reward += float(CALL_REWARD)
    except Exception:
        pass
    return reward


def _action_deals_in(move: Any, game_state: GamePerspective) -> bool:
    """Return True if the chosen action (Discard or Riichi) would deal in.

    Uses aka-insensitive comparison against `game_state.deal_in_tiles` when available.
    """
    try:
        from ..action import Discard, Riichi  # local import to avoid cycles in type checkers
    except Exception:
        return False
    if not isinstance(move, (Discard, Riichi)):
        return False
    dealins = getattr(game_state, 'deal_in_tiles', None)
    if not dealins:
        return False
    target = getattr(move, 'tile', None)
    if target is None:
        return False
    # Aka-insensitive containment check
    for t in dealins:
        if t.functionally_equal(target):
            return True
    return False
