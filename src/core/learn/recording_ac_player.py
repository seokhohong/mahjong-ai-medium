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
    Pon,
    Chi,
    Reaction,
    PassCall, KanDaimin,
)
from .ac_player import ACPlayer
from ..heuristics_player import MediumHeuristicsPlayer


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

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.joint_log_probs.clear()

    def __len__(self) -> int:
        return len(self.states)


class RecordingACPlayer(ACPlayer):
    """
    Extends ACPlayer to record (state, action, reward) tuples.

    Unlike the pure policy dataset path, we do not store legality masks here since
    they can be recomputed from each stored GamePerspective as needed.
    """

    def __init__(self, network: Any, temperature: float = 1.0, zero_network_reward: bool = False,
                 gsv_scaler: Any = None):
        super().__init__(network=network, gsv_scaler=gsv_scaler, temperature=temperature)
        self.experience = ExperienceBuffer()
        self._terminal_reward: Optional[float] = None
        self._zero_network_reward = bool(zero_network_reward)
        self.last_decision_reward: Optional[float] = None

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
        self.experience.rewards[-1] = float(terminal_value)

    # Record decisions along with value estimates from the network
    def play(self, game_state: GamePerspective):  # type: ignore[override]
        move, value, a_idx, t_idx, logp_joint = self.compute_play(game_state)
        self.experience.add(
            encode_game_perspective(game_state),
            (a_idx, t_idx),
            0.0,
            float(value if not self._zero_network_reward else 0.0),
            joint_logp=logp_joint,
            raw_state=game_state,
            action_obj=move,
        )
        return move

    def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]):  # type: ignore[override]
        move, value, a_idx, t_idx, logp_joint = self.compute_play(game_state)
        self.experience.add(
            encode_game_perspective(game_state),
            (a_idx, t_idx),
            0.0,
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

    def __init__(self, random_exploration: float = 0.0, gsv_scaler: Any = None) -> None:
        super().__init__()
        self.random_exploration = max(0.0, float(random_exploration))
        self.experience = ExperienceBuffer()
        # Note: gsv_scaler is not used by heuristic players but kept for API consistency

    def finalize_episode(self, terminal_value: float) -> None:
        if len(self.experience) == 0:
            return
        self.experience.rewards[-1] = float(terminal_value)
        # Keep heuristic value baseline at 0.0 for all steps (including terminal)

    def play(self, game_state: GamePerspective):  # type: ignore[override]
        # With probability = random_exploration, pick a random legal move (prefer discards); otherwise use heuristic strategy
        legal = game_state.legal_moves()
        if self.random_exploration > 0.0 and legal and random.random() < self.random_exploration:
            discards = [m for m in legal if isinstance(m, Discard)]
            move = random.choice(discards or legal)
        else:
            # Delegate to heuristic strategy from MediumHeuristicsPlayer
            move = super().play(game_state)
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

    def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]):  # type: ignore[override]
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
        move = super().choose_reaction(game_state, options)
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

