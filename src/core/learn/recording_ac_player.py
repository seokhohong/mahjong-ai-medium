from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
from .policy_utils import flat_index_for_action
from .feature_engineering import encode_game_perspective

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
from .ac_player import ACPlayer
from ..heuristics_player import MediumHeuristicsPlayer


class ExperienceBuffer:
    """Simple experience buffer for AC training: (encoded_features, action, reward).

    encoded_features is the dict returned by encode_game_perspective.
    """
    def __init__(self) -> None:
        self.states: List[Dict[str, Any]] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        # Optional value baseline per step (e.g., V(s_t) from a network)
        self.values: List[float] = []
        # Stored log-prob for the chosen flat action, and full flat policy for analysis
        self.main_log_probs: List[float] = []
        self.main_probs: List[List[float]] = []

    def add(self, state_features: Dict[str, Any], action: Any, reward: float, value: float,
            main_logp: float,
            main_probs: Optional[List[float]] = None) -> None:
        self.states.append(state_features)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.main_log_probs.append(float(main_logp))
        self.main_probs.append(list(main_probs) if main_probs is not None else [])

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.main_log_probs.clear()
        self.main_probs.clear()

    def __len__(self) -> int:
        return len(self.states)


class RecordingACPlayer(ACPlayer):
    """
    Extends ACPlayer to record (state, action, reward) tuples.

    Unlike the pure policy dataset path, we do not store legality masks here since
    they can be recomputed from each stored GamePerspective as needed.
    """

    def __init__(self, player_id: int, network: Any, temperature: float = 1.0, zero_network_reward: bool = False):
        super().__init__(player_id, network, temperature=temperature)
        self.experience = ExperienceBuffer()
        self._terminal_reward: Optional[float] = None
        self._zero_network_reward = bool(zero_network_reward)
        self.last_decision_reward: Optional[float] = None

    # Hooks for the engine or controller to assign final rewards when the round ends
    def set_final_reward(self, reward: float) -> None:
        self._terminal_reward = float(reward)

    def finalize_episode(self, winner_ids: List[int], loser_id: Optional[int]) -> None:
        """Assign terminal reward to the last recorded step based on actual outcome.

        This should be called once after the game ends (including draw/keiten). It overwrites
        the last step's stored reward with +1 for winners, -1 for the single loser (if any),
        or 0 otherwise.
        """
        if len(self.experience) == 0:
            return
        terminal = 0.0
        if self.player_id in (winner_ids or []):
            terminal = 1.0
        elif loser_id is not None and int(loser_id) == int(self.player_id):
            terminal = -1.0
        self.experience.rewards[-1] = float(terminal)

    # Record decisions along with value estimates from the network
    def play(self, game_state: GamePerspective):  # type: ignore[override]
        move, value, log_policy = self.compute_play(game_state)
        # Build minimal state/action dicts for flat index mapping
        flat_idx = int(flat_index_for_action(game_state, move))
        main_logp = float(log_policy[flat_idx]) if 0 <= flat_idx < log_policy.size else 0.0
        self.experience.add(
            encode_game_perspective(game_state),
            move,
            0.0,
            float(value if not self._zero_network_reward else 0.0),
            main_logp=main_logp,
            main_probs=np.exp(log_policy).tolist(),
        )
        return move

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]):  # type: ignore[override]
        move, value, log_policy = self.compute_play(game_state)
        # Build minimal state/action dicts for flat index mapping
        flat_idx = int(flat_index_for_action(game_state, move))
        main_logp = float(log_policy[flat_idx]) if 0 <= flat_idx < log_policy.size else 0.0
        self.experience.add(
            encode_game_perspective(game_state),
            move,
            0.0,
            float(value if not self._zero_network_reward else 0.0),
            main_logp=main_logp,
            main_probs=np.exp(log_policy).tolist(),
        )
        return move


class RecordingHeuristicACPlayer(MediumHeuristicsPlayer):
    """
    Generation-0 recorder using base heuristic policy (no network), recording rewards as 0
    for all decisions except the terminal step, which is set via finalize_episode.
    """

    def __init__(self, player_id: int, temperature: float = 0.0) -> None:
        super().__init__(player_id)
        self.temperature = max(0.0, float(temperature))
        self.experience = ExperienceBuffer()

    def finalize_episode(self, winner_ids: List[int], loser_id: Optional[int]) -> None:
        if len(self.experience) == 0:
            return
        terminal = 0.0
        if self.player_id in (winner_ids or []):
            terminal = 1.0
        elif loser_id is not None and int(loser_id) == int(self.player_id):
            terminal = -1.0
        self.experience.rewards[-1] = float(terminal)
        # Keep heuristic value baseline at 0.0 for all steps (including terminal)

    def play(self, game_state: GamePerspective):  # type: ignore[override]
        # With probability = temperature, pick a random legal move (prefer discards); otherwise use heuristic strategy
        legal = game_state.legal_moves()
        if self.temperature > 0.0 and legal and random.random() < self.temperature:
            discards = [m for m in legal if isinstance(m, Discard)]
            move = random.choice(discards or legal)
        else:
            # Delegate to heuristic strategy from MediumHeuristicsPlayer
            move = super().play(game_state)
        # Record encoded state with zero value/logp for heuristic policy
        self.experience.add(
            encode_game_perspective(game_state),
            move,
            0.0,
            0.0,
            main_logp=0.0,
            main_probs=None,
        )
        return move

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]):  # type: ignore[override]
        # With probability = temperature, pick a random legal reaction from options
        if self.temperature > 0.0:
            legal_reacts: List[Reaction] = []
            if game_state.can_ron():
                legal_reacts.append(Ron())
            for tiles in options.get('pon', []):
                legal_reacts.append(Pon(tiles))
            for tiles in options.get('chi', []):
                legal_reacts.append(Chi(tiles))
            legal_reacts.append(PassCall())
            if legal_reacts and random.random() < self.temperature:
                move = random.choice(legal_reacts)
                self.experience.add(
                    encode_game_perspective(game_state),
                    move,
                    0.0,
                    0.0,
                    main_logp=0.0,
                    main_probs=None,
                )
                return move
        # Otherwise, delegate to heuristic strategy
        move = super().choose_reaction(game_state, options)
        self.experience.add(
            encode_game_perspective(game_state),
            move,
            0.0,
            0.0,
            main_logp=0.0,
            main_probs=None,
        )
        return move

