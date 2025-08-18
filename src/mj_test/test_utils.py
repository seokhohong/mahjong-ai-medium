from typing import Dict, List

from core.game import Discard, Player, Tile, GamePerspective, Reaction, Action, PassCall


class ForceDiscardPlayer(Player):
    def __init__(self, pid, target: Tile):
        super().__init__(pid)
        self.target = target

    def play(self, gs):  # type: ignore[override]
        if self.target in gs.player_hand:
            return Discard(self.target)
        return super().play(gs)


class ForceActionPlayer(Player):
    def __init__(self, pid, action: Action):
        super().__init__(pid)
        self.action = action

    def play(self, gs):  # type: ignore[override]
        if gs.is_legal(self.action):
            return self.action
        return super().play(gs)

class NoReactionPlayer(Player):
    def __init__(self, pid):
        super().__init__(pid)

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:  # type: ignore[override]
        return PassCall()