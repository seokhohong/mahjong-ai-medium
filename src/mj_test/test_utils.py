from typing import Dict, List

from core.game import Player, GamePerspective
from core.action import Discard, Reaction, Action, PassCall
from core.tile import Tile, Suit, TileType, Honor


class ForceDiscardPlayer(Player):
    def __init__(self, target: Tile):
        super().__init__()
        self.target = target

    def play(self, gs):  # type: ignore[override]
        if self.target in gs.player_hand:
            return Discard(self.target)
        return super().play(gs)


class ForceActionPlayer(Player):
    def __init__(self, action: Action):
        super().__init__()
        self.action = action

    def play(self, gs):  # type: ignore[override]
        if gs.is_legal(self.action):
            return self.action
        return super().play(gs)

class NoReactionPlayer(Player):
    def __init__(self):
        super().__init__()

    def choose_reaction(self, game_state: GamePerspective, options: List[Reaction]) -> Reaction:  # type: ignore[override]
        return PassCall()


def _make_tenpai_hand():
    # 13 tiles: needs 6s to complete 456s; closed, standard hand
    return [
        Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
        Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
        Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
        Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
    ]


def _make_noten_hand():
    # 13 singles: all 7 honors + three suits' 1 and 9. This should not be in tenpai, assuming we do not have kokushi musou.
    return [
        Tile(Suit.HONORS, Honor.EAST), Tile(Suit.HONORS, Honor.SOUTH), Tile(Suit.HONORS, Honor.WEST),
        Tile(Suit.HONORS, Honor.NORTH), Tile(Suit.HONORS, Honor.WHITE), Tile(Suit.HONORS, Honor.GREEN),
        Tile(Suit.HONORS, Honor.RED),
        Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.NINE),
        Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.NINE),
        Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.NINE),
    ]