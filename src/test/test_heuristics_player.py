#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import MediumJong
from core.heuristics_player import MediumHeuristicsPlayer


class TestHeuristicsPlayerGame(unittest.TestCase):
    def test_heuristics_players_complete_a_game(self):
        players = [MediumHeuristicsPlayer(i) for i in range(4)]
        g = MediumJong(players)
        # Run until game over with a safety cap to prevent infinite loops in case of regression
        for _ in range(2000):
            if g.is_game_over():
                break
            g.play_turn()
        self.assertTrue(g.is_game_over(), "Game did not end within the expected number of turns")


if __name__ == '__main__':
    unittest.main(verbosity=2)


