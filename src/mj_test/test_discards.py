#!/usr/bin/env python3
import unittest

from mj_test.test_core.test_utils import ForceDiscardPlayer
from core.game import MediumJong
from core.tile import Tile, Suit, Honor


class TestDiscardEvents(unittest.TestCase):
    def test_last_n_discards_per_player_with_forced_discards(self):
        # Choose distinct honor tiles as forced discard targets
        targets = [
            Tile(Suit.HONORS, Honor.EAST),
            Tile(Suit.HONORS, Honor.SOUTH),
            Tile(Suit.HONORS, Honor.WEST),
            Tile(Suit.HONORS, Honor.NORTH),
        ]
        players = [
            ForceDiscardPlayer(targets[0]),
            ForceDiscardPlayer(targets[1]),
            ForceDiscardPlayer(targets[2]),
            ForceDiscardPlayer(targets[3]),
        ]
        # Assign public identifiers as strings 'A','B','C','D'
        pub_ids = ['A', 'B', 'C', 'D']
        for i, p in enumerate(players):
            p.identifier = pub_ids[i]
        g = MediumJong(players)

        # Ensure each player has their target in hand so it will be discarded on their action
        for i in range(4):
            # Replace the first tile to guarantee the target is present
            if g._player_hands[i]:
                g._player_hands[i][0] = targets[i]
            else:
                g._player_hands[i].append(targets[i])

        # Execute four turns: each player should discard their forced target once
        for _ in range(4):
            g.play_turn()

        # Verify last_n_discards_per_player returns the expected last discard (the forced tile)
        last1 = g.last_n_discards_per_player(1)
        for pid in range(4):
            key = pub_ids[pid]
            self.assertIn(key, last1)
            self.assertEqual(len(last1[key]), 1)
            self.assertEqual(last1[key][0], targets[pid])

        # Asking for more than available should still cap at actual count
        last2 = g.last_n_discards_per_player(3)
        for pid in range(4):
            key = pub_ids[pid]
            self.assertEqual(len(last2[key]), 1)
            self.assertEqual(last2[key][0], targets[pid])


if __name__ == '__main__':
    unittest.main()
