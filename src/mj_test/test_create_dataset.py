import unittest
import os

from core.game import Player, MediumJong
from core.learn.ac_constants import MAX_TURNS
from core.learn.recording_ac_player import RecordingHeuristicACPlayer
from core.tenpai import hand_is_tenpai_with_calls
from mj_test import test_utils


# Test helpers
def _groups_by_game_and_actor(game_ids, actor_ids):
    groups = {}
    for i in range(len(game_ids)):
        key = (int(game_ids[i]), int(actor_ids[i]))
        groups.setdefault(key, []).append(i)
    return groups


def _post_action_concealed_and_calls(gp, move):
    from core.action import Discard, Riichi, Chi, Pon, KanDaimin, KanAnkan, KanKakan
    from core.game import CalledSet
    from core.tile import Tile
    concealed = list(gp.player_hand)
    called_sets = list(gp.called_sets.get(0, []))
    last = gp._reactable_tile

    if isinstance(move, (Discard, Riichi)):
        removed = False
        new_hand = []
        for t in concealed:
            if not removed and t.exactly_equal(move.tile):
                removed = True
                continue
            new_hand.append(t)
        concealed = new_hand
    elif isinstance(move, Chi) and last is not None:
        for t in move.tiles:
            removed = False
            new_hand = []
            for h in concealed:
                if not removed and h.exactly_equal(t):
                    removed = True
                    continue
                new_hand.append(h)
            concealed = new_hand
        seq = sorted([move.tiles[0], last, move.tiles[1]], key=lambda t: (t.suit.value, int(t.tile_type.value)))
        # Preserve aka by using the actual tile instances
        called_sets.append(CalledSet(tiles=seq, call_type='chi', called_tile=last, caller_position=0, source_position=gp._owner_of_reactable_tile))
    elif isinstance(move, Pon) and last is not None:
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 2 and t.functionally_equal(last):
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[last, last, last], call_type='pon', called_tile=last, caller_position=0, source_position=gp._owner_of_reactable_tile))
    elif isinstance(move, KanDaimin) and last is not None:
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 3 and t.functionally_equal(last):
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[last, last, last, last], call_type='kan_daimin', called_tile=last, caller_position=0, source_position=gp._owner_of_reactable_tile))
    elif isinstance(move, KanAnkan):
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 4 and t.functionally_equal(move.tile):
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[move.tile, move.tile, move.tile, move.tile], call_type='kan_ankan', called_tile=None, caller_position=0, source_position=None))
    elif isinstance(move, KanKakan):
        removed = False
        new_hand = []
        for t in concealed:
            if not removed and t.exactly_equal(move.tile):
                removed = True
                continue
            new_hand.append(t)
        concealed = new_hand
        for cs in called_sets:
            if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].functionally_equal(move.tile):
                cs.call_type = 'kan_kakan'
                cs.tiles.append(move.tile)
                cs.called_tile = None
                cs.source_position = None
                break

    return concealed, called_sets


class TestCreateDataset(unittest.TestCase):
    def test_build_ac_dataset_heuristic_single_game(self):
        from run.create_dataset_parallel import build_ac_dataset

        players = [
            RecordingHeuristicACPlayer()
        ] * 4

        data = build_ac_dataset(
            games=2,
            seed=123,
            n_step=3,
            gamma=0.99,
            prebuilt_players=players
        )
        self.assertTrue("called_discards" in data)
        # just test that this runs

    def test_records_legal(self):
        # Play a game with recording heuristic players and confirm recorded (state, action) are legal
        import random
        import numpy as np
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from core.game import MediumJong
        from core.learn.recording_ac_player import RecordingHeuristicACPlayer
        from core.learn.feature_engineering import decode_game_perspective
        from core.learn.policy_utils import build_move_from_two_head

        random.seed(123)
        np.random.seed(123)

        players = [
            RecordingHeuristicACPlayer(random_exploration=0.0),
            RecordingHeuristicACPlayer(random_exploration=0.0),
            RecordingHeuristicACPlayer(random_exploration=0.0),
            RecordingHeuristicACPlayer(random_exploration=0.0),
        ]
        g = MediumJong(players)

        steps = 0
        while not g.is_game_over() and steps < MAX_TURNS:
            g.play_turn()
            steps += 1

        # Aggregate experiences from all players
        total = 0
        for p in players:
            self.assertEqual(len(p.experience.states), len(p.experience.actions))
            for st, act in zip(p.experience.states, p.experience.actions):
                gp = decode_game_perspective(st)
                # act is a tuple (action_idx, tile_idx)
                move = build_move_from_two_head(gp, int(act[0]), int(act[1]))
                self.assertIsNotNone(move)
                if not gp.is_legal(move):
                    print(move, gp)
                self.assertTrue(gp.is_legal(move))
                total += 1
        self.assertGreater(total, 0)

    def test_parallel_two_games_two_procs_combines_samples(self):
        """Run 2 games across 2 processes and verify combined per-sample fields length > 2.
        This ensures per-sample arrays are concatenated by sample, not by game.
        """
        import numpy as np
        from run.create_dataset_parallel import create_dataset_parallel

        out_path = create_dataset_parallel(
            games=4,
            num_processes=2,
            seed=123,
            n_step=1,
            gamma=0.99,
            out='ac_parallel_test_tmp.npz',
            chunk_size=2,
            keep_partials=False,
            stream_combine=True
        )
        try:
            data = np.load(out_path, allow_pickle=True)
            # Pick a few representative per-sample fields
            self.assertGreater(len(data['called_idx']), 4)
            self.assertGreater(len(data['action_idx']), 4)
            self.assertGreater(len(data['called_discards']), 4)
            self.assertGreater(len(data['riichi_declarations']), 4)
            self.assertGreater(len(data['deal_in_tiles']), 4)
            self.assertGreater(len(data['wall_count']), 4)
            data.close()
        finally:
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
            except Exception:
                pass



if __name__ == '__main__':
    unittest.main(verbosity=2)


