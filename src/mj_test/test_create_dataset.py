import unittest
import os

from core.game import Player, MediumJong
from core.learn.ac_constants import MAX_TURNS
from core.tenpai import hand_is_tenpai_with_calls
from mj_test import test_utils


# Test helpers
def _groups_by_game_and_actor(game_ids, actor_ids):
    groups = {}
    for i in range(len(game_ids)):
        key = (int(game_ids[i]), int(actor_ids[i]))
        groups.setdefault(key, []).append(i)
    return groups


def _decode_gp_from_row(data, row_idx):
    from core.learn.feature_engineering import decode_game_perspective
    features = {
        'hand_idx': data['hand_idx'][row_idx],
        'called_idx': data['called_idx'][row_idx],
        'disc_idx': data['disc_idx'][row_idx],
        'game_state': data['game_state'][row_idx],
        'called_discards': data['called_discards'][row_idx],
    }
    return decode_game_perspective(features)


def _post_action_concealed_and_calls(gp, move):
    from core.game import Discard, Riichi, Chi, Pon, KanDaimin, KanAnkan, KanKakan, CalledSet, Tile
    concealed = list(gp.player_hand)
    called_sets = list(gp.called_sets.get(0, []))
    last = gp.last_discarded_tile

    if isinstance(move, (Discard, Riichi)):
        removed = False
        new_hand = []
        for t in concealed:
            if not removed and t.suit == move.tile.suit and t.tile_type == move.tile.tile_type:
                removed = True
                continue
            new_hand.append(t)
        concealed = new_hand
    elif isinstance(move, Chi) and last is not None:
        for t in move.tiles:
            removed = False
            new_hand = []
            for h in concealed:
                if not removed and h.suit == t.suit and h.tile_type == t.tile_type:
                    removed = True
                    continue
                new_hand.append(h)
            concealed = new_hand
        seq = sorted([move.tiles[0], last, move.tiles[1]], key=lambda t: (t.suit.value, int(t.tile_type.value)))
        called_sets.append(CalledSet(tiles=seq, call_type='chi', called_tile=Tile(last.suit, last.tile_type), caller_position=0, source_position=gp.last_discard_player))
    elif isinstance(move, Pon) and last is not None:
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 2 and t.suit == last.suit and t.tile_type == last.tile_type:
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(3)], call_type='pon', called_tile=Tile(last.suit, last.tile_type), caller_position=0, source_position=gp.last_discard_player))
    elif isinstance(move, KanDaimin) and last is not None:
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 3 and t.suit == last.suit and t.tile_type == last.tile_type:
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(4)], call_type='kan_daimin', called_tile=Tile(last.suit, last.tile_type), caller_position=0, source_position=gp.last_discard_player))
    elif isinstance(move, KanAnkan):
        consumed = 0
        new_hand = []
        for t in concealed:
            if consumed < 4 and t.suit == move.tile.suit and t.tile_type == move.tile.tile_type:
                consumed += 1
                continue
            new_hand.append(t)
        concealed = new_hand
        called_sets.append(CalledSet(tiles=[Tile(move.tile.suit, move.tile.tile_type) for _ in range(4)], call_type='kan_ankan', called_tile=None, caller_position=0, source_position=None))
    elif isinstance(move, KanKakan):
        removed = False
        new_hand = []
        for t in concealed:
            if not removed and t.suit == move.tile.suit and t.tile_type == move.tile.tile_type:
                removed = True
                continue
            new_hand.append(t)
        concealed = new_hand
        for cs in called_sets:
            if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].suit == move.tile.suit and cs.tiles[0].tile_type == move.tile.tile_type:
                cs.call_type = 'kan_kakan'
                cs.tiles.append(move.tile)
                cs.called_tile = None
                cs.source_position = None
                break

    return concealed, called_sets


class TestCreateDataset(unittest.TestCase):
    def test_build_ac_dataset_heuristic_single_game(self):
        from run.create_dataset import build_ac_dataset

        data = build_ac_dataset(
            games=1,
            seed=123,
            use_heuristic=True,
            temperature=0.0,
            n_step=3,
            gamma=0.99,
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
        from core.learn.policy_utils import build_move_from_flat

        random.seed(123)
        np.random.seed(123)

        players = [
            RecordingHeuristicACPlayer(0, random_exploration=0.0),
            RecordingHeuristicACPlayer(1, random_exploration=0.0),
            RecordingHeuristicACPlayer(2, random_exploration=0.0),
            RecordingHeuristicACPlayer(3, random_exploration=0.0),
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
                move = build_move_from_flat(gp, int(act))
                self.assertIsNotNone(move)
                self.assertTrue(gp.is_legal(move))
                total += 1
        self.assertGreater(total, 0)


    def test_verify_rewards(self):
        # Use the dataset builder to compute rewards and export metadata identifying draws.
        from run.create_dataset import build_ac_dataset
        built = build_ac_dataset(
            games=20,
            seed=321,
            use_heuristic=True,
            temperature=0.0,
            n_step=1,
            gamma=1.0,
        )
        # Identify draw games via metadata
        outcomes = built['game_outcomes_obj']
        gids = built['game_ids']
        returns = built['returns']
        actors = built['actor_ids']
        # For each draw game, verify per-player reward sign matches tenpai/noten status
        draw_games = set(int(i) for i, o in enumerate(outcomes) if bool(o['is_draw']))
        eps = 1e-9
        for gid in draw_games:
            outcome = outcomes[gid]
            # Find last step index per actor for this game id
            last_idx_by_actor = {}
            for idx in range(len(gids)):
                if int(gids[idx]) == gid:
                    a = int(actors[idx])
                    last_idx_by_actor[a] = idx
            # Check reward sign per actor vs tenpai
            for pid in range(4):
                if pid not in last_idx_by_actor:
                    continue
                idx = last_idx_by_actor[pid]
                reward = float(returns[idx])
                is_tenpai = bool(outcome['players'][pid]['tenpai']) if outcome.get('players') else False
                if is_tenpai:
                    self.assertGreaterEqual(reward, 0.0 - eps)
                else:
                    self.assertLessEqual(reward, 0.0 + eps)

        # For each non-draw game, verify that any winner (ron or tsumo) has non-negative terminal reward
        win_games = set(int(i) for i, o in enumerate(outcomes) if not bool(o['is_draw']))
        for gid in win_games:
            outcome = outcomes[gid]
            winners = [int(w) for w in outcome.get('winners', [])]
            if not winners:
                continue
            # Last step index per actor for this game id
            last_idx_by_actor = {}
            for idx in range(len(gids)):
                if int(gids[idx]) == gid:
                    a = int(actors[idx])
                    last_idx_by_actor[a] = idx
            for pid in winners:
                if pid not in last_idx_by_actor:
                    continue
                idx = last_idx_by_actor[pid]
                reward = float(returns[idx])
                self.assertGreaterEqual(reward, 0.0 - eps)


if __name__ == '__main__':
    unittest.main(verbosity=2)


