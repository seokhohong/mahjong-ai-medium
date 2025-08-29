from __future__ import annotations

from typing import List, Dict, Any
import argparse
import numpy as np

from core.learn.feature_engineering import decode_game_perspective
from core.learn.policy_utils import build_move_from_two_head
from core.learn.data_utils import build_state_from_npz_row, make_npz_state_row_getter
from core.game import GameOutcome
from core.tile import Suit


def inspect_data(states: List[Dict[str, Any]], action_pairs: List[tuple[int, int]], *, include_winds: bool = True, actor_id: int | None = None) -> List[str]:
    """Pretty-print entries using two-head actions (action_idx, tile_idx)."""
    lines: List[str] = []

    def _fmt_tile(t) -> str:
        try:
            return str(t)
        except Exception:
            return f"{t}"

    def _infer_call_type(tiles: list) -> str:
        try:
            if not tiles:
                return 'set'
            # All identical: pon (3) or kan (4)
            if all(t.suit == tiles[0].suit and t.tile_type == tiles[0].tile_type for t in tiles):
                return 'kan' if len(tiles) == 4 else 'pon'
            # Suited run (sequence): chi
            suits = {t.suit for t in tiles}
            if len(suits) == 1 and tiles[0].suit != Suit.HONORS:
                vals = sorted(int(t.tile_type.value) for t in tiles)
                if len(vals) == 3 and vals[1] == vals[0] + 1 and vals[2] == vals[1] + 1:
                    return 'chi'
            return 'set'
        except Exception:
            return 'set'

    def _fmt_called_sets(csets) -> str:
        try:
            parts: List[str] = []
            for cs in csets or []:
                tiles = getattr(cs, 'tiles', [])
                ctype = getattr(cs, 'call_type', None)
                if not ctype or ctype == 'chi':
                    ctype = _infer_call_type(tiles)
                parts.append(f"{ctype}:[" + ', '.join(_fmt_tile(t) for t in tiles) + "]")
            return '[' + '; '.join(parts) + ']'
        except Exception:
            return '[]'

    for i, (s_obj, pair) in enumerate(zip(states, action_pairs)):
        try:
            gp = decode_game_perspective(s_obj)
        except Exception as e:
            lines.append(f"[0] <decode error: {e}>")
            continue

        hand_s = '[' + ', '.join(_fmt_tile(t) for t in sorted(gp.player_hand, key=lambda t: (t.suit.value, int(t.tile_type.value)))) + ']'
        disc_s = {
            pid: '[' + ', '.join(_fmt_tile(t) for t in gp.player_discards.get(pid, [])) + ']'
            for pid in range(4)
        }
        called_s = {pid: _fmt_called_sets(gp.called_sets.get(pid, [])) for pid in range(4)}
        last_discard = _fmt_tile(gp._reactable_tile) if gp._reactable_tile is not None else 'None'
        # Convert two-head indices to a concrete move (Action/Reaction)
        try:
            a_idx = int(pair[0])
            t_idx = int(pair[1])
            mv = build_move_from_two_head(gp, a_idx, t_idx)
            if mv is not None:
                action_desc = f"{type(mv).__name__}: {mv}"
            else:
                action_desc = {'action_idx': a_idx, 'tile_idx': t_idx}
        except Exception:
            action_desc = {'action_idx': int(pair[0]), 'tile_idx': int(pair[1])}
        if include_winds:
            lines.append(
                f"[{i}] P{(actor_id if actor_id is not None else '?')} | RW:{gp.round_wind.name} | SW["
                + ', '.join(gp.seat_winds[j].name for j in range(4)) + "] | Riichi:" + str(int(gp.riichi_declared.get(actor_id, False) if hasattr(gp, 'riichi_declared') else 0)) + " | Hand " + hand_s
            )
        else:
            lines.append(
                f"[{i}] P{(actor_id if actor_id is not None else '?')} | Riichi:" + str(int(gp.riichi_declared.get(actor_id, False) if hasattr(gp, 'riichi_declared') else 0)) + " | Hand " + hand_s
            )
        lines.append(
            f"     LastDiscard:{last_discard} by {gp._owner_of_reactable_tile} | CanCall:{int(bool(gp.can_call))} | CanRon:{int(gp.can_ron())} | CanTsumo:{int(gp.can_tsumo())}"
        )
        lines.append(
            f"     Called:{called_s}"
        )
        lines.append(
            f"     Discards:{disc_s}"
        )
        lines.append(
            f"     Action:{action_desc}"
        )

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an AC dataset (.npz) produced by create_dataset_parallel.py and print the first N games")
    parser.add_argument("dataset", type=str, help="Path to .npz produced by create_dataset_parallel.py")
    parser.add_argument("--games", type=int, default=1, help="Number of games to display")
    parser.add_argument("--start_game", type=int, default=0, help="Game index offset before printing")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    # Compact dataset format fields
    game_ids = data["game_ids"]
    step_ids = data["step_ids"]
    actor_ids = data["actor_ids"]
    returns = data["returns"]
    advantages = data["advantages"] if "advantages" in data.files else None
    action_idx = data["action_idx"]
    tile_idx = data["tile_idx"]
    joint_log_probs = data["joint_log_probs"] if "joint_log_probs" in data.files else None
    get_state_row = make_npz_state_row_getter(data)
    outcomes_arr = data["game_outcomes_obj"] if "game_outcomes_obj" in data.files else None

    # Determine unique games in order of appearance
    ordered_unique = []
    seen = set()
    for gid in game_ids.tolist():
        ig = int(gid)
        if ig not in seen:
            seen.add(ig)
            ordered_unique.append(ig)

    start = max(0, int(args.start_game))
    count = max(0, int(args.games))
    selected_games = ordered_unique[start:start + count]
    if not selected_games:
        print("No games to display")
        return 0

    for gi, gid in enumerate(selected_games, start=start + 1):
        idxs = [i for i, g in enumerate(game_ids.tolist()) if int(g) == int(gid)]
        idxs.sort(key=lambda i: int(step_ids[i]))
        print("\n" + "=" * 80)
        print(f"Game {gi} (id={gid}) | steps={len(idxs)}")
        # Print winds once for the game using the first state's perspective
        if idxs:
            first_state = get_state_row(idxs[0])
            gp0 = decode_game_perspective(first_state)
            winds_line = "RW:" + gp0.round_wind.name + " | SW:[" + ', '.join(gp0.seat_winds[j].name for j in range(4)) + "]"
            print(winds_line)
        print("-" * 80)
        for i in idxs:
            st = get_state_row(i)
            actor = int(actor_ids[i])
            lines = inspect_data([st], [(int(action_idx[i]), int(tile_idx[i]))], include_winds=False, actor_id=actor)
            # Header with reward/advantage/actor/step
            rew = float(returns[i])
            adv = (float(advantages[i]) if advantages is not None else None)
            actor = int(actor_ids[i])
            step = int(step_ids[i])
            # Probability of chosen action from stored joint log-prob (if present)
            prob_s = ""
            if joint_log_probs is not None:
                j_lp = float(joint_log_probs[i])
                joint_p = float(np.exp(j_lp))
                prob_s = f" | joint_p={joint_p:.4f}"
            # Extra context from latest schema
            try:
                remaining_tiles = int(st.get('remaining_tiles', -1))
                wall_count = st.get('wall_count', None)
                extra_ctx = f" | remaining_tiles={remaining_tiles}"
                if wall_count is not None:
                    try:
                        extra_ctx += f" | wall_sum={int(np.asarray(wall_count).sum())}"
                    except Exception:
                        pass
            except Exception:
                extra_ctx = ""
            if adv is not None:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f} | advantage={adv:+.3f}{prob_s}{extra_ctx}")
            else:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f}{prob_s}{extra_ctx}")
            for line in lines:
                print(line)
        # After all steps, print the GameOutcome if present
        if outcomes_arr is not None:
            try:
                gi_idx = int(gid)
                if gi_idx < 0 or gi_idx >= len(outcomes_arr):
                    raise IndexError(f"game_outcomes_obj length={len(outcomes_arr)} doesn't include game id {gi_idx}")
                raw = outcomes_arr[gi_idx]
                outcome_dict = raw.item() if hasattr(raw, 'item') else raw
                outcome = GameOutcome.deserialize(outcome_dict)
                print("-" * 80)
                print("GameOutcome:")
                print(outcome)
            except Exception as e:
                print(f"[warn] Could not load GameOutcome for game id {gid}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


