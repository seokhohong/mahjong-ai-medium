from __future__ import annotations

from typing import List
import argparse
import numpy as np

from core.learn.feature_engineering import decode_game_perspective


def inspect_data(states: np.ndarray, actions: np.ndarray, *, start: int = 0, count: int = 5, include_winds: bool = True) -> List[str]:
    """Pretty-print a slice of dataset entries by rehydrating GamePerspective via feature engineering.

    - states: numpy object array of dicts from encode_game_perspective
    - actions: numpy object array of serialized action dicts (policy_utils.serialize_action)
    - start/count: slice controls

    Returns a list of human-readable lines.
    """
    lines: List[str] = []

    def _fmt_tile(t) -> str:
        try:
            return str(t)
        except Exception:
            return f"{t}"

    def _fmt_called_sets(csets) -> str:
        try:
            parts: List[str] = []
            for cs in csets or []:
                tiles = getattr(cs, 'tiles', [])
                call_type = getattr(cs, 'call_type', '?')
                parts.append(f"{call_type}:[" + ', '.join(_fmt_tile(t) for t in tiles) + "]")
            return '[' + '; '.join(parts) + ']'
        except Exception:
            return '[]'

    n = int(min(len(states), start + count))
    for i in range(int(start), n):
        s_obj = states[i]
        a_obj = actions[i]
        # Handle numpy object arrays with .item()
        if hasattr(s_obj, 'item'):
            s_obj = s_obj.item()
        if hasattr(a_obj, 'item'):
            a_obj = a_obj.item()
        try:
            gp = decode_game_perspective(s_obj)
        except Exception as e:
            lines.append(f"[{i}] <decode error: {e}>")
            continue

        hand_s = '[' + ', '.join(_fmt_tile(t) for t in sorted(gp.player_hand, key=lambda t: (t.suit.value, int(t.tile_type.value)))) + ']'
        disc_s = {
            pid: '[' + ', '.join(_fmt_tile(t) for t in gp.player_discards.get(pid, [])) + ']'
            for pid in range(4)
        }
        called_s = {pid: _fmt_called_sets(gp.called_sets.get(pid, [])) for pid in range(4)}
        last_discard = _fmt_tile(gp.last_discarded_tile) if gp.last_discarded_tile is not None else 'None'
        action_desc = a_obj if isinstance(a_obj, dict) else {'type': str(a_obj)}
        if include_winds:
            lines.append(
                f"[{i}] P{gp.player_id} | RW:{gp.round_wind.name} | SW:["
                + ', '.join(gp.seat_winds[j].name for j in range(4)) + "] | Hand " + hand_s
            )
        else:
            lines.append(
                f"[{i}] P{gp.player_id} | Hand " + hand_s
            )
        lines.append(
            f"     LastDiscard:{last_discard} by {gp.last_discard_player} | CanCall:{int(bool(gp.can_call))} | CanRon:{int(gp.can_ron())} | CanTsumo:{int(gp.can_tsumo())}"
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
    parser = argparse.ArgumentParser(description="Inspect an AC dataset (.npz) and print the first N games")
    parser.add_argument("dataset", type=str, help="Path to .npz produced by create_dataset.py")
    parser.add_argument("--games", type=int, default=1, help="Number of games to display")
    parser.add_argument("--start_game", type=int, default=0, help="Game index offset before printing")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    states = data["states"]
    actions = data["actions"]
    game_ids = data["game_ids"]
    step_ids = data["step_ids"]
    actor_ids = data["actor_ids"]
    returns = data["returns"]
    advantages = data.get("advantages") if "advantages" in data.files else None

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
            first_state = states[idxs[0]].item() if hasattr(states[idxs[0]], 'item') else states[idxs[0]]
            try:
                gp0 = decode_game_perspective(first_state)
                winds_line = "RW:" + gp0.round_wind.name + " | SW:[" + ', '.join(gp0.seat_winds[j].name for j in range(4)) + "]"
                print(winds_line)
            except Exception:
                pass
        print("-" * 80)
        for i in idxs:
            lines = inspect_data(states[i:i+1], actions[i:i+1], start=0, count=1, include_winds=False)
            # Header with reward/advantage/actor/step
            rew = float(returns[i])
            adv = (float(advantages[i]) if advantages is not None else None)
            actor = int(actor_ids[i])
            step = int(step_ids[i])
            if adv is not None:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f} | advantage={adv:+.3f}")
            else:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f}")
            for line in lines:
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


