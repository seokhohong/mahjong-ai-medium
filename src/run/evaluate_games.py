#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np

from core.game import GameOutcome, OutcomeType
from core.learn.ac_constants import ACTION_HEAD_ORDER

# Edit this list to point to your dataset files. Absolute or relative paths are fine.
# Examples:
# DATASET_PATHS = [
#     # "training_data/ac_parallel_20250828_160927.npz",
#     # "training_data/ac_parallel_20250828_183646.npz",
#     # "training_data/ac_parallel_20250828_183646.npz",
# ]
DATASET_PATHS = [
    "training_data/ac_parallel_20250828_203723.npz",
    "training_data/ac_parallel_20250828_201525.npz",
    'training_data/ac_parallel_20250829_153655.npz'
]


def _infer_per_game_riichi_and_open_calls(npz: np.lib.npyio.NpzFile) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]]]:
    """Return (riichi_by_game, open_call_by_game) as sets of player ids per game id.

    - riichi: any step where actor took 'riichi'.
    - open_call: any Chi/Pon, or Kan taken on another player's discard (daiminkan). We avoid
      counting closed Ankan as an open call. Kakan (added to an existing Pon) is already open
      due to the prior Pon and will be captured via that Pon.
    """
    if not all(k in npz for k in ("action_idx", "actor_ids", "game_ids")):
        return {}, {}

    action_idx = np.asarray(npz["action_idx"])  # [steps]
    actor_ids = np.asarray(npz["actor_ids"])    # [steps]
    game_ids = np.asarray(npz["game_ids"])      # [steps]

    # Optional context to distinguish daiminkan vs ankan
    reactable_tile = np.asarray(npz.get("reactable_tile", []))
    owner_of_reactable_tile = np.asarray(npz.get("owner_of_reactable_tile", []))

    riichi_by_game: Dict[int, Set[str]] = {}
    open_by_game: Dict[int, Set[str]] = {}

    for i in range(len(action_idx)):
        gid = int(game_ids[i])
        # Public player identifier as string
        pid = str(actor_ids[i])
        a_idx = int(action_idx[i])
        if a_idx < 0 or a_idx >= len(ACTION_HEAD_ORDER):
            continue
        name = ACTION_HEAD_ORDER[a_idx]

        if name == "riichi":
            riichi_by_game.setdefault(gid, set()).add(pid)
            continue
        # Any chi/pon is an open call
        if name.startswith("chi_") or name.startswith("pon_"):
            open_by_game.setdefault(gid, set()).add(pid)
            continue
        # Kan: only count if clearly on another's discard (daiminkan)
        if name == "kan":
            try:
                rt = int(reactable_tile[i]) if reactable_tile.size == action_idx.size else -1
                owner_raw = owner_of_reactable_tile[i] if owner_of_reactable_tile.size == action_idx.size else -1
                owner = str(owner_raw)
            except Exception:
                rt, owner = -1, "-1"
            if rt != -1 and owner != pid:
                open_by_game.setdefault(gid, set()).add(pid)

    return riichi_by_game, open_by_game


def _actors_by_game(npz: np.lib.npyio.NpzFile) -> Dict[int, Set[str]]:
    """Collect the set of actor identifiers observed per game id.

    Uses `actor_ids` (public identifiers) grouped by `game_ids`.
    Returns empty dict if arrays are missing.
    """
    if not all(k in npz for k in ("actor_ids", "game_ids")):
        return {}
    actors = np.asarray(npz["actor_ids"])  # [steps]
    gids = np.asarray(npz["game_ids"])     # [steps]
    by_game: Dict[int, Set[str]] = {}
    for i in range(len(actors)):
        gid = int(gids[i])
        pid = str(actors[i])
        by_game.setdefault(gid, set()).add(pid)
    return by_game


def _aggregate_from_file(path: str, totals: Dict[str, Dict[str, float]]) -> Tuple[int, int]:
    """Aggregate stats from one NPZ file into totals per public player identifier.

    Returns (games_in_file, draw_games_in_file) for additional reference.
    """
    data = np.load(path, allow_pickle=True)
    try:
        outcomes_raw = list(data.get("game_outcomes_obj", []))
        if not outcomes_raw:
            return 0, 0

        # Per-game riichi/open-call sets by identifier (public pid) and all actors per game
        riichi_by_game, open_by_game = _infer_per_game_riichi_and_open_calls(data)
        actors_by_game = _actors_by_game(data)

        games_in_file = len(outcomes_raw)
        draw_games = 0

        for local_gid, raw in enumerate(outcomes_raw):
            go = GameOutcome.deserialize(raw)

            # Map seats -> public identifiers as strings from the outcome
            seat_to_pub = {seat: str(po.player_id) for seat, po in go.players.items() if po is not None}

            # Determine which pids to credit participation for this game (public IDs)
            pids_in_game: Set[str] = set(seat_to_pub.values())
            pids_in_game |= set(riichi_by_game.get(local_gid, set()))
            pids_in_game |= set(open_by_game.get(local_gid, set()))
            pids_in_game |= set(actors_by_game.get(local_gid, set()))

            # Ensure totals buckets exist and increment games for all participants
            for pub_id in sorted(pids_in_game):
                if pub_id not in totals:
                    totals[pub_id] = {
                        "games": 0.0,
                        "ron_wins": 0.0,
                        "tsumo_wins": 0.0,
                        "deal_in": 0.0,
                        "draws": 0.0,
                        "tenpai_on_draw": 0.0,
                        "riichi_games": 0.0,
                        "open_call_games": 0.0,
                        "sum_win_value": 0.0,
                        "win_count": 0.0,
                        "sum_deal_in_value": 0.0,
                        "deal_in_count": 0.0,
                    }
                totals[pub_id]["games"] += 1.0

            # Riichi / open-call flags for this hand (apply to public IDs)
            riichi_pids = riichi_by_game.get(local_gid, set())
            open_call_pids = open_by_game.get(local_gid, set())

            if go.is_draw:
                draw_games += 1

            # Increment riichi/open once per participant per game to avoid double-counts
            for pub_id in (riichi_pids & pids_in_game):
                totals[pub_id]["riichi_games"] += 1.0
            for pub_id in (open_call_pids & pids_in_game):
                totals[pub_id]["open_call_games"] += 1.0

            # Attribute per-seat outcomes to their public identifiers
            for seat in range(4):
                po = go.players.get(seat)
                pub_id = seat_to_pub.get(seat, None)
                if pub_id is None:
                    continue

                if po is None:
                    continue
                ot = po.outcome_type
                pd = int(po.points_delta)

                if ot == OutcomeType.RON:
                    totals[pub_id]["ron_wins"] += 1.0
                    totals[pub_id]["sum_win_value"] += max(0, pd)
                    totals[pub_id]["win_count"] += 1.0
                elif ot == OutcomeType.TSUMO:
                    totals[pub_id]["tsumo_wins"] += 1.0
                    totals[pub_id]["sum_win_value"] += max(0, pd)
                    totals[pub_id]["win_count"] += 1.0
                elif ot == OutcomeType.DEAL_IN:
                    totals[pub_id]["deal_in"] += 1.0
                    totals[pub_id]["sum_deal_in_value"] += -min(0, pd)
                    totals[pub_id]["deal_in_count"] += 1.0

                if go.is_draw:
                    totals[pub_id]["draws"] += 1.0
                    if ot == OutcomeType.TENPAI:
                        totals[pub_id]["tenpai_on_draw"] += 1.0

            # No need for extra per-id riichi/open increments; handled above via set intersections

        return games_in_file, draw_games
    finally:
        try:
            data.close()
        except Exception:
            pass


def _format_rate(numer: float, denom: float) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{(100.0 * numer / denom):.2f}%"


def _safe_avg(total: float, count: float) -> float:
    return (total / count) if count > 0 else 0.0


def _print_totals_table(title: str, totals: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}\n")
    headers = [
        "PlayerID",
        "Ron%",
        "Tsumo%",
        "DealIn%",
        "Tenpai@Draw%",
        "Riichi%",
        "Call%",
        "AvgWin",
        "AvgDealIn",
    ]
    print("\t".join(headers))

    for pid in sorted(totals.keys()):
        t = totals[pid]
        games = t.get("games", 0.0)
        draws = t.get("draws", 0.0)
        row = [
            str(pid),
            _format_rate(t.get("ron_wins", 0.0), games),
            _format_rate(t.get("tsumo_wins", 0.0), games),
            _format_rate(t.get("deal_in", 0.0), games),
            _format_rate(t.get("tenpai_on_draw", 0.0), draws),
            _format_rate(t.get("riichi_games", 0.0), games),
            _format_rate(t.get("open_call_games", 0.0), games),
            f"{_safe_avg(t.get('sum_win_value', 0.0), t.get('win_count', 0.0)):.1f}",
            f"{_safe_avg(t.get('sum_deal_in_value', 0.0), t.get('deal_in_count', 0.0)):.1f}",
        ]
        print("\t".join(row))

    print("\nNotes:")
    print("- Tenpai@Draw% is conditioned on draw hands only.")
    print("- Call% counts Chi/Pon and daiminkan; ankan is not counted as it keeps the hand closed.")


def main() -> int:
    overall: Dict[str, Dict[str, float]] = {}
    total_games_all = 0
    total_draws_all = 0

    if not DATASET_PATHS:
        print("DATASET_PATHS is empty. Edit src/run/evaluate_games.py and add your .npz paths to DATASET_PATHS.")
        return 0

    # Sort paths by timestamp in filename: ac_parallel_YYYYMMDD_HHMMSS.npz
    ts_re = re.compile(r"(\d{8})_(\d{6})")
    def _ts_key(p: str) -> tuple:
        m = ts_re.search(os.path.basename(p))
        if m:
            return (m.group(1), m.group(2))
        try:
            return ("", f"{os.path.getmtime(p):020.0f}")
        except Exception:
            return ("", "")
    sorted_paths = sorted(DATASET_PATHS, key=_ts_key)

    for p in sorted_paths:
        if not os.path.isfile(p):
            print(f"Warning: not a file, skipping: {p}")
            continue
        file_totals: Dict[str, Dict[str, float]] = {}
        g, d = _aggregate_from_file(p, file_totals)
        total_games_all += g
        total_draws_all += d
        if g == 0:
            print(f"\n{p}: no games found.")
        else:
            _print_totals_table(f"Per-player statistics for file: {os.path.basename(p)}", file_totals)
        # Accumulate into overall
        for pid, stats in file_totals.items():
            dst = overall.setdefault(pid, {k: 0.0 for k in stats.keys()})
            for k, v in stats.items():
                dst[k] = dst.get(k, 0.0) + float(v)

    if total_games_all == 0:
        print("No games found across provided files.")
        return 0

    _print_totals_table("Per-player statistics (across all files):", overall)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
