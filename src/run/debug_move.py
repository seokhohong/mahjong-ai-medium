import argparse
import pickle
import sys
from typing import Any
from pathlib import Path


def _ensure_src_importable() -> None:
    # If running as a script (python src/run/debug_move.py), ensure project root is in sys.path
    # so that 'src.core.game' can be imported for unpickling
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # .../mahjong-ai-medium
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved illegal move dump (.pkl)")
    parser.add_argument("file", help="Path to the .pkl dump created on illegal move")
    args = parser.parse_args()

    _ensure_src_importable()

    try:
        with open(args.file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle from {args.file}: {e}")
        sys.exit(1)

    actor_id = data.get("actor_id")
    perspective = data.get("perspective")
    move = data.get("move")
    game = data.get("game")
    perspective.legal_flat_mask()
    perspective.is_legal(move)
    print("===== MediumJong Illegal Move Debug =====")
    print(f"File: {args.file}")
    print(f"Actor ID: {actor_id}")
    print("Hint: run as 'python -m src.run.debug_move <file.pkl>' from project root for best results.")

    # Pretty print move
    print("\n-- Move --")
    try:
        from src.core.game import Discard, Riichi, Pon, Chi, KanDaimin, KanKakan, KanAnkan, Tsumo, Ron, Tile
    except Exception:
        # Import only used for isinstance checks; if it fails, fallback to generic print
        Discard = Riichi = Pon = Chi = KanDaimin = KanKakan = KanAnkan = Tsumo = Ron = object  # type: ignore
        Tile = object  # type: ignore

    def _fmt_tile(t: Any) -> str:
        # Try to use __str__ if available (Tile defines __str__)
        try:
            s = str(t)
        except Exception:
            s = repr(t)
        return s

    def _fmt_tiles(ts: Any) -> str:
        try:
            return '[' + ', '.join(_fmt_tile(t) for t in ts) + ']'
        except Exception:
            return repr(ts)

    if isinstance(move, Discard):
        print(f"Discard: {_fmt_tile(move.tile)}")
    elif isinstance(move, Riichi):
        print(f"Riichi (discard): {_fmt_tile(move.tile)}")
    elif isinstance(move, Pon):
        print(f"Pon: {_fmt_tiles(move.tiles)}")
    elif isinstance(move, Chi):
        print(f"Chi: {_fmt_tiles(move.tiles)}")
    elif isinstance(move, KanDaimin):
        print(f"Kan (daiminkan): {_fmt_tiles(move.tiles)}")
    elif isinstance(move, KanKakan):
        print(f"Kan (kakan): {_fmt_tile(move.tile)}")
    elif isinstance(move, KanAnkan):
        print(f"Kan (ankan): {_fmt_tile(move.tile)}")
    elif isinstance(move, Tsumo):
        print("Tsumo")
    elif isinstance(move, Ron):
        print("Ron")
    else:
        print(repr(move))

    # Pretty print perspective
    print("\n-- Game Perspective --")
    try:
        # __repr__ of GamePerspective is informative
        print(perspective)
    except Exception as e:
        print(f"<failed to print perspective: {e}>")
        print(repr(perspective))

    # Some helpful derived info if available
    print("\n-- Derived Info --")
    try:
        can_tsumo = perspective.can_tsumo() if hasattr(perspective, 'can_tsumo') else None
        can_ron = perspective.can_ron() if hasattr(perspective, 'can_ron') else None
        waits = perspective._waits() if hasattr(perspective, '_waits') else None
        if can_tsumo is not None:
            print(f"can_tsumo: {can_tsumo}")
        if can_ron is not None:
            print(f"can_ron: {can_ron}")
        if waits is not None:
            try:
                print("waits:", _fmt_tiles(waits))
            except Exception:
                print("waits:", waits)
    except Exception as e:
        print(f"<failed to compute derived info: {e}>")

    # Summarize game object if present
    if game is not None:
        print("\n-- Game Summary --")
        try:
            # Basic flags
            cur = getattr(game, 'current_player_idx', None)
            over = getattr(game, 'game_over', None)
            last_tile = getattr(game, 'last_discarded_tile', None)
            last_from = getattr(game, 'last_discard_player', None)
            winners = getattr(game, 'winners', None)
            loser = getattr(game, 'loser', None)
            rs = getattr(game, 'riichi_sticks_pot', None)
            points = getattr(game, 'points', None)
            cum = getattr(game, 'cumulative_points', None)
            print(f"current_player_idx: {cur}")
            print(f"game_over: {over}")
            print(f"last_discarded_tile: {last_tile}")
            print(f"last_discard_player: {last_from}")
            print(f"winners: {winners}")
            print(f"loser: {loser}")
            print(f"riichi_sticks_pot: {rs}")
            print(f"points (this hand): {points}")
            print(f"cumulative_points: {cum}")
        except Exception as e:
            print(f"<failed to summarize game: {e}>")


if __name__ == "__main__":
    main()
