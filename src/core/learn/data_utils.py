"""
Data loading utilities for AC learning.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Dict, Callable, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_gsv_scaler(data_path: str | None) -> StandardScaler | None:
    """Load the StandardScaler from a dataset if available.

    Args:
        data_path: Path to the .npz dataset file, or None

    Returns:
        The fitted StandardScaler for game state vectors, or None if not available

    Example:
        # Load dataset and scaler
        data = np.load('training_data/dataset.npz')
        scaler = load_gsv_scaler('training_data/dataset.npz')

        # Use at inference time
        scaled_gsv = scaler.transform([raw_gsv])
    """
    if data_path is None:
        return None

    import os
    if not os.path.exists(data_path):
        print(f"Warning: Scaler dataset not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)
        if 'gsv_scaler' in data:
            scaler_bytes = data['gsv_scaler']
            scaler = pickle.loads(scaler_bytes)
            print(f"Loaded GSV scaler from: {data_path}")
            return scaler
        else:
            print(f"Warning: No GSV scaler found in {data_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load GSV scaler from {data_path}: {e}")
        return None


def build_state_from_arrays(
    hand_idx: Any,
    disc_idx: Any,
    called_idx: Any,
    game_state: Any,
    called_discards: Any | None = None,
) -> Dict[str, Any]:
    """Construct a standardized state dict from packed array fields.

    Ensures consistent dtypes expected by downstream decoders and models.

    Args:
        hand_idx: Hand indices array-like
        disc_idx: Discards indices array-like
        called_idx: Called tiles indices array-like
        game_state: Flat game-state vector
        called_discards: Optional called-discard mask/indices per player

    Returns:
        Dict with keys: 'hand_idx', 'disc_idx', 'called_idx', 'game_state', 'called_discards'
    """
    st: Dict[str, Any] = {
        'hand_idx': np.asarray(hand_idx, dtype=np.int32),
        'disc_idx': np.asarray(disc_idx, dtype=np.int32),
        'called_idx': np.asarray(called_idx, dtype=np.int32),
        'game_state': np.asarray(game_state, dtype=np.float32),
    }
    if called_discards is not None:
        st['called_discards'] = np.asarray(called_discards, dtype=np.int32)
    return st


def build_state_from_npz_row(data: Any, row_index: int) -> Dict[str, Any]:
    """Build a state dict from a row in a loaded .npz dataset.

    Expects standard keys in the .npz: 'hand_idx', 'disc_idx', 'called_idx',
    'game_state', and optionally 'called_discards'.
    """
    hand = data['hand_idx'][row_index]
    disc = data['disc_idx'][row_index]
    called = data['called_idx'][row_index]
    gsv = data['game_state'][row_index]
    cdm = data['called_discards'][row_index] if 'called_discards' in data else None
    return build_state_from_arrays(hand, disc, called, gsv, cdm)


def make_npz_state_row_getter(data: Any) -> Callable[[int], Dict[str, Any]]:
    """Create a fast row getter that avoids repeated key lookups and reloading.

    Captures references to arrays within the provided npz dataset object so that
    subsequent per-row calls only perform array indexing and dtype normalization.
    """
    hand_arr = data['hand_idx']
    disc_arr = data['disc_idx']
    called_arr = data['called_idx']
    gsv_arr = data['game_state']
    cdm_arr = data['called_discards'] if 'called_discards' in getattr(data, 'files', getattr(data, 'keys', lambda: [])()) else None

    def _get(row_index: int) -> Dict[str, Any]:
        return build_state_from_arrays(
            hand_arr[row_index],
            disc_arr[row_index],
            called_arr[row_index],
            gsv_arr[row_index],
            (cdm_arr[row_index] if cdm_arr is not None else None),
        )

    return _get


class DebugSnapshot:
    """Utilities to persist debug snapshots such as illegal moves.

    By default, saves to the repository's root `illegal_moves/` directory.
    """

    @staticmethod
    def _default_out_dir() -> str:
        # __file__ is src/core/learn/data_utils.py -> go four levels up to repo root
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
        )
        return os.path.join(repo_root, 'illegal_moves')

    @staticmethod
    def save_illegal_move(
        *,
        action_index: Optional[int] = None,
        game_perspective: Optional[Any] = None,
        action_obj: Optional[Any] = None,
        encoded_state: Optional[Dict[str, Any]] = None,
        value: Optional[float] = None,
        main_logp: Optional[float] = None,
        main_probs: Optional[list] = None,
        reason: str = 'illegal_action_index',
        out_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Serialize an illegal move snapshot to a .pkl file.

        Returns the file path on success; None on failure.
        """
        try:
            target_dir = out_dir or DebugSnapshot._default_out_dir()
            os.makedirs(target_dir, exist_ok=True)

            payload = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'reason': reason,
                'action_index': (int(action_index) if action_index is not None else None),
                'game_perspective': game_perspective,
                'action_obj': action_obj,
                'encoded_state': encoded_state,
                'value': (float(value) if value is not None else None),
                'main_logp': (float(main_logp) if main_logp is not None else None),
                'main_probs': list(main_probs) if main_probs is not None else None,
            }
            idx_part = str(action_index) if action_index is not None else 'NA'
            fname = f"illegal_move_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
            fpath = os.path.join(target_dir, fname)
            with open(fpath, 'wb') as f:
                pickle.dump(payload, f)
            print(f"[DebugSnapshot] Saved illegal move to {fpath}")
            return fpath
        except Exception as e:
            print("[DebugSnapshot] Failed to save illegal move:", e)
            return None
