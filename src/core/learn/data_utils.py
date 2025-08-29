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
    """Build a state dict from a row in a loaded .npz dataset (new explicit schema).

    Required keys (produced by create_dataset_parallel.py):
    - hand_idx, called_idx, disc_idx, called_discards
    - round_wind, seat_winds, legal_action_mask, riichi_declarations
    - remaining_tiles, owner_of_reactable_tile, reactable_tile, newly_drawn_tile
    - dora_indicator_tiles, deal_in_tiles, wall_count
    """
    st: Dict[str, Any] = {
        'hand_idx': np.asarray(data['hand_idx'][row_index], dtype=np.int32),
        'called_idx': np.asarray(data['called_idx'][row_index], dtype=np.int32),
        'disc_idx': np.asarray(data['disc_idx'][row_index], dtype=np.int32),
        'called_discards': np.asarray(data['called_discards'][row_index], dtype=np.int32),
        'round_wind': int(data['round_wind'][row_index]),
        'seat_winds': np.asarray(data['seat_winds'][row_index], dtype=np.int32),
        'legal_action_mask': np.asarray(data['legal_action_mask'][row_index], dtype=np.int32),
        'riichi_declarations': np.asarray(data['riichi_declarations'][row_index], dtype=np.int32),
        'remaining_tiles': int(data['remaining_tiles'][row_index]),
        'owner_of_reactable_tile': int(data['owner_of_reactable_tile'][row_index]),
        'reactable_tile': int(data['reactable_tile'][row_index]),
        'newly_drawn_tile': int(data['newly_drawn_tile'][row_index]),
        'dora_indicator_tiles': np.asarray(data['dora_indicator_tiles'][row_index], dtype=np.int32),
        'wall_count': np.asarray(data['wall_count'][row_index], dtype=np.int8),
    }
    # Optional/variable-length objects
    if 'deal_in_tiles' in getattr(data, 'files', getattr(data, 'keys', lambda: [])()):
        st['deal_in_tiles'] = data['deal_in_tiles'][row_index]
    else:
        st['deal_in_tiles'] = []
    return st


def make_npz_state_row_getter(data: Any) -> Callable[[int], Dict[str, Any]]:
    """Create a fast row getter for the new explicit NPZ schema.

    Captures references to arrays so subsequent calls only index and cast.
    """
    hand_arr = data['hand_idx']
    called_arr = data['called_idx']
    disc_arr = data['disc_idx']
    cdm_arr = data['called_discards']
    rw_arr = data['round_wind']
    sw_arr = data['seat_winds']
    lam_arr = data['legal_action_mask']
    riichi_arr = data['riichi_declarations']
    rem_arr = data['remaining_tiles']
    owner_arr = data['owner_of_reactable_tile']
    react_arr = data['reactable_tile']
    newly_arr = data['newly_drawn_tile']
    dora_arr = data['dora_indicator_tiles']
    wall_arr = data['wall_count']
    dealin_arr = data['deal_in_tiles'] if 'deal_in_tiles' in getattr(data, 'files', getattr(data, 'keys', lambda: [])()) else None

    def _get(row_index: int) -> Dict[str, Any]:
        st: Dict[str, Any] = {
            'hand_idx': np.asarray(hand_arr[row_index], dtype=np.int32),
            'called_idx': np.asarray(called_arr[row_index], dtype=np.int32),
            'disc_idx': np.asarray(disc_arr[row_index], dtype=np.int32),
            'called_discards': np.asarray(cdm_arr[row_index], dtype=np.int32),
            'round_wind': int(rw_arr[row_index]),
            'seat_winds': np.asarray(sw_arr[row_index], dtype=np.int32),
            'legal_action_mask': np.asarray(lam_arr[row_index], dtype=np.int32),
            'riichi_declarations': np.asarray(riichi_arr[row_index], dtype=np.int32),
            'remaining_tiles': int(rem_arr[row_index]),
            'owner_of_reactable_tile': int(owner_arr[row_index]),
            'reactable_tile': int(react_arr[row_index]),
            'newly_drawn_tile': int(newly_arr[row_index]),
            'dora_indicator_tiles': np.asarray(dora_arr[row_index], dtype=np.int32),
            'wall_count': np.asarray(wall_arr[row_index], dtype=np.int8),
        }
        st['deal_in_tiles'] = (dealin_arr[row_index] if dealin_arr is not None else [])
        return st

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
        action_index: Optional[tuple[int, int]] = None,
        game_perspective: Optional[Any] = None,
        action_obj: Optional[Any] = None,
        encoded_state: Optional[Dict[str, Any]] = None,
        value: Optional[float] = None,
        # Two-head logging (primary action + auxiliary tile)
        action_logp: Optional[float] = None,
        tile_logp: Optional[float] = None,
        # Per-head probability dumps (optional)
        action_probs: Optional[list] = None,
        tile_probs: Optional[list] = None,
        # Optional legality masks at the time of selection
        action_mask: Optional[list] = None,
        tile_mask: Optional[list] = None,
        reason: str = 'default',
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
                # Store tuple for two-head (action_idx, tile_idx)
                'action_index': action_index[0] if action_index is not None else None,
                'tile_index': action_index[1] if action_index is not None else None,
                'game_perspective': game_perspective,
                'action_obj': action_obj,
                'encoded_state': encoded_state,
                'value': (float(value) if value is not None else None),
                # Two-head logs
                'action_logp': (float(action_logp) if action_logp is not None else None),
                'tile_logp': (float(tile_logp) if tile_logp is not None else None),
                'action_probs': list(action_probs) if action_probs is not None else None,
                'tile_probs': list(tile_probs) if tile_probs is not None else None,
                'action_mask': list(action_mask) if action_mask is not None else None,
                'tile_mask': list(tile_mask) if tile_mask is not None else None,
            }
            fname = f"illegal_move_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
            fpath = os.path.join(target_dir, fname)
            with open(fpath, 'wb') as f:
                pickle.dump(payload, f)
            print(f"[DebugSnapshot] Saved illegal move to {fpath}")
            return fpath
        except Exception as e:
            print("[DebugSnapshot] Failed to save illegal move:", e)
            return None
