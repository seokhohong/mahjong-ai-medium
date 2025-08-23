#!/usr/bin/env python3
"""
Validate a trained AC model on a dataset by checking policy accuracy.

Usage:
    python validate_model.py --model path/to/model.pt --data path/to/dataset.npz --acc 0.8
"""

import argparse
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.learn.ac_network import ACNetwork
from core.learn.ac_player import ACPlayer
from core.learn.data_utils import load_gsv_scaler, build_state_from_arrays
from core.learn.feature_engineering import decode_game_perspective
from core.learn.policy_utils import encode_two_head_action


def load_dataset(data_path: str) -> Dict[str, Any]:
    """Load and preprocess dataset similar to ACDataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    # Extract data in the same format as ACDataset
    hand_arr = data['hand_idx']
    disc_arr = data['disc_idx']
    called_arr = data['called_idx']
    gsv_arr = data['game_state']
    called_discards_arr = data['called_discards']
    action_idx = data['action_idx']
    tile_idx = data['tile_idx']

    N = len(hand_arr)
    print(f"Loading dataset with {N} samples...")

    # Convert to the format expected by the network
    states = []
    actions = []  # list of (action_idx, tile_idx)

    for i in range(N):
        state_dict = build_state_from_arrays(
            hand_arr[i],
            disc_arr[i],
            called_arr[i],
            gsv_arr[i],
            called_discards_arr[i],
        )
        states.append(state_dict)
        actions.append((int(action_idx[i]), int(tile_idx[i])))

    return {'states': states, 'actions': actions}


def validate_model(model_path: str, data_path: str, accuracy_threshold: float, max_samples: int | None = None) -> bool:
    """Validate model accuracy on the dataset."""

    # Initialize network on CPU for faster per-sample evaluation
    device = torch.device('cpu')
    print(f"Using device: {device}")

    ac_player = ACPlayer.from_directory(model_path, temperature=1.0)

    # Load dataset
    dataset = load_dataset(data_path)
    states = dataset['states']
    actions = dataset['actions']

    # Limit samples if max_samples is specified
    if max_samples is not None and max_samples < len(states):
        print(f"Limiting evaluation to {max_samples} out of {len(states)} samples...")
        indices = np.random.choice(len(states), max_samples, replace=False)
        states = [states[i] for i in indices]
        actions = [actions[i] for i in indices]

    print(f"Evaluating {len(states)} samples...")

    correct_predictions = 0
    total_samples = len(states)

    # Evaluate each sample
    for i, (state_dict, (true_a_idx, true_t_idx)) in enumerate(tqdm(zip(states, actions), desc="Evaluating")):
        # Decode state into GamePerspective
        game_perspective = decode_game_perspective(state_dict)

        # Get model prediction
        predicted_move, _, _ = ac_player.compute_play(game_perspective)
        pa_idx, pt_idx = encode_two_head_action(predicted_move)
        # Check if prediction matches ground truth on both heads
        if pa_idx == true_a_idx and pt_idx == true_t_idx:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"\nValidation Results:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Policy accuracy: {accuracy:.4f}")
    print(f"Required threshold: {accuracy_threshold:.4f}")

    # Check threshold
    if accuracy >= accuracy_threshold:
        print("✅ Validation PASSED: Model meets accuracy threshold")
        return True
    else:
        print("❌ Validation FAILED: Model below accuracy threshold")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate AC model policy accuracy on dataset')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset file (.npz)')
    parser.add_argument('--acc', type=float, default=0.8,
                       help='Minimum required policy accuracy (default: 0.8)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')

    args = parser.parse_args()

    try:
        success = validate_model(args.model, args.data, args.acc, args.max_samples)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
