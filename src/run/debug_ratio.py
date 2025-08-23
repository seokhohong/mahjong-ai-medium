import torch
import argparse
import numpy as np

from run.train_model import _prepare_batch_tensors, ACDataset
from torch.utils.data import DataLoader
from core.learn.ac_player import ACPlayer

def diagnose_ratio_mismatch(model: torch.nn.Module, dataset: ACDataset, device: torch.device, *, batch_size: int = 128, max_batches: int | None = 1):
    """
    Test to understand why initial ratios are not close to 1.0
    """
    model.eval()
    dl = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    print("=== DIAGNOSIS: Why are ratios not starting at 1.0? ===")

    total_n = 0
    ratio_sum = 0.0
    ratio_sumsq = 0.0
    clipped_cnt = 0
    batches_run = 0

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            (
                hand, calls, disc, gsv,
                action_idx, tile_idx,
                joint_old_log_probs, advantages, returns,
            ) = _prepare_batch_tensors(batch, device)

            a_pp, t_pp, _ = model(hand.float(), calls.float(), disc.float(), gsv.float())
            a_idx = action_idx.view(-1, 1)
            t_idx = tile_idx.view(-1, 1)
            chosen_a = torch.gather(a_pp.clamp_min(1e-8), 1, a_idx).squeeze(1)
            chosen_t = torch.gather(t_pp.clamp_min(1e-8), 1, t_idx).squeeze(1)
            current_logp_a = chosen_a.log()
            current_logp_t = chosen_t.log()
            current_joint_logp = current_logp_a + current_logp_t
            log_prob_diff = current_joint_logp - joint_old_log_probs
            ratios = torch.exp(log_prob_diff)

            # Per-batch diagnostics
            print(f"[Batch {bi+1}] old_joint_logp mean={joint_old_log_probs.mean():.4f} std={joint_old_log_probs.std():.4f} | current_joint_logp mean={current_joint_logp.mean():.4f} std={current_joint_logp.std():.4f}")
            print(f"[Batch {bi+1}] ratio mean={ratios.mean():.4f} std={ratios.std():.4f} min={ratios.min():.4f} max={ratios.max():.4f}")
            # Approx head-wise contributions (for intuition only)
            approx_old_logp_a = joint_old_log_probs * 0.5
            approx_old_logp_t = joint_old_log_probs * 0.5
            ratio_a_approx = torch.exp(current_logp_a - approx_old_logp_a)
            ratio_t_approx = torch.exp(current_logp_t - approx_old_logp_t)
            print(f"[Batch {bi+1}] approx ratio(action) mean={ratio_a_approx.mean():.4f} | approx ratio(tile) mean={ratio_t_approx.mean():.4f}")
            print(f"[Batch {bi+1}] action_pp mean={a_pp.mean():.6f} max={a_pp.max():.6f} min={a_pp.min():.6f} | tile_pp mean={t_pp.mean():.6f} max={t_pp.max():.6f} min={t_pp.min():.6f}")

            extremes = (ratios < 0.1) | (ratios > 10.0)
            if extremes.any():
                frac = 100.0 * float(extremes.float().mean().item())
                print(f"[Batch {bi+1}] extreme ratios (<0.1 or >10): {int(extremes.sum().item())}/{len(ratios)} ({frac:.1f}%)")
                idxs = extremes.nonzero(as_tuple=True)[0][:5]
                for idx in idxs:
                    print(f"    idx {int(idx)}: ratio={float(ratios[idx].item()):.4f} old_logp={float(joint_old_log_probs[idx].item()):.4f} new_logp={float(current_joint_logp[idx].item()):.4f}")

            # Accumulate
            bsz = int(gsv.size(0))
            total_n += bsz
            ratio_sum += float(ratios.sum().item())
            ratio_sumsq += float((ratios * ratios).sum().item())
            clipped_cnt += int(((ratios < 0.8) | (ratios > 1.2)).sum().item())
            batches_run += 1

            if max_batches is not None and batches_run >= int(max_batches):
                break

    # Summary
    if total_n > 0:
        r_mean = ratio_sum / total_n
        r_var = max(0.0, (ratio_sumsq / total_n) - r_mean * r_mean)
        r_std = float(np.sqrt(r_var))
        clip_rate = clipped_cnt / max(1, total_n)
        print(f"=== Summary over {total_n} examples ({batches_run} batch(es)) ===")
        print(f"ratio mean={r_mean:.4f} std={r_std:.4f} | clipped(|r-1|>0.2) rate={clip_rate:.3f}")


def test_data_collection_consistency(player, game_states_sample):
    """
    Test if the same game state produces consistent outputs between
    data collection and training
    """
    print("=== TESTING DATA COLLECTION CONSISTENCY ===")

    # Simulate data collection
    collected_data = []
    for gs in game_states_sample:
        move, value, action_idx, tile_idx, logp_joint = player.compute_play(gs)
        collected_data.append((gs, action_idx, tile_idx, logp_joint))

    # Simulate training evaluation
    player.network.torch_module.eval()
    for gs, expected_action_idx, expected_tile_idx, expected_logp_joint in collected_data:
        # Evaluate the same way as training
        a_probs, t_probs, value = player.network.evaluate(gs)

        # Get log prob for the same action
        action_logp = np.log(max(1e-8, a_probs[expected_action_idx]))
        tile_logp = np.log(max(1e-8, t_probs[expected_tile_idx]))
        training_logp_joint = action_logp + tile_logp

        diff = abs(training_logp_joint - expected_logp_joint)
        print(f"Action {expected_action_idx}, Tile {expected_tile_idx}:")
        print(f"  Data collection logp: {expected_logp_joint:.6f}")
        print(f"  Training eval logp:   {training_logp_joint:.6f}")
        print(f"  Difference:          {diff:.6f}")

        if diff > 0.01:  # Significant difference
            print(f"  ⚠️  INCONSISTENCY DETECTED!")


def check_model_loading_consistency(model_path):
    """
    Test if model loading preserves the exact same outputs
    """
    print("=== TESTING MODEL LOADING CONSISTENCY ===")

    # Load model twice
    player1 = ACPlayer.from_directory(model_path)
    player2 = ACPlayer.from_directory(model_path)

    # Test on dummy input
    dummy_gsv = torch.randn(1, player1.network._gsv_scaler.n_features_in_)
    dummy_hand = torch.randn(1, 34)
    dummy_calls = torch.randn(1, 4, 4)
    dummy_disc = torch.randn(1, 4, 34)

    with torch.no_grad():
        a1, t1, v1 = player1.network.torch_module(dummy_hand, dummy_calls, dummy_disc, dummy_gsv)
        a2, t2, v2 = player2.network.torch_module(dummy_hand, dummy_calls, dummy_disc, dummy_gsv)

        action_diff = (a1 - a2).abs().max().item()
        tile_diff = (t1 - t2).abs().max().item()
        value_diff = (v1 - v2).abs().max().item()

        print(f"Action head max diff: {action_diff:.10f}")
        print(f"Tile head max diff: {tile_diff:.10f}")
        print(f"Value head max diff: {value_diff:.10f}")

        if max(action_diff, tile_diff, value_diff) > 1e-6:
            print("⚠️  MODEL LOADING INCONSISTENCY!")
        else:
            print("✅ Model loading is consistent")


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for PPO ratio and model consistency")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Ratio diagnostic
    ap_ratio = sub.add_parser("ratio", help="Diagnose initial PPO ratios against stored joint log-probs")
    ap_ratio.add_argument("--data", type=str, required=True, help="Path to dataset .npz")
    ap_ratio.add_argument("--model", type=str, required=True, help="Path to model dir or .pt file (with scaler.pkl alongside)")
    ap_ratio.add_argument("--batch_size", type=int, default=128)
    ap_ratio.add_argument("--max_batches", type=int, default=1, help="How many batches to scan (None for all)")
    ap_ratio.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if None)")

    # Model load consistency
    ap_load = sub.add_parser("check-load", help="Check model loading produces identical outputs")
    ap_load.add_argument("--model", type=str, required=True, help="Path to model dir or .pt file (with scaler.pkl alongside)")

    args = parser.parse_args()

    if args.cmd == "ratio":
        # Load player/model and dataset (dataset needs network for feature extraction + scaler fitting)
        player = ACPlayer.from_directory(args.model)
        net = player.network
        ds = ACDataset(args.data, net, fit_scaler=False)
        dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net.torch_module.to(dev)
        diagnose_ratio_mismatch(net.torch_module, ds, dev, batch_size=int(args.batch_size), max_batches=(None if args.max_batches is None else int(args.max_batches)))
    elif args.cmd == "check-load":
        check_model_loading_consistency(args.model)


if __name__ == "__main__":
    main()