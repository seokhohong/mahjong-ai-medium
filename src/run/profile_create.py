#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import pstats
import io
import os
import sys
from typing import Any, Dict

# Ensure repo src/ is importable when run as module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run.create_dataset import build_ac_dataset


def _run_build(params: Dict[str, Any]) -> None:
	build_ac_dataset(
		games=int(params["games"]),
		seed=params.get("seed"),
		temperature=float(params["temperature"]),
		zero_network_reward=bool(params["zero_network_reward"]),
		n_step=int(params["n_step"]),
		gamma=float(params["gamma"]),
		use_heuristic=bool(params["use_heuristic"]),
		model_path=params.get("model"),
	)


def main() -> int:
	ap = argparse.ArgumentParser(description="Profile the create_dataset build loop")
	# create_dataset knobs
	ap.add_argument("--games", type=int, default=5)
	ap.add_argument("--seed", type=int, default=123)
	ap.add_argument("--temperature", type=float, default=0.1)
	ap.add_argument("--zero_network_reward", action="store_true")
	ap.add_argument("--n_step", type=int, default=3)
	ap.add_argument("--gamma", type=float, default=0.99)
	ap.add_argument("--use_heuristic", action="store_true")
	ap.add_argument("--model", type=str, default=None)
	# profiling knobs
	ap.add_argument("--outfile", type=str, default=None, help="Optional .prof output path")
	ap.add_argument("--sort", type=str, default="tottime", help="Sort key for pstats (tottime,cumtime,...)" )
	ap.add_argument("--limit", type=int, default=50, help="Rows to print from stats")
	args = ap.parse_args()

	params: Dict[str, Any] = vars(args)

	prof = cProfile.Profile()
	prof.enable()
	try:
		_run_build(params)
	finally:
		prof.disable()

	# Save to file if requested
	if args.outfile:
		prof.dump_stats(args.outfile)
		print(f"Profile saved to {args.outfile}")

	# Print summary to stdout
	s = io.StringIO()
	ps = pstats.Stats(prof, stream=s).strip_dirs().sort_stats(args.sort)
	ps.print_stats(args.limit)
	print(s.getvalue())
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
