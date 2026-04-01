import argparse
import os
import sys

from save_system import load_simulation, new_save_dir, write_crash_dump
from simulation import Simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ant sim without pygame/rendering.")
    parser.add_argument("--gens", type=int, default=3, help="How many generations to simulate.")
    parser.add_argument("--save-dir", type=str, default="", help="Optional existing save folder to resume from.")
    parser.add_argument("--print-every", type=int, default=600, help="Progress line interval in ticks.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.save_dir:
            save_dir = os.path.abspath(args.save_dir)
            sim = load_simulation(None, save_dir)
            print(f"[HEADLESS] loaded {save_dir} at gen {sim.gen}")
        else:
            save_dir = new_save_dir()
            sim = Simulation(None, save_dir)
            print(f"[HEADLESS] new run {save_dir}")
        sim.run_headless(generations=args.gens, print_every_ticks=args.print_every, save_on_exit=True)
        print(f"[HEADLESS] done. logs in: {save_dir}")
    except Exception:
        dump = write_crash_dump(sys.exc_info(), "headless.py top-level crash")
        print(f"[HEADLESS CRASH] {dump}")
        raise


if __name__ == "__main__":
    main()
