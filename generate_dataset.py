#!/usr/bin/env python3
"""
generate_dataset.py
-------------------
Script sinh SAMSP dataset. Chạy từ bất kỳ đâu:

    python3 /Users/tuyenkv/Documents/SAMSP/generate_dataset.py [options]

Ví dụ:
    python3 generate_dataset.py                          # dùng defaults
    python3 generate_dataset.py --preset easy --n 20    # 20 easy instances
    python3 generate_dataset.py --paper --types 1 2 --copies 5 10  # paper Set I
    python3 generate_dataset.py --out /tmp/samsp_data   # đổi output dir
    python3 generate_dataset.py --all                   # sinh toàn bộ benchmark sets
"""

import argparse
import json
import os
import sys

# ── Setup path ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RCPSP_SRC  = SCRIPT_DIR
sys.path.insert(0, RCPSP_SRC)

from src.samsp.generator import (
    SAMSPGenerator, GeneratorConfig,
    PAPER_TEST_TYPES, DIFFICULTY_PRESETS,
)
from src.samsp.problem import SAMSPProblem


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_instance(inst: SAMSPProblem, path: str, verbose: bool = True):
    """Validate, run closure, then save to JSON."""
    errors = inst.validate()
    ok = "OK" if not errors else f"ERRORS: {errors[:1]}"
    lb  = inst.lower_bound or 0
    sha = inst.sha256()
    if verbose:
        print(f"  n={inst.number_of_activities:4d}  LB={lb:5d}  sha={sha}  {ok}")
    inst.to_json(path)
    return not bool(errors)


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ── Benchmark set generators ──────────────────────────────────────────────────

def gen_paper_set_i(out_dir: str, seeds: range):
    """Reproduce paper Set I (24 instances × len(seeds) seeds)."""
    make_dir(out_dir)
    configs = [
        {1: 5}, {1: 10}, {1: 20},
        {2: 5}, {2: 10}, {2: 20},
        {3: 5}, {3: 10}, {3: 20},
        {4: 5}, {4: 10}, {4: 20},
        {1: 5,  2: 5},  {1: 10, 2: 10},
        {1: 5,  3: 5},  {1: 10, 3: 10},
        {1: 5,  4: 5},  {1: 10, 4: 10},
        {2: 5,  3: 5},  {2: 10, 3: 10},
        {2: 5,  4: 5},  {2: 10, 4: 10},
        {3: 5,  4: 5},  {3: 10, 4: 10},
    ]
    total = ok = 0
    for seed in seeds:
        gen = SAMSPGenerator(seed=seed)
        for tc in configs:
            name = "_".join(f"t{t}x{c}" for t, c in sorted(tc.items()))
            path = os.path.join(out_dir, f"{name}_s{seed}.json")
            inst = gen.from_paper_types(tc)
            ok += save_instance(inst, path)
            total += 1
    print(f"  {out_dir}: {ok}/{total} valid")


def gen_synthetic(out_dir: str, preset: str, n: int, seeds: range):
    """Synthetic random instances for a given difficulty preset."""
    make_dir(out_dir)
    ok = total = 0
    for seed in seeds:
        gen = SAMSPGenerator(seed=seed)
        for i in range(n):
            cfg = GeneratorConfig(preset=preset)
            inst = gen.generate(cfg)
            path = os.path.join(out_dir, f"{preset}_s{seed}_i{i:03d}.json")
            ok += save_instance(inst, path)
            total += 1
            gen._rng.seed(seed * 10000 + i + 1)   # advance rng between instances
    print(f"  {out_dir}: {ok}/{total} valid")


def gen_unsat(out_dir: str, n: int, seeds: range):
    """UNSAT instances: positive cycles."""
    make_dir(out_dir)
    from src.samsp.problem import SAMSPProblem, StorageZone
    import random
    ok = total = 0
    for seed in seeds:
        rng = random.Random(seed)
        for i in range(n):
            # Build tiny instance with deliberate positive cycle
            d = [rng.randint(2, 5) for _ in range(6)]
            lags = [(0,1,d[0]),(1,2,d[1]),(2,3,d[2]),(3,4,d[3]),(4,5,d[4])]
            # Add cycle: (2,1, val) where val > 0 for positive cycle
            cycle_val = rng.randint(5, 15)
            lags.append((2, 1, cycle_val))
            inst = SAMSPProblem(
                number_of_activities=6,
                number_of_resources=1,
                durations=d,
                requests=[[1]]*6,
                capacities=[1],
                lags=lags,
                storages=[],
                storage_deltas=[[]]*6,
                test_metadata={"type": "unsat_cycle", "seed": seed, "i": i},
            )
            errors = inst.validate()
            is_unsat = any("cycle" in e.lower() for e in errors)
            path = os.path.join(out_dir, f"unsat_s{seed}_i{i:03d}.json")
            inst.to_json(path)
            print(f"  n=6  UNSAT={'YES' if is_unsat else 'NO'}  seed={seed}")
            ok += is_unsat
            total += 1
    print(f"  {out_dir}: {ok}/{total} are UNSAT")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SAMSP Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out",    default=os.path.join(SCRIPT_DIR, "data"),
                   help="Output root directory (default: ./data)")
    p.add_argument("--seeds",  type=int, nargs="+", default=[0],
                   help="Random seeds (default: 0)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all",    action="store_true",
                      help="Generate all benchmark sets (Set I + Easy/Med/Hard/UNSAT)")
    mode.add_argument("--paper",  action="store_true",
                      help="Paper Set I instances only")
    mode.add_argument("--preset", choices=["easy","medium","hard"],
                      help="Generate synthetic instances with given preset")
    mode.add_argument("--unsat",  action="store_true",
                      help="Generate UNSAT instances only")

    p.add_argument("--types",  type=int, nargs="+", default=[1],
                   help="Test type IDs for --paper mode (default: 1)")
    p.add_argument("--copies", type=int, nargs="+", default=[5],
                   help="Number of copies per type for --paper mode (default: 5)")
    p.add_argument("-n", "--n-instances", type=int, default=10,
                   help="Number of instances per seed for --preset (default: 10)")
    p.add_argument("-q", "--quiet", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    out  = args.out
    seeds = args.seeds
    verbose = not args.quiet

    print(f"Output: {out}")
    print(f"Seeds : {seeds}")

    if args.all:
        print("\n── Paper Set I ──")
        gen_paper_set_i(os.path.join(out, "set_i"), seeds=range(1))
        for preset in ("easy", "medium", "hard"):
            n = {"easy": 100, "medium": 100, "hard": 50}[preset]
            print(f"\n── Synthetic {preset.capitalize()} ({n} instances) ──")
            gen_synthetic(os.path.join(out, f"synthetic_{preset}"),
                          preset=preset, n=n // max(len(seeds),1),
                          seeds=seeds)
        print("\n── UNSAT ──")
        gen_unsat(os.path.join(out, "unsat"), n=5, seeds=seeds)

    elif args.paper:
        if len(args.types) != len(args.copies):
            if len(args.copies) == 1:
                args.copies = args.copies * len(args.types)
            else:
                print("ERROR: --types and --copies must have same length")
                sys.exit(1)
        tc = dict(zip(args.types, args.copies))
        name = "_".join(f"t{t}x{c}" for t, c in sorted(tc.items()))
        print(f"\n── Paper types {tc} ──")
        make_dir(out)
        for seed in seeds:
            gen = SAMSPGenerator(seed=seed)
            inst = gen.from_paper_types(tc)
            path = os.path.join(out, f"{name}_s{seed}.json")
            save_instance(inst, path, verbose=verbose)

    elif args.preset:
        print(f"\n── Synthetic {args.preset} ──")
        gen_synthetic(
            os.path.join(out, f"synthetic_{args.preset}"),
            preset=args.preset,
            n=args.n_instances,
            seeds=seeds,
        )

    elif args.unsat:
        print("\n── UNSAT ──")
        gen_unsat(os.path.join(out, "unsat"), n=args.n_instances, seeds=seeds)

    elif args.set_i:
        gen_set_i(os.path.join(out, 'set_i'), verbose=verbose)

    elif args.set_ii:
        gen_set_ii(os.path.join(out, 'set_ii'), verbose=verbose)

    else:
        # Default: generate a small mixed demo set
        print("\n── Demo: paper Set I (5 configs × seed 0) ──")
        for tc in [{1:5}, {2:5}, {3:5}, {4:5}, {1:5,2:5}]:
            gen = SAMSPGenerator(seed=0)
            inst = gen.from_paper_types(tc)
            name = "_".join(f"t{t}x{c}" for t,c in sorted(tc.items()))
            path = os.path.join(make_dir(os.path.join(out,"demo")), f"{name}.json")
            save_instance(inst, path, verbose=True)
        print("\n── Demo: synthetic medium (5 instances, seed 42) ──")
        gen = SAMSPGenerator(seed=42)
        for i in range(5):
            cfg = GeneratorConfig(preset="medium")
            inst = gen.generate(cfg)
            path = os.path.join(make_dir(os.path.join(out,"demo")), f"medium_i{i:03d}.json")
            save_instance(inst, path, verbose=True)

    print("\nDone.")


if __name__ == "__main__":
    main()


# ── Set I / Set II configs ─────────────────────────────────────────────────────

SET_I_CONFIGS = [
    {1:5}, {1:10}, {1:20},
    {2:5}, {2:10}, {2:20},
    {3:5}, {3:10}, {3:20},
    {4:5}, {4:10}, {4:20},
    {1:5,  2:5},  {1:10, 2:10},
    {1:5,  3:5},  {1:10, 3:10},
    {1:5,  4:5},  {1:10, 4:10},
    {2:5,  3:5},  {2:10, 3:10},
    {2:5,  4:5},  {2:10, 4:10},
    {3:5,  4:5},  {3:10, 4:10},
]

SET_II_CONFIGS = [
    {1:60}, {1:120},
    {2:60}, {2:120},
    {3:60}, {3:120},
    {4:60}, {4:120},
    {1:20,  2:20}, {1:20,  3:20}, {1:20,  4:20},
    {2:20,  3:20}, {2:20,  4:20}, {3:20,  4:20},
    {1:5,  2:5,  3:5},  {1:5,  2:5,  4:5},
    {1:5,  3:5,  4:5},  {2:5,  3:5,  4:5},
    {1:10, 2:10, 3:10}, {1:10, 2:10, 4:10},
    {1:10, 3:10, 4:10}, {2:10, 3:10, 4:10},
    {1:5,  2:5,  3:5,  4:5},
    {1:10, 2:10, 3:10, 4:10},
]

# Paper Table 2 reference (for comparison)
PAPER_TABLE2 = {
    "{1:5}":         (27, 131,  341,  25, 10,  25,  59),
    "{1:10}":        (52, 261, 1206,  50, 20,  25, 100),
    "{1:20}":       (102, 521, 4511, 100, 40,  25, 185),
    "{2:5}":         (62, 541, 1501,  60, 20,  62, 128),
    "{2:10}":       (122,1081, 5401, 120, 40,  62, 200),
    "{2:20}":       (242,2161,20401, 240, 80,  62, 365),
    "{3:5}":         (37, 251,  681,  35, 15,  44,  88),
    "{3:10}":        (72, 501, 2436,  70, 30,  44, 139),
    "{3:20}":       (142,1001, 9171, 140, 60,  44, 249),
    "{4:5}":         (37, 206,  546,  35, 15,  38,  66),
    "{4:10}":        (72, 411, 1941,  70, 30,  38, 101),
    "{4:20}":       (142, 821, 7281, 140, 60,  38, 171),
    "{1:5,2:5}":     (87, 671, 1891,  85, 30,  62, 128),
    "{1:10,2:10}":  (172,1341, 6606, 170, 60,  62, 200),
}


def tc_label(tc):
    return "{" + ",".join(f"{t}:{c}" for t,c in sorted(tc.items())) + "}"


def gen_set_i(out_dir, verbose=True):
    """Generate all 24 Set I instances and print Table-2-style comparison."""
    make_dir(out_dir)
    gen = SAMSPGenerator(seed=0)
    print("\n%-20s %6s %6s  %8s %8s  %8s %8s  %7s %7s" % (
        "Instance","Our_n","Pap_n","OurPr_nb","PapPr_nb","OurPr_wb","PapPr_wb","OurLBwb","PapLBwb"))
    print("-"*90)
    for tc in SET_I_CONFIGS:
        inst = gen.from_paper_types(tc)
        stats = inst.compute_stats()
        path = os.path.join(out_dir, tc_label(tc).replace(",","-").replace(":","")+".json")
        inst.to_json(path)
        ref = PAPER_TABLE2.get(tc_label(tc))
        pn,ppr_nb,ppr_wb,_,_,_,plb_wb = ref if ref else (0,0,0,0,0,0,0)
        print("%-20s %6d %6d  %8d %8d  %8d %8d  %7d %7d" % (
            tc_label(tc),
            stats["n_acts"], pn,
            stats["prec_no_sb"], ppr_nb,
            stats["prec_w_sb"], ppr_wb,
            stats["lb_w_sb"], plb_wb,
        ))
    print(f"\nSet I: {len(SET_I_CONFIGS)} instances saved to {out_dir}")


def gen_set_ii(out_dir, verbose=True):
    """Generate all 24 Set II instances."""
    make_dir(out_dir)
    gen = SAMSPGenerator(seed=0)
    print("\nGenerating Set II (%d instances)..." % len(SET_II_CONFIGS))
    for tc in SET_II_CONFIGS:
        inst = gen.from_paper_types(tc)
        stats = inst.compute_stats()
        path = os.path.join(out_dir, tc_label(tc).replace(",","-").replace(":","")+".json")
        inst.to_json(path)
        errors = inst.validate()
        status = "OK" if not errors else "ERR"
        print("  %-30s  n=%4d  LBwSB=%5d  prec_nb=%6d  prec_wb=%7d  %s" % (
            tc_label(tc), stats["n_acts"], stats["lb_w_sb"],
            stats["prec_no_sb"], stats["prec_w_sb"], status))
    print(f"Set II: {len(SET_II_CONFIGS)} instances saved to {out_dir}")
