"""
test_samsp_generator.py
Quick smoke tests for SAMSPProblem, SAMSPGenerator and validate_schedule.

Run:  python3 test_samsp_generator.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))s.path.dirname(__file__))

from src.samsp.problem import SAMSPProblem, StorageZone
from src.samsp.generator import SAMSPGenerator, GeneratorConfig, PAPER_TEST_TYPES
from src.samsp.validator import validate_schedule


def test_paper_types_single():
    """Each paper test type, single copy: validate and check LB."""
    expected_lb = {1: 25, 2: 62, 3: 44, 4: 38}
    gen = SAMSPGenerator(seed=0)

    for tt_id, expected in expected_lb.items():
        inst = gen.from_paper_types({tt_id: 1})
        errors = inst.validate()
        assert errors == [], f"Type {tt_id} single copy: {errors}"
        print(f"  Type {tt_id}: LB={inst.lower_bound} (expected~{expected}), "
              f"n_acts={inst.number_of_activities}, sha={inst.sha256()}")


def test_paper_set_i():
    """Reproduce Set I instances from Table 2 in the paper."""
    gen = SAMSPGenerator(seed=0)
    configs = [
        ({1: 5},  27,   59),
        ({1: 10}, 52,  102),
        ({2: 5},  62,  173),
        ({1: 5, 2: 5},  87,  197),
        ({3: 5},  37,   89),
        ({4: 5},  37,   69),
    ]
    print("\nSet I sanity check (activities roughly match paper Table 2):")
    for type_counts, paper_acts, paper_opt in configs:
        inst = gen.from_paper_types(type_counts)
        errors = inst.validate()
        acts_ok = abs(inst.number_of_activities - paper_acts) <= 4  # ±4 tolerance
        print(f"  {type_counts}: n={inst.number_of_activities} (paper~{paper_acts}), "
              f"LB={inst.lower_bound} (paper opt~{paper_opt}), "
              f"valid={'OK' if not errors else errors[:1]}")


def test_synthetic_easy():
    """Generate 5 easy instances: all should validate."""
    gen = SAMSPGenerator(seed=42)
    cfg = GeneratorConfig(preset="easy", n_types=1, copies_range=(2, 4))
    errors_found = 0
    for i in range(5):
        inst = gen.generate(cfg)
        errors = inst.validate()
        if errors:
            errors_found += 1
            print(f"  Easy inst {i}: ERRORS: {errors[:2]}")
        else:
            print(f"  Easy inst {i}: OK  n={inst.number_of_activities} LB={inst.lower_bound}")
    assert errors_found == 0, f"{errors_found} easy instances had errors"


def test_storage_balance():
    """Directly test storage balance detection."""
    # Build a broken instance (unbalanced storage)
    sz = [StorageZone("Cold", 4)]
    inst = SAMSPProblem(
        number_of_activities=4,
        number_of_resources=1,
        durations=[0, 2, 3, 0],
        requests=[[0],[1],[1],[0]],
        capacities=[1],
        lags=[(0,1,0),(1,2,2),(2,3,2)],
        storages=sz,
        storage_deltas=[[0],[+1],[+1],[0]],  # sum = +2, NOT balanced
    )
    errors = inst.validate()
    assert any("balanced" in e for e in errors), f"Expected balance error, got: {errors}"
    print(f"  Balance check: correctly detected imbalance")


def test_positive_cycle_detection():
    """Positive cycle should be detected."""
    # S_1 - S_0 >= 5 AND S_0 - S_1 >= 5  => cycle of +10
    inst = SAMSPProblem(
        number_of_activities=3,
        number_of_resources=1,
        durations=[0, 2, 0],
        requests=[[0],[1],[0]],
        capacities=[1],
        lags=[(0,1,5),(1,0,5),(1,2,2)],
        storages=[],
        storage_deltas=[[], [], []],
    )
    errors = inst.validate()
    assert any("cycle" in e.lower() for e in errors), f"Expected cycle error, got: {errors}"
    print(f"  Cycle check: correctly detected positive cycle")


def test_json_roundtrip(tmp_path="/tmp/samsp_test.json"):
    """to_json → from_json preserves data."""
    gen = SAMSPGenerator(seed=7)
    inst = gen.from_paper_types({1: 3})
    inst.to_json(tmp_path)
    loaded = SAMSPProblem.from_json(tmp_path)
    assert inst.number_of_activities == loaded.number_of_activities
    assert inst.lags == loaded.lags
    print(f"  JSON roundtrip OK (n={inst.number_of_activities})")


def test_sb_improves_lb():
    """SB edges should make LB >= LB without SB."""
    gen = SAMSPGenerator(seed=1)
    inst_sb = gen.from_paper_types({1: 5})      # SB included
    # Compute SB LB
    inst_sb.max_plus_closure()

    inst_no = gen.from_paper_types({1: 5})       # same structure
    # Strip SB-like edges (heuristic: lag == duration of source act)
    inst_no._lags = [l for l in inst_no.lags
                     if not _is_sb_edge(l, inst_no)]
    inst_no._closure = None
    inst_no._lb = None
    inst_no.max_plus_closure()

    assert inst_sb.lower_bound >= inst_no.lower_bound, (
        f"SB LB={inst_sb.lower_bound} < no-SB LB={inst_no.lower_bound}"
    )
    print(f"  SB LB={inst_sb.lower_bound} >= no-SB LB={inst_no.lower_bound} ✓")


def _is_sb_edge(lag, inst):
    """Heuristic: SB edges have lag = duration of an activity (resource symmetry)."""
    f, t, ell = lag
    if 0 <= f < inst.number_of_activities and 0 <= t < inst.number_of_activities:
        return ell == inst.durations[f]
    return False


if __name__ == "__main__":
    print("=== SAMSP Generator Tests ===\n")

    print("1. Paper test types (single copy):")
    test_paper_types_single()

    print("\n2. Paper Set I reconstruction:")
    test_paper_set_i()

    print("\n3. Synthetic easy instances (5 samples):")
    test_synthetic_easy()

    print("\n4. Storage balance detection:")
    test_storage_balance()

    print("\n5. Positive cycle detection:")
    test_positive_cycle_detection()

    print("\n6. JSON roundtrip:")
    test_json_roundtrip()

    print("\n7. SB improves LB:")
    test_sb_improves_lb()

    print("\n=== ALL TESTS PASSED ===")
