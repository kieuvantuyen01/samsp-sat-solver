# SAMSP — SAT-based Solver for the Sample Analysis Machine Scheduling Problem

Research codebase for solving the **Sample Analysis Machine Scheduling Problem (SAMSP)** using SAT/MaxSAT encodings, with a focus on instance generation and preprocessing.

Based on: *Bofill et al. (2022) — "The Sample Analysis Machine Scheduling Problem: Definition and comparison of exact solving approaches"*

---

## Structure

```
SAMSP/
├── src/
│   ├── samsp/              # Core SAMSP module
│   │   ├── problem.py      # SAMSPProblem: max-plus closure, time windows, validation
│   │   ├── generator.py    # SAMSPGenerator: paper test types + synthetic instances
│   │   └── validator.py    # Schedule feasibility checker
│   └── rcpsp/              # Base RCPSP module (extended by SAMSP)
│       ├── problem.py
│       └── solver.py
├── tests/
│   └── test_samsp_generator.py
├── data/
│   ├── set_i/              # 24 Set I instances (paper Table 2)
│   └── set_ii/             # 24 Set II instances (paper Table 3)
├── docs/
│   └── mo_hinh_sat_samsp.tex  # Research notes / SAT model (Vietnamese)
├── generate_dataset.py     # CLI script for dataset generation
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run tests
python3 tests/test_samsp_generator.py

# Generate Set I instances (matches paper Table 2)
python3 generate_dataset.py --set-i --out data/

# Generate Set II instances (matches paper Table 3)
python3 generate_dataset.py --set-ii --out data/

# Generate synthetic benchmark
python3 generate_dataset.py --preset medium --n 100 --seeds 0 1 2 --out data/
```

---

## Dataset Generation

### Paper Test Types (Table 1)

| Type | Acts | Prec | LB | Shuttle | S.Arm | R.Arm | Storages |
|------|------|------|-----|---------|-------|-------|----------|
| 1 | 7 | 8 | 25 | 3 | 1 | 1 | HotI, Obs |
| 2 | 14 | 17 | 62 | 6 | 5 | 1 | Cold×2, HotII, Obs |
| 3 | 9 | 11 | 44 | 4 | 1 | 2 | HotI, HotII, Obs |
| 4 | 9 | 10 | 38 | 4 | 1 | 2 | Cold, HotII, Obs |

### Instance naming

- `{1:5}` → 5 copies of Type 1 → 27 activities
- `{1:5,2:5}` → mixed 5×Type1 + 5×Type2 → 87 activities

### CLI modes

| Command | Description |
|---------|-------------|
| `--set-i` | All 24 Set I instances |
| `--set-ii` | All 24 Set II instances |
| `--paper --types 1 2 --copies 10 10` | Custom paper type config |
| `--preset easy\|medium\|hard -n N` | Synthetic random instances |
| `--all` | Everything (Set I + II + synthetic) |
| `--out DIR` | Output directory (default: `./data`) |
| `--seeds 0 1 2` | Random seeds |

---

## Instance Format (JSON)

```json
{
  "number_of_activities": 27,
  "number_of_resources": 3,
  "durations": [...],
  "requests": [...],
  "capacities": [1, 1, 1],
  "lags": [[0,1,0], [1,2,2], ...],
  "storages": [{"name":"Cold","capacity":4}, ...],
  "storage_deltas": [...],
  "metadata": {
    "lower_bound": 49,
    "sha256": "124dd3b36fb96a8d",
    "generator": "from_paper_types"
  }
}
```

---

## References

- Bofill et al. (2022). *The Sample Analysis Machine Scheduling Problem*. EJOR.
- Schutt et al. (2013). *Solving RCPSP with Lazy Clause Generation*.
- Biere et al. (2021). *Handbook of Satisfiability (2nd ed.)*.
