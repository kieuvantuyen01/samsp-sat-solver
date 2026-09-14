"""
samsp/validator.py

Validate a concrete schedule (dict: activity → start_time) against
a SAMSPProblem instance.

Returns list of violations (empty = valid).
"""
from __future__ import annotations

from typing import Dict, List

from src.samsp.problem import SAMSPProblem


def validate_schedule(
    problem: SAMSPProblem,
    schedule: Dict[int, int],
) -> List[str]:
    """
    Validate a concrete schedule.

    schedule[i] = start time of activity i.

    Checks:
      1. All activities scheduled
      2. Lag constraints: S_j - S_i >= ell for all (i, j, ell) in lags
      3. Resource constraints: at most B_r activities using resource r overlap
      4. Storage constraints: O_s(t) in [0, P_s] at every time point
    """
    errors = []
    n = problem.number_of_activities
    d = problem.durations
    lags = problem.lags
    n_res = problem.number_of_resources
    caps = problem.capacities
    storages = problem.storages
    deltas = problem.storage_deltas

    # 1. All activities present
    for i in range(n):
        if i not in schedule:
            errors.append(f"Activity {i} not scheduled.")
    if errors:
        return errors

    S = schedule
    horizon = max(S[i] + d[i] for i in range(n))

    # 2. Lag constraints
    for (i, j, ell) in lags:
        diff = S[j] - S[i]
        if diff < ell:
            errors.append(
                f"Lag violation: S[{j}]-S[{i}] = {diff} < {ell}"
            )

    # 3. Resource constraints
    for r in range(n_res):
        B = caps[r]
        for t in range(horizon):
            usage = sum(
                1
                for i in range(n)
                if (problem.requests[i][r] > 0
                    and S[i] <= t < S[i] + d[i])
            )
            if usage > B:
                errors.append(
                    f"Resource {r} overloaded at t={t}: {usage} > {B}"
                )

    # 4. Storage constraints
    if storages:
        for s_idx, sz in enumerate(storages):
            # Build occupancy trace
            occ = [0] * (horizon + 1)
            for i in range(n):
                if s_idx < len(deltas[i]):
                    c = deltas[i][s_idx]
                    if c > 0:
                        # occupies at start of i
                        for t in range(S[i], horizon + 1):
                            occ[t] += c
                    elif c < 0:
                        # vacates at completion of i
                        for t in range(S[i] + d[i], horizon + 1):
                            occ[t] += c  # c is negative

            for t in range(horizon + 1):
                if occ[t] < 0:
                    errors.append(
                        f"Storage '{sz.name}' occupancy negative at t={t}: {occ[t]}"
                    )
                if occ[t] > sz.capacity:
                    errors.append(
                        f"Storage '{sz.name}' overflow at t={t}: {occ[t]} > {sz.capacity}"
                    )

    return errors
