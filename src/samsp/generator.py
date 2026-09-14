"""
samsp/generator.py  —  SAMSPGenerator (Parameterized Template, Approach 2)

Paper machine:
  Resources : Shuttle(1), Sample Arm(1), Reagent Arm(1)   [indices 0,1,2]
  Storages  : Cold(4), Hot-I(4), Hot-II(4), Observation(2) [indices 0,1,2,3]

Key decisions (aligned with paper §10 of technical note):
  - Storage events generated in pairs (put_act, vacate_act)
  - d_range and lag_range are separate  (lag/duration ≈ 5-10× in paper)
  - multi_event: one activity may vacate s' AND occupy s simultaneously
  - Max lags added only when they do not introduce a positive cycle
  - SB-R and SB-C edges are added BEFORE max-plus closure
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.samsp.problem import SAMSPProblem, StorageZone


# -----------------------------------------------------------------------
# Paper machine configuration
# -----------------------------------------------------------------------

PAPER_MACHINE = {
    "resources": [
        {"name": "Shuttle",     "capacity": 1},
        {"name": "Sample Arm",  "capacity": 1},
        {"name": "Reagent Arm", "capacity": 1},
    ],
    "storages": [
        {"name": "Cold",        "capacity": 4},
        {"name": "Hot-I",       "capacity": 4},
        {"name": "Hot-II",      "capacity": 4},
        {"name": "Observation", "capacity": 2},
    ],
}

# -----------------------------------------------------------------------
# Paper's 4 test types (validated: LB ≈ paper Table 1, no cycles, balanced)
#
# Activity format: (duration, resource_idx, [Cold, HotI, HotII, Obs])
# Lag format:      (from_local, to_local, ell)   meaning S_j - S_i >= ell
# Max lag S_j - S_i <= L is encoded as reverse edge (j, i, -L)
#
# Index 0 = dummy start,  last index = dummy end
# -----------------------------------------------------------------------

PAPER_TEST_TYPES: Dict[int, Dict] = {

    # ---- Type 1: 7 nodes (5 real + 2 dummies), LB=25 ----
    # Table 1: Act=7, Prec=8, Shuttle=3, S.Arm=1, R.Arm=1, HotI=1, Obs=1
    1: {
        "description": "Type 1 — HotI -> Observation protocol (LB=25)",
        "activities": [
            (0, -1, [0,  0,  0,  0]),   # 0  dummy start
            (2,  0, [0, +1,  0,  0]),   # 1  fetch -> Hot-I    (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 2  aspirate          (S.Arm)
            (2,  0, [0, -1,  0, +1]),   # 3  Hot-I -> Obs      (Shuttle)
            (2,  2, [0,  0,  0,  0]),   # 4  add reagent       (R.Arm)
            (2,  0, [0,  0,  0, -1]),   # 5  Obs -> waste      (Shuttle)
            (0, -1, [0,  0,  0,  0]),   # 6  dummy end
        ],
        "lags": [
            (0, 1,  0), (1, 2,  2), (2, 3,  5), (3, 4,  4),
            (4, 5,  8), (5, 6,  2),
            (1, 5, 23),   # critical path: LB = 0+23+2 = 25
            (3, 4, -8),   # max lag: S_4 - S_3 <= 8
        ],
    },

    # ---- Type 2: 14 nodes (12 real + 2 dummies), LB=62 ----
    # Table 1: Act=14, Prec=17, Shuttle=6, S.Arm=5, R.Arm=1, Cold=2, HotII=1, Obs=1
    2: {
        "description": "Type 2 — two-cuvette Cold->HotII->Obs protocol (LB=62)",
        "activities": [
            (0, -1, [0,  0,  0,  0]),   # 0  dummy start
            (2,  0, [+1, 0,  0,  0]),   # 1  C1 fetch -> Cold  (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 2  C1 aspirate       (S.Arm)
            (2,  0, [+1, 0,  0,  0]),   # 3  C2 fetch -> Cold  (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 4  C2 aspirate       (S.Arm)
            (2,  0, [-1, 0, +1,  0]),   # 5  C1 Cold->HotII    (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 6  C1 add diluent    (S.Arm)
            (3,  1, [0,  0,  0,  0]),   # 7  C1 mix            (S.Arm)
            (2,  0, [-1, 0,  0,  0]),   # 8  C2 dispose Cold   (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 9  C1 process HotII  (S.Arm)
            (2,  0, [0,  0, -1, +1]),   # 10 C1 HotII->Obs     (Shuttle)
            (2,  2, [0,  0,  0,  0]),   # 11 C1 add reagent    (R.Arm)
            (2,  0, [0,  0,  0, -1]),   # 12 C1 Obs->waste     (Shuttle)
            (0, -1, [0,  0,  0,  0]),   # 13 dummy end
        ],
        "lags": [
            (0,  1,  0), (0,  3,  0),
            (1,  2,  2), (2,  5,  3),
            (3,  4,  2), (4,  6,  3), (6,  7,  3),
            (7,  8,  2), (5,  9, 25),
            (9, 10,  2), (10, 11,  6), (11, 12,  2), (12, 13, 2),
            (1, 10, 52),    # critical path: LB = 0+52+6+2+2 = 62
            (5,  8,  0),    # after C1 transfer, C2 can be disposed
            (7,  9,  3),    # after mix, C1 HotII processing begins
            (11, 10, -9),   # max lag: S_11 - S_10 <= 9
        ],
    },

    # ---- Type 3: 9 nodes (7 real + 2 dummies), LB=44 ----
    # Table 1: Act=9, Prec=11, Shuttle=4, S.Arm=1, R.Arm=2, HotI=1, HotII=1, Obs=1
    3: {
        "description": "Type 3 — HotI -> HotII -> Obs protocol (LB=44)",
        "activities": [
            (0, -1, [0,  0,  0,  0]),   # 0  dummy start
            (2,  0, [0, +1,  0,  0]),   # 1  fetch -> Hot-I    (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 2  aspirate          (S.Arm)
            (2,  2, [0,  0,  0,  0]),   # 3  add reagent 1     (R.Arm)
            (2,  0, [0, -1, +1,  0]),   # 4  Hot-I -> Hot-II   (Shuttle)
            (2,  2, [0,  0,  0,  0]),   # 5  add reagent 2     (R.Arm)
            (2,  0, [0,  0, -1, +1]),   # 6  Hot-II -> Obs     (Shuttle)
            (2,  0, [0,  0,  0, -1]),   # 7  Obs -> waste      (Shuttle)
            (0, -1, [0,  0,  0,  0]),   # 8  dummy end
        ],
        "lags": [
            (0, 1,  0), (1, 2,  2), (2, 3,  3), (3, 4,  4),
            (4, 5, 14), (5, 6,  4), (6, 7,  6), (7, 8,  2),
            (2, 6, 34),    # incubation: 0->1->2->6->7->end = 0+2+34+6+2 = 44
            (4, 5, -18),   # max lag: S_5 - S_4 <= 18
            (1, 7, 42),    # direct: 0->1->7->end = 0+42+2 = 44
        ],
    },

    # ---- Type 4: 9 nodes (7 real + 2 dummies), LB=38 ----
    # Table 1: Act=9, Prec=10, Shuttle=4, S.Arm=1, R.Arm=2, Cold=1, HotII=1, Obs=1
    4: {
        "description": "Type 4 — Cold -> HotII -> Obs protocol (LB=38)",
        "activities": [
            (0, -1, [0,  0,  0,  0]),   # 0  dummy start
            (2,  0, [+1, 0,  0,  0]),   # 1  fetch -> Cold     (Shuttle)
            (3,  1, [0,  0,  0,  0]),   # 2  aspirate          (S.Arm)
            (2,  0, [-1, 0, +1,  0]),   # 3  Cold -> Hot-II    (Shuttle)
            (2,  2, [0,  0,  0,  0]),   # 4  add reagent 1     (R.Arm)
            (2,  0, [0,  0, -1, +1]),   # 5  Hot-II -> Obs     (Shuttle)
            (2,  2, [0,  0,  0,  0]),   # 6  add reagent 2     (R.Arm)
            (2,  0, [0,  0,  0, -1]),   # 7  Obs -> waste      (Shuttle)
            (0, -1, [0,  0,  0,  0]),   # 8  dummy end
        ],
        "lags": [
            (0, 1,  0), (1, 2,  2), (2, 3,  3), (3, 4,  8),
            (4, 5,  4), (5, 6,  6), (6, 7,  2), (7, 8,  2),
            (2, 5, 26),    # incubation: 0->1->2->5->6->7->end = 0+2+26+6+2+2 = 38
            (3, 4, -12),   # max lag: S_4 - S_3 <= 12
        ],
    },
}



# -----------------------------------------------------------------------
# Generator configuration
# -----------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Parameters for SAMSPGenerator.generate()."""
    n_types: int = 1
    copies_range: Tuple[int, int] = (5, 10)
    n_acts_range: Tuple[int, int] = (6, 12)

    # Durations (fast operations: shuttle/arm)
    d_range: Tuple[int, int] = (2, 5)
    # Chemical incubation lags (separate from d_range; ratio ≈5-10× paper)
    lag_range: Tuple[int, int] = (8, 30)
    lag_density: float = 0.3        # P(add shortcut lag between non-adjacent acts)
    max_lag_ratio: float = 0.30     # P(min lag also gets a max lag)
    max_lag_slack: Tuple[float, float] = (0.3, 1.5)  # L = ell*(1+slack)

    # Resources
    n_resources: int = 3
    resource_caps: List[int] = field(default_factory=lambda: [1, 1, 1])

    # Storages
    storages: List[Dict] = field(default_factory=lambda: [
        {"name": "Cold",        "capacity": 4},
        {"name": "Hot-I",       "capacity": 4},
        {"name": "Hot-II",      "capacity": 4},
        {"name": "Observation", "capacity": 2},
    ])

    # Storage events
    storage_density: float = 0.50   # P(activity participates in a put event)
    multi_event: bool = True        # allow vacate s' + occupy s in same activity

    preset: str = "medium"


DIFFICULTY_PRESETS = {
    "easy":   {"copies_range": (1, 5),   "n_types": 1,
               "max_lag_ratio": 0.10, "storage_density": 0.30},
    "medium": {"copies_range": (5, 20),  "n_types": 2,
               "max_lag_ratio": 0.30, "storage_density": 0.50},
    "hard":   {"copies_range": (20, 60), "n_types": 4,
               "max_lag_ratio": 0.50, "storage_density": 0.60},
}


# -----------------------------------------------------------------------
# SAMSPGenerator
# -----------------------------------------------------------------------

class SAMSPGenerator:
    """
    Usage:
        gen = SAMSPGenerator(seed=42)
        inst = gen.from_paper_types({1: 5, 2: 5})
        inst = gen.generate(GeneratorConfig(preset='medium'))
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._base_seed = seed

    # ------------------------------------------------------------------
    # from_paper_types
    # ------------------------------------------------------------------

    def from_paper_types(
        self,
        type_counts: Dict[int, int],
        machine: Dict = None,
    ) -> SAMSPProblem:
        """Build instance from paper's 4 test types with SB edges."""
        if machine is None:
            machine = PAPER_MACHINE

        storages = [StorageZone(s["name"], s["capacity"]) for s in machine["storages"]]
        resource_caps = [r["capacity"] for r in machine["resources"]]
        n_resources = len(resource_caps)
        n_storages = len(storages)

        # Accumulate across copies
        global_acts: List[Tuple] = []   # (duration, demands, storage_deltas)
        global_lags: List[Tuple] = []

        # Index 0 = global dummy start
        global_acts.append((0, [0]*n_resources, [0]*n_storages))
        act_offset = 1

        # Map (tt_id, copy_idx, local_idx) -> global_idx
        sb_groups: Dict[int, List[List[int]]] = {}  # tt_id -> per-copy global indices
        deferred_lags = []

        for tt_id, n_copies in sorted(type_counts.items()):
            tt = PAPER_TEST_TYPES[tt_id]
            tt_acts = tt["activities"]
            tt_lags_local = tt["lags"]
            n_local = len(tt_acts)

            copy_indices: List[List[int]] = []

            for copy_idx in range(n_copies):
                indices = list(range(act_offset, act_offset + (n_local - 2)))
                copy_indices.append(indices)

                for loc in range(1, n_local - 1):
                    dur, res_idx, st_d = tt_acts[loc]
                    demands = [0] * n_resources
                    if 0 <= res_idx < n_resources:
                        demands[res_idx] = 1
                    global_acts.append((dur, demands, list(st_d)))

                deferred_lags.append((act_offset, tt_lags_local, n_local))

                # Dummy start → first real activity of copy
                global_lags.append((0, act_offset, 0))

                act_offset += (n_local - 2)

            sb_groups[tt_id] = copy_indices

            # ---- SB-R: resource ordering between consecutive copies ----
            for cp in range(n_copies - 1):
                for loc in range(1, n_local - 1):
                    dur, res_idx, _ = tt_acts[loc]
                    if res_idx < 0:
                        continue
                    g_i = copy_indices[cp][loc - 1]
                    g_j = copy_indices[cp + 1][loc - 1]
                    global_lags.append((g_i, g_j, dur))

            # ---- SB-C: storage capacity ordering ----
            for s_idx, sz in enumerate(storages):
                for loc in range(1, n_local - 1):
                    _, _, st_d = tt_acts[loc]
                    c = st_d[s_idx] if s_idx < len(st_d) else 0
                    if c <= 0:
                        continue
                    h = sz.capacity // c
                    if h <= 0 or h >= n_copies:
                        continue
                    pt = self._compute_pt(tt_acts, tt_lags_local, loc, s_idx, n_local)
                    if pt is None or pt <= 0:
                        continue
                    for cp in range(n_copies - h):
                        g_i = copy_indices[cp][loc - 1]
                        g_j = copy_indices[cp + h][loc - 1]
                        global_lags.append((g_i, g_j, pt))

        # Global dummy end
        n_end = act_offset
        global_acts.append((0, [0]*n_resources, [0]*n_storages))

        # Resolve deferred intra-copy lags
        for act_off, tt_lags_local, n_local in deferred_lags:
            for (f, t, ell) in tt_lags_local:
                g_f = 0 if f == 0 else (n_end if f == n_local - 1 else act_off + (f - 1))
                g_t = 0 if t == 0 else (n_end if t == n_local - 1 else act_off + (t - 1))
                global_lags.append((g_f, g_t, ell))

        # Connect last real activity of each copy to dummy end
        for tt_id, copy_indices in sb_groups.items():
            tt_acts = PAPER_TEST_TYPES[tt_id]["activities"]
            n_local = len(tt_acts)
            last_real_local = n_local - 2
            last_dur = tt_acts[last_real_local][0]   # duration of real last act
            for indices in copy_indices:
                global_lags.append((indices[-1], n_end, last_dur))

        n_total = n_end + 1
        durations      = [a[0] for a in global_acts]
        requests       = [a[1] for a in global_acts]
        storage_deltas = [a[2] for a in global_acts]

        return SAMSPProblem(
            number_of_activities=n_total,
            number_of_resources=n_resources,
            durations=durations,
            requests=requests,
            capacities=resource_caps,
            lags=global_lags,
            storages=storages,
            storage_deltas=storage_deltas,
            test_metadata={
                "type_counts": type_counts,
                "generator": "from_paper_types",
                "seed": self._base_seed,
            },
        )

    # ------------------------------------------------------------------
    # generate (Approach 2 — Parameterized Template)
    # ------------------------------------------------------------------

    def generate(self, config: GeneratorConfig) -> SAMSPProblem:
        """Generate a random SAMSP instance from a GeneratorConfig."""
        preset = DIFFICULTY_PRESETS.get(config.preset, {})
        for k, v in preset.items():
            setattr(config, k, v)

        storages = [StorageZone(s["name"], s["capacity"]) for s in config.storages]
        n_resources = config.n_resources
        n_storages = len(storages)

        global_acts: List[Tuple] = []
        global_lags: List[Tuple] = []

        # Dummy start
        global_acts.append((0, [0]*n_resources, [0]*n_storages))
        act_offset = 1

        all_groups = []  # (copy_indices_list, n_local, durations_local)

        for _tt in range(config.n_types):
            n_local = self._rng.randint(*config.n_acts_range)
            n_copies = self._rng.randint(*config.copies_range)
            dur_local = [self._rng.randint(*config.d_range) for _ in range(n_local)]
            res_local = [self._rng.randint(0, n_resources - 1) for _ in range(n_local)]

            tt_lags = self._gen_lags(n_local, dur_local, config)
            st_deltas = self._gen_storage_events(
                n_local, n_storages, storages, tt_lags, dur_local, config
            )

            copy_indices = []
            for _cp in range(n_copies):
                indices = list(range(act_offset, act_offset + n_local))
                copy_indices.append(indices)

                for i in range(n_local):
                    demands = [0] * n_resources
                    demands[res_local[i]] = 1
                    global_acts.append((dur_local[i], demands, list(st_deltas[i])))

                for (f, t, ell) in tt_lags:
                    global_lags.append((act_offset + f, act_offset + t, ell))

                global_lags.append((0, act_offset, 0))
                act_offset += n_local

            all_groups.append((copy_indices, n_local, dur_local))

            # SB-R
            for cp in range(n_copies - 1):
                for loc in range(n_local):
                    g_i = copy_indices[cp][loc]
                    g_j = copy_indices[cp + 1][loc]
                    global_lags.append((g_i, g_j, dur_local[loc]))

        # Dummy end
        n_end = act_offset
        global_acts.append((0, [0]*n_resources, [0]*n_storages))
        for copy_indices, n_local, dur_local in all_groups:
            for indices in copy_indices:
                global_lags.append((indices[n_local - 1], n_end, dur_local[n_local - 1]))

        n_total = n_end + 1
        return SAMSPProblem(
            number_of_activities=n_total,
            number_of_resources=n_resources,
            durations=[a[0] for a in global_acts],
            requests=[a[1] for a in global_acts],
            capacities=list(config.resource_caps),
            lags=global_lags,
            storages=storages,
            storage_deltas=[a[2] for a in global_acts],
            test_metadata={
                "n_types": config.n_types,
                "preset": config.preset,
                "generator": "parameterized_template",
                "seed": self._base_seed,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gen_lags(self, n: int, durations: List[int], cfg: GeneratorConfig):
        lags = []
        # Backbone chain with chemical incubation lags
        for i in range(n - 1):
            ell = self._rng.randint(*cfg.lag_range)
            lags.append((i, i + 1, ell))

        # Shortcut min-lags
        for i in range(n):
            for j in range(i + 2, n):
                if self._rng.random() < cfg.lag_density:
                    lags.append((i, j, self._rng.randint(*cfg.lag_range)))

        # Max lags (only if no positive cycle introduced)
        max_lags = []
        for (f, t, ell_min) in list(lags):
            if self._rng.random() < cfg.max_lag_ratio:
                slack = self._rng.uniform(*cfg.max_lag_slack)
                L = int(math.ceil(ell_min * (1.0 + slack)))
                cand = (t, f, -L)
                if not self._has_positive_cycle(lags + max_lags + [cand], n):
                    max_lags.append(cand)
        lags.extend(max_lags)
        return lags

    def _gen_storage_events(self, n, n_storages, storages, lags, durations, cfg):
        deltas = [[0] * n_storages for _ in range(n)]
        topo = self._topo_rank(n, lags)

        for s in range(n_storages):
            cap = storages[s].capacity
            n_cuvettes = self._rng.randint(0, cap)
            for _ in range(n_cuvettes):
                if self._rng.random() > cfg.storage_density:
                    continue
                puts = [i for i in range(n) if topo[i] < n - 1]
                if not puts:
                    continue
                put = self._rng.choice(puts)
                vacs = [i for i in range(n) if topo[i] > topo[put]]
                if not vacs:
                    continue
                vac = self._rng.choice(vacs)
                deltas[put][s] += 1
                deltas[vac][s] -= 1

                # multi_event: vac simultaneously enters another storage
                if cfg.multi_event and n_storages > 1:
                    s2s = [s2 for s2 in range(n_storages) if s2 != s
                           and sum(deltas[i][s2] for i in range(n)) < storages[s2].capacity]
                    if s2s and self._rng.random() < 0.4:
                        s2 = self._rng.choice(s2s)
                        vac2s = [i for i in range(n) if topo[i] > topo[vac]]
                        if vac2s:
                            vac2 = self._rng.choice(vac2s)
                            deltas[vac][s2] += 1
                            deltas[vac2][s2] -= 1
        return deltas

    def _topo_rank(self, n: int, lags: List[Tuple]) -> List[int]:
        in_deg = [0] * n
        adj = [[] for _ in range(n)]
        for (f, t, ell) in lags:
            if 0 <= f < n and 0 <= t < n and ell > 0 and f != t:
                adj[f].append(t)
                in_deg[t] += 1
        queue = [i for i in range(n) if in_deg[i] == 0]
        rank = [0] * n
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for u in adj[v]:
                in_deg[u] -= 1
                if in_deg[u] == 0:
                    queue.append(u)
        for r, v in enumerate(order):
            rank[v] = r
        return rank

    def _has_positive_cycle(self, lags: List[Tuple], n: int) -> bool:
        NEG_INF = -math.inf
        dist = [NEG_INF] * n
        dist[0] = 0.0
        for _ in range(n - 1):
            changed = False
            for (f, t, ell) in lags:
                if 0 <= f < n and 0 <= t < n and dist[f] > NEG_INF:
                    val = dist[f] + ell
                    if val > dist[t] + 1e-9:
                        dist[t] = val
                        changed = True
            if not changed:
                break
        for (f, t, ell) in lags:
            if 0 <= f < n and 0 <= t < n and dist[f] > NEG_INF:
                if dist[f] + ell > dist[t] + 1e-9:
                    return True
        return False

    def _compute_pt(self, tt_acts, tt_lags, local_a, s_idx, n):
        vacators = [
            j for j, (_, _, st) in enumerate(tt_acts)
            if s_idx < len(st) and st[s_idx] < 0
        ]
        if not vacators:
            return None
        NEG_INF = -math.inf
        dist = [NEG_INF] * n
        dist[local_a] = 0.0
        for _ in range(n):
            for (f, t, ell) in tt_lags:
                if 0 <= f < n and 0 <= t < n and ell > 0 and dist[f] > NEG_INF:
                    val = dist[f] + ell
                    if val > dist[t]:
                        dist[t] = val
        pts = [
            int(math.ceil(dist[v])) + tt_acts[v][0]
            for v in vacators if dist[v] > NEG_INF
        ]
        return min(pts) if pts else None
