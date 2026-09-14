"""
samsp/problem.py
SAMSPProblem: extends RCPSPProblem with generalized lags and storage zones.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.rcpsp.problem import RCPSPProblem


class StorageZone:
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity  # P_s

    def __repr__(self):
        return f"StorageZone({self.name!r}, cap={self.capacity})"


class TimeWindows:
    """ES, LS, EC, LC for every activity."""
    def __init__(self, ES, LS, EC, LC):
        self.ES = ES
        self.LS = LS
        self.EC = EC
        self.LC = LC


class SAMSPProblem(RCPSPProblem):
    """
    SAMSP instance = RCPSP + generalized lags + storage zones.

    Lags: list of (i, j, ell)  meaning  S_j - S_i >= ell.
    Maximum lag  S_j - S_i <= L  is encoded as  (j, i, -L).

    storage_deltas[i][s] = c_{i,s}:
        > 0  activity i occupies c_{i,s} positions in storage s at start
        < 0  activity i vacates  |c_{i,s}| positions in storage s at completion
        = 0  no interaction
    """

    def __init__(
        self,
        number_of_activities: int = None,
        number_of_resources: int = None,
        durations: List[int] = None,
        requests: List[List[int]] = None,
        capacities: List[int] = None,
        lags: List[Tuple[int, int, int]] = None,
        storages: List[StorageZone] = None,
        storage_deltas: List[List[int]] = None,
        test_metadata: Optional[Dict] = None,
    ):
        super().__init__(
            number_of_activities=number_of_activities,
            number_of_resources=number_of_resources,
            durations=durations,
            precedence_graph=nx.DiGraph(),
            requests=requests,
            capacities=capacities,
        )
        n = number_of_activities or 0
        s = len(storages) if storages else 0

        self._lags: List[Tuple[int, int, int]] = lags or []
        self._storages: List[StorageZone] = storages or []
        self._storage_deltas: List[List[int]] = (
            storage_deltas if storage_deltas is not None
            else [[0] * s for _ in range(n)]
        )
        self._test_metadata: Dict = test_metadata or {}

        # Computed by preprocessing
        self._closure: Optional[Dict] = None
        self._tw: Optional[TimeWindows] = None
        self._lb: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lags(self):
        return self._lags

    @property
    def storages(self):
        return self._storages

    @property
    def storage_deltas(self):
        return self._storage_deltas

    @property
    def n_storages(self):
        return len(self._storages)

    @property
    def metadata(self):
        return self._test_metadata

    @property
    def time_windows(self):
        return self._tw

    @property
    def lower_bound(self):
        return self._lb

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_rcpsp(cls, rcpsp: RCPSPProblem) -> "SAMSPProblem":
        """Lift an RCPSP instance into SAMSP (no storage, no max lag)."""
        lags = [
            (i, j, rcpsp.durations[i])
            for i, j in rcpsp.precedence_graph.edges()
        ]
        n = rcpsp.number_of_activities
        return cls(
            number_of_activities=n,
            number_of_resources=rcpsp.number_of_resources,
            durations=list(rcpsp.durations),
            requests=[list(r) for r in rcpsp.requests],
            capacities=list(rcpsp.capacities),
            lags=lags,
            storages=[],
            storage_deltas=[[]] * n,
            test_metadata={"source": "rcpsp", "file": str(rcpsp.file_path)},
        )

    @classmethod
    def from_json(cls, path: str) -> "SAMSPProblem":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sz = [StorageZone(s["name"], s["capacity"]) for s in data.get("storages", [])]
        acts = data["activities"]
        n = len(acts)
        return cls(
            number_of_activities=n,
            number_of_resources=len(data.get("resources", [])),
            durations=[a["duration"] for a in acts],
            requests=[a.get("demands", []) for a in acts],
            capacities=[r["capacity"] for r in data.get("resources", [])],
            lags=[(lg["from"], lg["to"], lg["lag"]) for lg in data.get("lags", [])],
            storages=sz,
            storage_deltas=[a.get("storage_deltas", [0] * len(sz)) for a in acts],
            test_metadata=data.get("metadata", {}),
        )

    def to_json(self, path: str) -> None:
        data = {
            "metadata": self._test_metadata,
            "resources": [
                {"id": r, "capacity": self.capacities[r]}
                for r in range(self.number_of_resources)
            ],
            "storages": [
                {"id": s, "name": self._storages[s].name, "capacity": self._storages[s].capacity}
                for s in range(self.n_storages)
            ],
            "activities": [
                {
                    "id": i,
                    "duration": self.durations[i],
                    "demands": self.requests[i] if self.requests else [],
                    "storage_deltas": self._storage_deltas[i],
                }
                for i in range(self.number_of_activities)
            ],
            "lags": [
                {"from": i, "to": j, "lag": int(ell)}
                for i, j, ell in self._lags
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Preprocessing: max-plus closure (Floyd-Warshall)
    # ------------------------------------------------------------------

    def max_plus_closure(self) -> Dict:
        """
        Compute max-plus closure L* of the lag graph.
        L*[i][j] = longest-path lag from i to j.
        Sets self._lb and self._closure.
        Raises ValueError on positive cycle.
        """
        n = self.number_of_activities
        NEG_INF = -math.inf

        # Build matrix as flat dict for speed
        L = [[NEG_INF] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = 0.0
        for (i, j, ell) in self._lags:
            if 0 <= i < n and 0 <= j < n:
                if L[i][j] < ell:
                    L[i][j] = float(ell)

        # Floyd-Warshall (max-plus)
        for k in range(n):
            for i in range(n):
                if L[i][k] == NEG_INF:
                    continue
                for j in range(n):
                    if L[k][j] == NEG_INF:
                        continue
                    val = L[i][k] + L[k][j]
                    if val > L[i][j]:
                        L[i][j] = val

        # Positive cycle check
        for i in range(n):
            if L[i][i] > 1e-9:
                raise ValueError(f"Positive cycle detected at activity {i}.")

        self._closure = L
        lb_val = L[0][n - 1]
        self._lb = max(0, int(math.ceil(lb_val))) if lb_val > NEG_INF else 0
        return L

    def compute_time_windows(self, upper_bound: int) -> TimeWindows:
        """Compute ES/LS/EC/LC. Calls closure if not computed yet."""
        if self._closure is None:
            self.max_plus_closure()

        n = self.number_of_activities
        d = self.durations
        L = self._closure
        NEG_INF = -math.inf

        ES, LS, EC, LC = [], [], [], []
        for i in range(n):
            es = max(0, int(math.ceil(L[0][i]))) if L[0][i] > NEG_INF else 0
            ls_raw = L[i][n - 1]
            ls = upper_bound - (max(0, int(math.ceil(ls_raw))) if ls_raw > NEG_INF else 0)
            ES.append(es)
            LS.append(ls)
            EC.append(es + d[i])
            LC.append(ls + d[i])

        self._tw = TimeWindows(ES=ES, LS=LS, EC=EC, LC=LC)
        return self._tw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Return list of error messages (empty = valid).
        Checks: storage balance, no positive cycle, ES <= LS.
        """
        errors = []
        n = self.number_of_activities

        # 1. Storage balance
        for s in range(self.n_storages):
            total = sum(
                self._storage_deltas[i][s]
                for i in range(n)
                if s < len(self._storage_deltas[i])
            )
            if total != 0:
                errors.append(
                    f"Storage '{self._storages[s].name}' not balanced: sum c_is = {total}"
                )

        # 2. Positive cycle
        try:
            self.max_plus_closure()
        except ValueError as e:
            errors.append(str(e))
            return errors

        # 3. Time windows
        ub = max(self._lb * 3, self._lb + 100)
        self.compute_time_windows(ub)
        for i in range(n):
            if self._tw.ES[i] > self._tw.LS[i]:
                errors.append(
                    f"Activity {i}: ES={self._tw.ES[i]} > LS={self._tw.LS[i]}"
                )
        return errors

    def sha256(self) -> str:
        """16-char deterministic checksum."""
        payload = json.dumps(
            {
                "n": self.number_of_activities,
                "d": self.durations,
                "lags": sorted([(i, j, int(e)) for i, j, e in self._lags]),
                "caps": self.capacities,
                "sc": [s.capacity for s in self._storages],
                "deltas": self._storage_deltas,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def __repr__(self):
        return (
            f"SAMSPProblem(n={self.number_of_activities}, "
            f"lags={len(self._lags)}, storages={self.n_storages})"
        )

    # ------------------------------------------------------------------
    # Instance statistics (Table 2 style)
    # ------------------------------------------------------------------

    def compute_stats(self, with_sb_lags: bool = True) -> dict:
        """
        Compute Table-2-style statistics for this instance.
        Returns dict with keys:
          n_acts, prec_no_sb, prec_w_sb, lb_no_sb, lb_w_sb,
          act_dem_renewable, act_dem_storage
        """
        n = self.number_of_activities

        # ── LB w/o SB (remove SB-R edges: lag == d[i] for resource acts) ──
        # Heuristic: SB edges are lags where ell == d[source] AND source is real
        no_sb_lags = [
            (f, t, e) for (f, t, e) in self._lags
            if not (0 < f < n - 1 and e == self.durations[f])
        ]
        L_no = self._floyd(no_sb_lags, n)
        lb_no_sb = max(0, int(math.ceil(L_no[0][n - 1]))) if L_no[0][n - 1] > -math.inf else 0

        # ── TC precedences w/o SB ──
        prec_no_sb = sum(
            1 for i in range(n) for j in range(n)
            if i != j and L_no[i][j] > -math.inf and L_no[i][j] >= 0
        )

        # ── LB w SB ──
        if self._closure is None:
            self.max_plus_closure()
        lb_w_sb = self._lb

        # ── TC precedences w SB ──
        prec_w_sb = sum(
            1 for i in range(n) for j in range(n)
            if i != j and self._closure[i][j] > -math.inf and self._closure[i][j] >= 0
        )

        # ── Activity demand counts ──
        act_dem_ren = sum(
            1 for i in range(1, n - 1)
            if self.requests and any(self.requests[i][r] > 0 for r in range(self.number_of_resources))
        )
        act_dem_st = sum(
            1 for i in range(1, n - 1)
            if self._storage_deltas and any(self._storage_deltas[i][s] > 0 for s in range(self.n_storages))
        )

        return {
            "n_acts": n,
            "prec_no_sb": prec_no_sb,
            "prec_w_sb": prec_w_sb,
            "lb_no_sb": lb_no_sb,
            "lb_w_sb": lb_w_sb,
            "act_dem_renewable": act_dem_ren,
            "act_dem_storage": act_dem_st,
        }

    def _floyd(self, lags, n):
        """Floyd-Warshall max-plus for an arbitrary lag list."""
        NEG_INF = -math.inf
        L = [[NEG_INF] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = 0.0
        for (f, t, e) in lags:
            if 0 <= f < n and 0 <= t < n and L[f][t] < e:
                L[f][t] = float(e)
        for k in range(n):
            for i in range(n):
                if L[i][k] == NEG_INF:
                    continue
                for j in range(n):
                    if L[k][j] == NEG_INF:
                        continue
                    v = L[i][k] + L[k][j]
                    if v > L[i][j]:
                        L[i][j] = v
        return L
