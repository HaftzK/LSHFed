# vr.py
from typing import List, Dict, Any, Tuple, Iterable
import hashlib
import numpy as np
import torch
from lsh import LSH, LSH_cmp

def _quantile_from_ranks(values: List[float], ascending: bool = True) -> List[float]:
    R = len(values)
    idx = np.argsort(values) if ascending else np.argsort(-np.array(values))
    ranks = np.empty(R, dtype=int)
    ranks[idx] = np.arange(R)  # 0=best
    if R == 1:
        return [1.0]
    return [(R - r) / (R - 1) for r in ranks.tolist()]

def _sha_ptr(round_id: int, seed: str) -> float:
    s = f"{round_id}_{seed}".encode()
    hv = int(hashlib.sha256(s).hexdigest(), 16)
    return (hv % 10**12) / 10**12

def _ring_select(candidates: List[int], weights: List[float], k: int, start_ptr: float) -> List[int]:
    if len(candidates) == 0 or k <= 0:
        return []
    total = sum(weights)
    if total <= 0:
        cycle = list(candidates)
        return [cycle[i % len(cycle)] for i in range(k)]

    arc = [w / total for w in weights]
    prefix = np.cumsum(arc)  # in (0,1]
    selected = []
    ptr = start_ptr % 1.0

    j = 0
    while len(selected) < k and j < len(candidates) * 2:
        idx = np.searchsorted(prefix, ptr, side="right")
        if idx >= len(candidates):
            ptr = 0.0
            continue
        sel = candidates[idx]
        if sel not in selected:
            selected.append(sel)
        ptr = (prefix[idx] + 1e-12) % 1.0
        j += 1

    i = 0
    while len(selected) < k and i < len(candidates):
        if candidates[i] not in selected:
            selected.append(candidates[i])
        i += 1
    return selected[:k]

class Verifier:
    def __init__(self,
                 all_node_ids: List[int],
                 dn_ids: List[int],
                 nondn_ids: List[int],
                 alpha1: float = 0,
                 alpha2: float = 1.0,  # For best robustness in exp, use alpha1=0 alpha2=1
                 seed: str = "42"):
        assert abs(alpha1 + alpha2 - 1.0) < 1e-8, "alpha1 + alpha2 must equals 1"
        self.all_node_ids = list(all_node_ids)
        self.dn_ids = list(dn_ids)
        self.nondn_ids = list(nondn_ids)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.seed = seed

        self.prev_benchmark_bitstring = None  # B^{t-1}
        self.last_scores: Dict[int, float] = {i: 1.0 for i in all_node_ids}

    def select_best_ag(self, ag_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        inpur：
          - ag_id: int
          - agg_update: torch.Tensor
          - bitstring: List[List[int]]
          - elapsed: float
        """
        if len(ag_reports) == 0:
            raise ValueError("ag_reports 为空")

        winner = None
        if self.prev_benchmark_bitstring is None:
            # in round 0 prev_benchmark_bitstring is obtained through local training by VR
            winner = min(ag_reports, key=lambda r: r["elapsed"])
            self.prev_benchmark_bitstring = winner["bitstring"]
            return winner

        best = None
        best_dist = None
        for rep in ag_reports:
            dist = LSH_cmp(rep["bitstring"], self.prev_benchmark_bitstring)
            if best_dist is None or dist < best_dist:
                best = rep
                best_dist = dist

        self.prev_benchmark_bitstring = best["bitstring"]
        return best

    def compute_reputation(self,
                           time_by_node: Dict[int, float],
                           dist_by_node: Dict[int, float]) -> Dict[int, float]:
        """
        S(i) = α1 * φ_Q^{(1)}(i) + α2 * φ_Q^{(2)}(i)
        - φ_Q^{(1)}：time
        - φ_Q^{(2)}：LSHGM
        """
        nodes = list(self.all_node_ids)
        times = [time_by_node.get(i, float("inf")) for i in nodes]
        dists = [dist_by_node.get(i, float("inf")) for i in nodes]

        q_time = _quantile_from_ranks(times, ascending=True)
        q_dist = _quantile_from_ranks(dists, ascending=True)

        scores = {nodes[i]: self.alpha1 * q_time[i] + self.alpha2 * q_dist[i] for i in range(len(nodes))}
        self.last_scores = scores
        return scores

    def elect_roles(self,
                    round_id: int,
                    Q: int,
                    P: int) -> Tuple[List[int], List[int]]:

        dn_scores = [max(self.last_scores.get(i, 0.0), 0.0) for i in self.dn_ids]
        non_scores = [max(self.last_scores.get(i, 0.0), 0.0) for i in self.nondn_ids]
        ptr = _sha_ptr(round_id, self.seed)

        lt_selected = _ring_select(self.dn_ids, dn_scores, max(Q, 0), ptr)
        ag_selected = _ring_select(self.nondn_ids, non_scores, max(P, 0), ptr)
        return lt_selected, ag_selected

    def group_lts_to_ags(self, lt_ids: List[int], ag_ids: List[int]) -> Dict[int, List[int]]:
        if len(ag_ids) == 0:
            return {}
        groups = {aid: [] for aid in ag_ids}
        for j, lt in enumerate(lt_ids):
            groups[ag_ids[j % len(ag_ids)]].append(lt)
        return groups

    def run_round(self,
                  round_id: int,
                  ag_reports: List[Dict[str, Any]],
                  time_by_node: Dict[int, float],
                  dist_by_node: Dict[int, float],
                  Q: int,
                  P: int) -> Dict[str, Any]:

        best = self.select_best_ag(ag_reports)
        winner_ag = best["ag_id"]
        global_update = best["agg_update"]

        _ = self.compute_reputation(time_by_node, dist_by_node)

        lt_selected, ag_selected = self.elect_roles(round_id, Q=Q, P=P)

        groups = self.group_lts_to_ags(lt_selected, ag_selected)

        return {
            "winner_ag": winner_ag,
            "global_update": global_update,
            "lt_selected": lt_selected,
            "ag_selected": ag_selected,
            "groups": groups
        }
