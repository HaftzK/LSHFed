import torch
import hashlib
import numpy as np
from lsh import func_LSH

class LSHFed:
    def __init__(self, num_nodes, alpha1=0.5, alpha2=0.5, group_size=3, seed="42"):
        self.num_nodes = num_nodes
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.group_size = group_size
        self.seed = seed
        self.reputation_scores = [1.0 for _ in range(num_nodes)]
        self.history_distances = [0.0 for _ in range(num_nodes)]
        self.history_times = [1.0 for _ in range(num_nodes)]
        self.prev_gradient = None  # Benchmark gradient LSHGM bit string

    def update_metrics(self, gradient_list, reference_gradient, time_list):
        self.history_distances = []
        self.history_times = []

        for i, grad in enumerate(gradient_list):
            d_lsh = func_LSH(reference_gradient, grad)
            self.history_distances.append(d_lsh)
            self.history_times.append(time_list[i])

    def _rank_to_quantile(self, scores):
        ranks = np.argsort(np.argsort(scores))
        quantile_scores = [(self.num_nodes - r) / (self.num_nodes - 1) for r in ranks]
        return quantile_scores

    def compute_reputation_scores(self):
        q1 = self._rank_to_quantile(self.history_times)
        q2 = self._rank_to_quantile(self.history_distances)
        self.reputation_scores = [
            self.alpha1 * q1[i] + self.alpha2 * q2[i]
            for i in range(self.num_nodes)
        ]

    def hash_ring_select(self, round_id, role_type="LT", count=3):
        total_score = sum(self.reputation_scores)
        arc_lengths = [s / total_score for s in self.reputation_scores]

        hash_input = f"{round_id}_{self.seed}".encode()
        hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
        pointer = (hash_val % 10**8) / 10**8

        selected = []
        curr = 0.0
        for i in range(self.num_nodes):
            curr += arc_lengths[i]
            if curr > pointer:
                selected.append(i)
                if len(selected) == count:
                    break

        i = 0
        while len(selected) < count:
            selected.append(i)
            i += 1
        return selected

    def aggregate_gradients(self, selected_nodes, gradient_list):
        agg_grad = gradient_list[selected_nodes[0]].clone()
        for idx in selected_nodes[1:]:
            agg_grad += gradient_list[idx]
        return agg_grad / len(selected_nodes)

    def step(self, round_id, gradient_list, time_list):
        if self.prev_gradient is None:
            self.prev_gradient = gradient_list[-1]

        self.update_metrics(gradient_list, self.prev_gradient, time_list)
        self.compute_reputation_scores()
        selected_nodes = self.hash_ring_select(round_id, count=self.group_size)
        aggregated = self.aggregate_gradients(selected_nodes, gradient_list)
        self.prev_gradient = aggregated
        return aggregated, selected_nodes
