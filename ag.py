# ag.py
from typing import List, Dict, Any, Tuple
import time
import torch
from lsh import LSH

class Aggregator:
    def __init__(self, node_id: int, lt_ids: List[int]):
        self.node_id = node_id
        self.lt_ids = list(lt_ids)

    @torch.no_grad()
    def aggregate(self, lt_updates: List[torch.Tensor]) -> torch.Tensor:
        if len(lt_updates) == 0:
            raise ValueError("lt_updates is empty")
        agg = lt_updates[0].clone()
        for t in lt_updates[1:]:
            agg.add_(t)
        agg.div_(len(lt_updates))
        return agg

    def process_round(self, grouped_updates: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """
        input：{lt_id -> update_tensor}
        output：{
            'ag_id': int,
            'agg_update': torch.Tensor,     # aggregation result tensor
            'bitstring': List[List[int]],   # LSHGM bit string
            'elapsed': float                # time consumption
        }
        """
        lt_updates = [grouped_updates[i] for i in self.lt_ids if i in grouped_updates]
        t0 = time.time()
        agg_update = self.aggregate(lt_updates)
        bitstring = LSH("delete this, uncomment and modify the following to match your model")
        #bitstring = LSH(agg_update) # perform lsh on separate layers, not the whole model
        elapsed = time.time() - t0
        return {
            "ag_id": self.node_id,
            "agg_update": agg_update,
            "bitstring": bitstring,
            "elapsed": elapsed
        }
