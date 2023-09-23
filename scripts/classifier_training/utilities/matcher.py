from scipy.optimize import linear_sum_assignment

import torch
from torch import nn

#target_classes[idx] = target_classes_o

def _get_src_permutation_idx(self, indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

class HungarianMatcher(nn.Module):
    def __init__(self, cost=1.0):
        super().__init__()
        self.cost = cost

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs.shape[:2]

        out_prob = outputs.flatten(0, 1).softmax(-1)
        tgt_ids = targets.flatten()

        cost_class = -out_prob[:, tgt_ids]

        cost_matrix = self.cost * cost_class
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(num_queries, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
