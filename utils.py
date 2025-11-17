import torch

def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.view(-1, 1)
    elif t.dim() == 2:
        return t
    else:
        return t.view(t.size(0), -1)

def _prep(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu().float()
    return t.reshape(-1, t.shape[-1])

def euclidean_distance_between_matrices(matrix1: torch.Tensor, matrix2: torch.Tensor) -> float:
    a, b = _prep(matrix1), _prep(matrix2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Two gradient tensors must have the same size in the last dimension (feature dimension) in order to calculate the Euclidean distance.")
    m, n = a.shape[0], b.shape[0]
    sum_sq_a = torch.sum(a.pow(2))
    sum_sq_b = torch.sum(b.pow(2))
    sum_a = torch.sum(a, dim=0)
    sum_b = torch.sum(b, dim=0)
    total_sq = n * sum_sq_a + m * sum_sq_b - 2 * torch.dot(sum_a, sum_b)
    total_sq = torch.clamp(total_sq, min=0)
    return float(torch.sqrt(total_sq))

def _rowwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    if a.shape != b.shape:
        raise ValueError("Two gradient tensors must have the same shape after expansion.")
    a = torch.nn.functional.normalize(a, p=2, dim=1, eps=1e-8)
    b = torch.nn.functional.normalize(b, p=2, dim=1, eps=1e-8)
    return (a * b).sum(dim=1)
