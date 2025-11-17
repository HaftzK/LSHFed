import torch
import os
from utils import _flatten_to_2d

#LSHGM Core

def load_random_vectors(vector_length):
    file_path = f'random_vectors/random_vectors_length_{vector_length}.pth'
    if os.path.exists(file_path):
        random_vectors = torch.load(file_path)
        return random_vectors
    else:
        raise ValueError(f"Vector of len:{vector_length} not found. Generate and save them first using rndvector.py.")

def LSH(gradient_history: torch.Tensor):
    grad_2d = _flatten_to_2d(gradient_history.detach().to('cpu'))
    random_vectors = load_random_vectors(grad_2d.size(0))
    hashes = []
    for rand_vec in random_vectors:  # rand_vec: (rows,)
        dots = torch.mv(grad_2d.t(), rand_vec)
        hashes.append((dots > 0).int().tolist())
    return hashes

def LSH_cmp(list1, list2):
    tensor1 = torch.tensor(list1)
    tensor2 = torch.tensor(list2)
    if tensor1.shape != tensor2.shape:
        raise ValueError("LSH cannot compare two tensors with different shapes")
    differences = torch.ne(tensor1.flatten(), tensor2.flatten())
    num_differences = torch.sum(differences).item()
    return num_differences

def func_LSH(gradient_history1: torch.Tensor, gradient_history2: torch.Tensor):
    '''
    LSHGM bit string compare func.

    To use the LSHGM module standalone, call `func_LSH(your_gradient1, your_gradient2)` to
    obtain the LSHGM distance between the two tensors;
    feed the tensors layer by layer to ensure numerical precision.
    '''
    hashes1 = LSH(gradient_history1)
    hashes2 = LSH(gradient_history2)
    return LSH_cmp(hashes1, hashes2)
