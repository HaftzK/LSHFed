import torch
import os

os.makedirs("random_vectors", exist_ok=True)

num_random_vectors = 5

def generate_and_save_random_vectors(vector_length, num_vectors=num_random_vectors):
    random_vectors = [torch.randn(vector_length) for _ in range(num_vectors)]
    file_path = f'random_vectors/random_vectors_length_{vector_length}.pth'
    print(random_vectors)
    torch.save(random_vectors, file_path)
    print(f"Rnd vectors saved to: '{file_path}'")

# example
generate_and_save_random_vectors(384)
