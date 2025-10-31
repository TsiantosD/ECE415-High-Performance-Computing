import random

def generate_kmeans_input(filename, num_points=100000, num_features=16):
    with open(filename, 'w') as f:
        for i in range(1, num_points + 1):
            features = [f"{random.uniform(-5, 5):.6f}" for _ in range(num_features)]
            f.write(f"{i} {' '.join(features)}\n")

# Example: 100k points, 16 dimensions
generate_kmeans_input("data_large.txt", num_points=100000, num_features=16)
