import torch
import torch.nn.functional as F
from tokenizer.tokenizer_image.vq_model import VQ_models
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

def mse(a, b):
    return torch.mean((a - b) ** 2, dim=-1)



def balanced_k_means_clustering(vectors, n, max_iter=100):
    num_vectors = vectors.size(0)
    m=num_vectors//n
    assert num_vectors == n * m
    indices = torch.randperm(num_vectors)[:n]
    centers = vectors[indices]

    for iteration in range(max_iter):
        distances = torch.cdist(vectors, centers, p=2) ** 2
        min_distances, min_indices = torch.min(distances, dim=1)
        sorted_indices = torch.argsort(min_distances)
        clusters = [[] for _ in range(n)]
        cluster_sizes = [0] * n
        assignments = torch.full((num_vectors,), -1, dtype=torch.long)

        for i in sorted_indices:
            for idx in torch.argsort(distances[i]):
                if cluster_sizes[idx] < m:
                    clusters[idx].append(vectors[i])
                    cluster_sizes[idx] += 1
                    assignments[i] = idx
                    break

        new_centers = []
        total_mse = 0.0
        for idx, cluster in enumerate(clusters):
            new_center = torch.mean(torch.stack(cluster), dim=0)
            new_centers.append(new_center)
            cluster_mse = mse(torch.stack(cluster), new_center).sum().item()
            total_mse += cluster_mse

        new_centers = torch.stack(new_centers)

        print(f"Iteration {iteration + 1}, Total Intra-cluster MSE: {total_mse}")
        if torch.allclose(centers, new_centers):
            break
        centers = new_centers

    return assignments



def inner_class_nearest_neighbor_sort(vectors, vectors_per_class):
    num_vectors = vectors.size(0)
    n = num_vectors//vectors_per_class
    assert num_vectors % n == 0, "The number of vectors must be divisible by n"

    final_order=torch.zeros(vectors.size(0)).int()
    for i in range(n):
        order = torch.zeros(vectors_per_class).int()
        visited = torch.zeros(vectors_per_class, dtype=torch.bool)

        start_idx = i * vectors_per_class
        end_idx = (i + 1) * vectors_per_class
        class_vectors = vectors[start_idx:end_idx]
        distance_matrix = torch.cdist(class_vectors, class_vectors, p=2) ** 2
        class_center = torch.mean(class_vectors, dim=0)

        center_distances = torch.norm(class_vectors - class_center, dim=1)

        closest_idx = torch.argmin(center_distances).int()
        order[vectors_per_class//2-1]=closest_idx
        visited[closest_idx] = True
        remaining_indices = torch.where(~visited)[0]
        distances_to_current = distance_matrix[closest_idx, remaining_indices]
        next_index = remaining_indices[torch.argmin(distances_to_current)].item()
        order[vectors_per_class//2] = next_index
        visited[next_index] = True

        for ii in range(0, vectors_per_class//2-1):
            current_index=order[vectors_per_class//2-1-ii]
            remaining_indices = torch.where(~visited)[0]
            distances_to_current = distance_matrix[current_index, remaining_indices]
            next_index = remaining_indices[torch.argmin(distances_to_current)].item()
            order[vectors_per_class//2-1-ii-1]=next_index
            visited[next_index] = True

            current_index = order[vectors_per_class//2 + ii]
            remaining_indices = torch.where(~visited)[0]
            distances_to_current = distance_matrix[current_index, remaining_indices]
            next_index = remaining_indices[torch.argmin(distances_to_current)].item()
            order[vectors_per_class//2+ii+1]=next_index
            visited[next_index] = True

        final_order[start_idx:end_idx] = order+i*vectors_per_class

    return final_order


def nearest_neighbor_sort(tensor, distance_matrix):
    n = distance_matrix.size(0)
    visited = torch.zeros(n, dtype=torch.bool)
    order = []

    current_index = 0
    order.append(current_index)
    visited[current_index] = True

    for ii in range(1, n):
        remaining_indices = torch.where(~visited)[0]
        distances_to_current = distance_matrix[current_index, remaining_indices]
        next_index = remaining_indices[torch.argmin(distances_to_current)].item()

        current_index = next_index
        order.append(current_index)
        visited[current_index] = True

    sorted_tensor = tensor[order]
    return sorted_tensor, order

def main():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models["VQ-16"](
        codebook_size=16384,
        codebook_embed_dim=8)
    vq_model.to(device)
    vq_model.eval()
    checkpoint0 = torch.load('./pretrained_models/vq_ds16_c2i.pt', map_location="cpu")
    vq_model.load_state_dict(checkpoint0["model"])
    embedding_normalized = F.normalize(vq_model.quantize.embedding.weight, p=2, dim=-1)
    vector_per_class = 128
    n = 16384 // vector_per_class
    assignments = balanced_k_means_clustering(embedding_normalized, n)
    embedding = vq_model.quantize.embedding.weight
    sorted_vectors = []
    for cluster_idx in range(n):
        for i in range(embedding.size(0)):
            if assignments[i] == cluster_idx:
                sorted_vectors.append(embedding[i])
    sorted_vectors = torch.stack(sorted_vectors)

    sorted_vectors_normalized = F.normalize(sorted_vectors, p=2, dim=-1)
    order=inner_class_nearest_neighbor_sort(sorted_vectors_normalized, vector_per_class)

    embedding1 = sorted_vectors[order]

    distances = torch.cdist(embedding, embedding1, p=2) ** 2
    index = torch.where(distances < 1e-6)
    checkpoint0["model"]['quantize.embedding.weight'] = embedding1
    torch.save(checkpoint0, f'./pretrained_models/vq_ds16_c2i-reorder-kmeans+nearest-cluster_size={vector_per_class}.pt')
    torch.save(index[1],f'./pretrained_models/mapping-kmeans+nearest-cluster_size={vector_per_class}.pt')


if __name__ == "__main__":
    main()
