import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec


def load_graph_from_file(file_path):
    """
    Load the graph from an edge list file.
    Each line in the file should represent an edge in the format: node1 node2
    """
    with open(file_path, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    print(f"Graph Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def perform_random_walks(graph, walk_length=30, num_walks=10):
    """
    Perform random walks on the graph.
    """
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = random_walk(graph, start_node=node, walk_length=walk_length)
            walks.append(walk)
    return walks


def random_walk(graph, start_node, walk_length):
    """
    Simulate a random walk of fixed length starting from a given node.
    """
    walk = [start_node]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if len(neighbors) == 0:
            break
        walk.append(np.random.choice(neighbors))
    return walk

def train_deepwalk(walks, dimensions=64, window_size=5, workers=4):
    """
    Train a DeepWalk model using Word2Vec on the random walks.
    """
    print("Training DeepWalk embeddings...")
    walks = [[str(node) for node in walk] for walk in walks]
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers)
    embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key}
    return embeddings

def perform_kmeans_clustering(embeddings, num_clusters):
    """
    Cluster embeddings using K-Means.
    """
    print(f"Clustering into {num_clusters} communities using K-Means...")
    embedding_matrix = np.array([embeddings[node] for node in sorted(embeddings.keys())])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding_matrix)
    return labels, kmeans


def visualize_communities(graph, labels, title="DeepWalk Community Detection"):
    """
    Visualize graph communities with a spring layout.
    """
    print("Visualizing communities...")
    pos = nx.spring_layout(graph, seed=42)
    unique_labels = set(labels)
    colors = plt.cm.tab20(range(len(unique_labels)))
    node_colors = [colors[labels[i]] for i in range(len(labels))]

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, node_size=500, alpha=0.9, font_size=10)
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show()


def evaluate_clustering(embeddings, labels):
    """
    Compute the silhouette score of the clustering.
    """
    embedding_matrix = np.array([embeddings[node] for node in sorted(embeddings.keys())])
    silhouette_avg = silhouette_score(embedding_matrix, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    return silhouette_avg


def optimize_kmeans_clustering(graph, embeddings, max_no_improvement=3, threshold=0.01):
    """
    Find the optimal number of clusters by maximizing the silhouette score.
    """
    max_silhouette_score = -1
    prev_silhouette_score = -1
    no_improvement_count = 0

    best_community_count = 0
    best_labels = None
    best_kmeans = None

    for num_clusters in range(2, graph.number_of_nodes()):
        labels, kmeans = perform_kmeans_clustering(embeddings, num_clusters)
        current_silhouette_score = evaluate_clustering(embeddings, labels)

        if current_silhouette_score > max_silhouette_score:
            max_silhouette_score = current_silhouette_score
            best_labels = labels
            best_kmeans = kmeans
            best_community_count = num_clusters
            no_improvement_count = 0
        else:
            if abs(current_silhouette_score - prev_silhouette_score) < threshold:
                no_improvement_count += 1

        if no_improvement_count >= max_no_improvement:
            print(f"Stopping: No significant improvement after {max_no_improvement} iterations.")
            break

        prev_silhouette_score = current_silhouette_score

    return best_labels, best_kmeans, best_community_count

def main(file_path):
    graph = load_graph_from_file(file_path)
    walks = perform_random_walks(graph, walk_length=30, num_walks=20)
    embeddings = train_deepwalk(walks)

    best_labels, _ , best_community_count= optimize_kmeans_clustering(graph, embeddings)

    evaluate_clustering(embeddings, best_labels)
    
    print(f"Clustering into {best_community_count} communities")
    visualize_communities(graph, best_labels)

if __name__ == "__main__":
    file_path = "./sample-graph (50).txt"
    main(file_path)
