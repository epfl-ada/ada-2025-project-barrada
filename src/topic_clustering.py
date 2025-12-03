import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Tuple, Union

class TopicClustering:
    """
    Runs K-Means clustering (K=40)
    """

    CLUSTER_LABELS_40 = {
        0: 'Spam',
        1: 'Games',
        2: 'NSFW-Fetish',
        3: 'Geographic/Countries',
        4: 'NSFW-Extreme',
        5: 'YouTube/Gaming Content',
        6: 'Science/Academia',
        7: 'Lifestyle/Outdoor',
        8: 'Niche',
        9: 'Personal/Community',
        10: 'Tech/Cryptocurrency',
        11: 'Small Random',
        12: 'Mix',
        13: 'Low Quality',
        14: 'YouTuber Personalities',
        15: 'Lifestyle',
        16: 'Memes/Cute/Funny',
        17: 'Movies/TV/Comics',
        18: 'Art/Imaginary',
        19: 'Celebrities/Models',
        20: 'Music',
        21: 'Random',
        22: 'Gaming-Mobile', 
        23: 'Spam/Personalities',
        24: 'Football',
        25: 'NSFW-Homosexual',
        26: 'Trading',
        27: 'Japanese',
        28: 'Shitpost',
        29: 'My Little Pony',
        30: 'Politics/Activism',
        31: 'Lifestyle/Home',
        32: 'Politics/extremism',
        33: 'Help/Advice/Q&A',
        34: 'Entertainment Mixed',
        35: 'Gaming-AAA Titles',
        36: 'Technology/Hardware',
        37: 'Meta-Commentary',
        38: 'Sports/American',
        39: 'NSFW',
    }

    def __init__(self, embeddings_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, print_samples: bool = True) -> Tuple[pd.DataFrame, float, float]:
        print("Loading raw embeddings...")
        df = pd.read_csv(self.embeddings_path, header=None)
        subreddit_names = df.iloc[:, 0].str.lower().str.strip().values
        embedding_matrix = df.iloc[:, 1:].values

        # Normalize vectors
        embedding_matrix_norm = normalize(embedding_matrix, norm='l2', axis=1)

        print(f"Running K-Means clustering (K=40) on {len(subreddit_names)} subreddits...")
        kmeans = KMeans(n_clusters=40, init='k-means++', n_init=10,
                        max_iter=300, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix_norm)
        centroids = kmeans.cluster_centers_

        # Quality metrics
        silhouette = silhouette_score(embedding_matrix_norm, cluster_labels,
                                      metric='cosine', sample_size=min(10000, len(subreddit_names)),
                                      random_state=42)
        davies_bouldin = davies_bouldin_score(embedding_matrix_norm, cluster_labels)
        print(f"  Clustering complete")
        print(f"  Silhouette (Cosine): {silhouette:.4f}")
        print(f"  Davies-Bouldin: {davies_bouldin:.4f}\n")

        # Create DataFrame
        topic_clusters_df = pd.DataFrame({
            'subreddit': subreddit_names,
            'topic_cluster': cluster_labels
        })

        # Map labels
        topic_clusters_df['topic_cluster_label'] = topic_clusters_df['topic_cluster'].map(self.CLUSTER_LABELS_40)
        topic_clusters_df['topic_cluster_label'] = topic_clusters_df.apply(
            lambda row: row['topic_cluster_label'] if pd.notna(row['topic_cluster_label'])
            else f"Cluster_{int(row['topic_cluster'])}", axis=1
        )

        # Save results
        output_path = self.output_dir / "embeddings_kmeans_40.csv"
        topic_clusters_df.to_csv(output_path, index=False)
        print(f" Topic clusters saved to: {output_path}")

        # Save the label map
        labelmap_path = self.output_dir / "cluster_labels_40.csv"
        pd.DataFrame(list(self.CLUSTER_LABELS_40.items()), columns=['topic_cluster', 'label']).to_csv(labelmap_path, index=False)
        print(f" Cluster label map saved to: {labelmap_path}\n")

        # print top 10 central + 10 random members
        if print_samples:
            print("Top 10 central + 10 random members per cluster:")
            for cluster_id in range(40):
                mask = cluster_labels == cluster_id
                cluster_names = subreddit_names[mask]
                cluster_embeddings = embedding_matrix_norm[mask]
                centroid = centroids[cluster_id]

                # Central members
                distances = np.array([1 - np.dot(cluster_embeddings[i], centroid) for i in range(len(cluster_embeddings))])
                central_idx = np.argsort(distances)[:10]

                # Random members
                random_idx = np.random.choice(len(cluster_names), size=min(10, len(cluster_names)), replace=False)

                print(f"\nCluster {cluster_id} ({self.CLUSTER_LABELS_40.get(cluster_id, cluster_id)}) - {len(cluster_names)} members")
                print("  Top 10 central:")
                for i, idx in enumerate(central_idx, 1):
                    print(f"    {i}. {cluster_names[idx]}")
                print("  10 Random:")
                for i, idx in enumerate(random_idx, 1):
                    print(f"    {i}. {cluster_names[idx]}")

        return topic_clusters_df, silhouette, davies_bouldin

def run_topic_clustering(embeddings_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        print_samples: bool = True
                        ) -> Tuple[pd.DataFrame, float, float]:
    
    cluster_runner = TopicClustering(embeddings_path, output_dir)
    return cluster_runner.run(print_samples=print_samples)

if __name__ == "__main__":
    pass