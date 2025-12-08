import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Tuple, Union

class TopicClustering:
    """
    Runs K-Means clustering (K=40) with definitive manual semantic overrides.
    "Final Boss" version: Balances academic critique vs. toxic mockery.
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
        22: 'Esports/Online Games', # Renamed from Gaming-Mobile
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

    MANUAL_FIXES = {
        # --- 1. ACADEMIC & INTELLECTUAL ---
        'badhistory': 'Science/Academia',
        'badscience': 'Science/Academia',
        'badlinguistics': 'Science/Academia',
        'badeconomics': 'Science/Academia',
        'badphilosophy': 'Science/Academia',
        'askhistorians': 'Science/Academia',
        
        # --- 2. POLITICS ---
        'ukpolitics': 'Politics/Activism',
        'incels': 'Politics/extremism',
        'sjwnews': 'Politics/extremism',
        'enough_sanders_spam': 'Politics/Activism',
        'publichealthwatch': 'Politics/extremism',
        
        # --- 3. GEOGRAPHY RESCUES  ---
        'unitedkingdom': 'Geographic/Countries',
        'argentina': 'Geographic/Countries',
        'vzla': 'Geographic/Countries',
        'norwayonreddit': 'Geographic/Countries',
        'brdapublic': 'Geographic/Countries',
        'seattleshitshow': 'Geographic/Countries',
        
        # --- 4. HOBBY RESCUES ---
        'dota2': 'Esports/Online Games',
        'kappa': 'Esports/Online Games',
        'anime': 'Movies/TV/Comics',
        'squaredcircle': 'Sports/American',
        'snowboardcj': 'Sports/American',
        'kanye': 'Music',
        'hiphopheads': 'Music',
        'hiphopcirclejerk': 'Music', 
        'streetwear': 'Lifestyle',
        'civcringe': 'Games',
        'galacticracing': 'Games',
        'findagame': 'Games',
        
        # --- 5. META-COMMENTARY ) ---
        'bettersubredditdrama': 'Meta-Commentary',
        'shitamericanssay': 'Meta-Commentary',
        'shitpoliticssays': 'Meta-Commentary',
        'fitnesscirclejerk': 'Meta-Commentary',
        'moviescirclejerk': 'Meta-Commentary',
        'gamingcirclejerk': 'Meta-Commentary',
        'undelete': 'Meta-Commentary',
        'hatesubsinaction': 'Meta-Commentary',
        'botrights': 'Meta-Commentary',
        'botsscrewingup': 'Meta-Commentary',
        'dutchshitonreddit': 'Meta-Commentary',
        'atethepasta': 'Meta-Commentary',
        'foundtheprogrammer': 'Meta-Commentary',
        'thathappend': 'Meta-Commentary',
        'whoreddithatesnow': 'Meta-Commentary',
        'metafitnesscirclejerk': 'Meta-Commentary',
        'femrameta': 'Meta-Commentary',
        'gamerghazi': 'Meta-Commentary',
        
        # --- 6. MISC CLEAN UP ---
        'dailydot': 'Lifestyle',
        'news': 'Lifestyle',
        'worldnews': 'Lifestyle',
        'technology': 'Technology/Hardware',
        'skincareexchange': 'Lifestyle',
        'smalldickproblems': 'Personal/Community',
        'legaladviceofftopic': 'Help/Advice/Q&A',
        'motorcyclescirclejerk': 'Lifestyle/Outdoor',
        'dogecoinscamwatch': 'Tech/Cryptocurrency',
        'testingground4bots': 'Technology/Hardware',
        'nosleepworkshops': 'Art/Imaginary',
        '9m9h9e9': 'Art/Imaginary',
        'gonewildaudio': 'NSFW-Fetish',
        'rekt': 'Memes/Cute/Funny',
        'hapas': 'Personal/Community',
        
        # --- 7. BOTS ---
        'locationbot': 'Spam',
        'totesmessenger': 'Spam',
        'autowikibot': 'Spam',
        'tweetposter': 'Spam'
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

        embedding_matrix_norm = normalize(embedding_matrix, norm='l2', axis=1)

        print(f"Running K-Means clustering (K=40) on {len(subreddit_names)} subreddits...")
        kmeans = KMeans(n_clusters=40, init='k-means++', n_init=10,
                        max_iter=300, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix_norm)
        centroids = kmeans.cluster_centers_

        silhouette = silhouette_score(embedding_matrix_norm, cluster_labels,
                                      metric='cosine', sample_size=min(10000, len(subreddit_names)),
                                      random_state=42)
        davies_bouldin = davies_bouldin_score(embedding_matrix_norm, cluster_labels)
        print(f"  Clustering complete")
        print(f"  Silhouette (Cosine): {silhouette:.4f}")
        print(f"  Davies-Bouldin: {davies_bouldin:.4f}\n")

        topic_clusters_df = pd.DataFrame({
            'subreddit': subreddit_names,
            'topic_cluster': cluster_labels
        })

        topic_clusters_df['topic_cluster_label'] = topic_clusters_df['topic_cluster'].map(self.CLUSTER_LABELS_40)
        
        topic_clusters_df['topic_cluster_label'] = topic_clusters_df.apply(
            lambda row: row['topic_cluster_label'] if pd.notna(row['topic_cluster_label'])
            else f"Cluster_{int(row['topic_cluster'])}", axis=1
        )

        print(f"Applying {len(self.MANUAL_FIXES)} hardcoded overrides...")
        fixed_count = 0
        for sub, new_label in self.MANUAL_FIXES.items():
            if sub in topic_clusters_df['subreddit'].values:
                topic_clusters_df.loc[topic_clusters_df['subreddit'] == sub, 'topic_cluster_label'] = new_label
                fixed_count += 1
        
        print(f"  Total subreddits re-classified: {fixed_count}\n")

        output_path = self.output_dir / "embeddings_kmeans_40.csv"
        topic_clusters_df.to_csv(output_path, index=False)
        print(f" Topic clusters saved to: {output_path}")

        labelmap_path = self.output_dir / "cluster_labels_40.csv"
        pd.DataFrame(list(self.CLUSTER_LABELS_40.items()), columns=['topic_cluster', 'label']).to_csv(labelmap_path, index=False)
        print(f" Cluster label map saved to: {labelmap_path}\n")

        if print_samples:
            self._print_cluster_samples(topic_clusters_df)

        return topic_clusters_df, silhouette, davies_bouldin

    def _print_cluster_samples(self, df):
        print("Top 10 central + 10 random members per cluster (Post-Fix validation):")
        
        unique_labels = sorted(df['topic_cluster_label'].unique())
        
        for label in unique_labels:
            cluster_subs = df[df['topic_cluster_label'] == label]['subreddit'].values
            
            if len(cluster_subs) == 0:
                continue
                
            random_idx = np.random.choice(len(cluster_subs), size=min(10, len(cluster_subs)), replace=False)
            
            print(f"\nCluster: {label} - {len(cluster_subs)} members")
            print("  10 Random samples:")
            for i, idx in enumerate(random_idx, 1):
                print(f"    {i}. {cluster_subs[idx]}")

def run_topic_clustering(embeddings_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        print_samples: bool = True
                        ) -> Tuple[pd.DataFrame, float, float]:
    
    cluster_runner = TopicClustering(embeddings_path, output_dir)
    return cluster_runner.run(print_samples=print_samples)

if __name__ == "__main__":
    pass