import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class ClusterAggregator:
    """
    Aggregate subreddit-level data to cluster-level master dataset.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.df_final = None
        self.df_links = None
        
    def load_data(self) -> None:
        """Load required datasets."""
        print("Loading datasets...")
        
        self.df_final = pd.read_csv(self.data_dir / "final_dataset.csv")
        print(f"  Loaded final_dataset.csv: {len(self.df_final)} subreddits, {len(self.df_final.columns)} columns")
        
        self.df_links = pd.read_csv(self.data_dir / "combined_hyperlinks.csv")
        print(f"  Loaded combined_hyperlinks.csv: {len(self.df_links)} links")
    
    def get_cluster_subreddits(self, cluster_label: str) -> pd.DataFrame:
        """Get all subreddits in a cluster."""
        return self.df_final[self.df_final['topic_cluster_label'] == cluster_label]
    
    def extract_exemplars(self, cluster_label: str, metric: str = 'pagerank', top_n: int = 5) -> str:
        """
        Extract top N subreddits from a cluster by a given metric.
        
        Args:
            cluster_label: Label of the cluster
            metric: Metric to rank by (pagerank, betweenness, in_degree, etc.)
            top_n: Number of exemplars to return
            
        Returns:
            Comma-separated string of subreddit names
        """
        cluster_subs = self.get_cluster_subreddits(cluster_label)
        
        if metric not in cluster_subs.columns or len(cluster_subs) == 0:
            return ""
        
        cluster_subs_clean = cluster_subs.dropna(subset=[metric])
        if len(cluster_subs_clean) == 0:
            return ""
        
        top_subs = cluster_subs_clean.nlargest(top_n, metric)['subreddit'].tolist()
        return ", ".join(top_subs)
    
    def calculate_link_metrics(self, cluster_label: str) -> Dict:
        """
        Calculate link-based metrics for a cluster.
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Dictionary with link metrics
        """
        cluster_subs = self.get_cluster_subreddits(cluster_label)['subreddit'].tolist()
        
        if not cluster_subs:
            return {
                'total_links_in': 0,
                'total_links_out': 0,
                'total_links': 0,
                'avg_in_sentiment': 0,
                'avg_out_sentiment': 0,
                'pct_positive_in': 0,
                'pct_positive_out': 0
            }
        
        links_in = self.df_links[self.df_links['TARGET_SUBREDDIT'].isin(cluster_subs)]
        links_out = self.df_links[self.df_links['SOURCE_SUBREDDIT'].isin(cluster_subs)]
        
        total_in = len(links_in)
        total_out = len(links_out)
        
        avg_in_sent = links_in['LINK_SENTIMENT'].mean() if total_in > 0 else 0
        avg_out_sent = links_out['LINK_SENTIMENT'].mean() if total_out > 0 else 0
        
        pct_pos_in = (links_in['LINK_SENTIMENT'] == 1).sum() / total_in if total_in > 0 else 0
        pct_pos_out = (links_out['LINK_SENTIMENT'] == 1).sum() / total_out if total_out > 0 else 0
        
        return {
            'total_links_in': total_in,
            'total_links_out': total_out,
            'total_links': total_in + total_out,
            'avg_in_sentiment': avg_in_sent,
            'avg_out_sentiment': avg_out_sent,
            'pct_positive_in': pct_pos_in,
            'pct_positive_out': pct_pos_out
        }
    
    def calculate_insularity(self, cluster_label: str) -> float:
        """
        Calculate cluster insularity (internal links / total links).
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Insularity score between 0 and 1
        """
        cluster_subs = self.get_cluster_subreddits(cluster_label)['subreddit'].tolist()
        
        if not cluster_subs:
            return 0.0
        
        internal_links = self.df_links[
            self.df_links['SOURCE_SUBREDDIT'].isin(cluster_subs) &
            self.df_links['TARGET_SUBREDDIT'].isin(cluster_subs)
        ]
        
        all_links = self.df_links[
            self.df_links['SOURCE_SUBREDDIT'].isin(cluster_subs) |
            self.df_links['TARGET_SUBREDDIT'].isin(cluster_subs)
        ]
        
        if len(all_links) == 0:
            return 0.0
        
        return len(internal_links) / len(all_links)
    
    def aggregate_liwc_scores(self, cluster_label: str) -> Dict:
        """
        Aggregate all LIWC scores for a cluster.
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Dictionary with mean LIWC scores
        """
        cluster_data = self.get_cluster_subreddits(cluster_label)
        
        liwc_cols = [col for col in cluster_data.columns if col.startswith('LIWC_') and '_mean' in col]
        
        liwc_means = {}
        for col in liwc_cols:
            liwc_means[col] = cluster_data[col].mean()
        
        return liwc_means
    
    def aggregate_network_metrics(self, cluster_label: str) -> Dict:
        """
        Aggregate network metrics for a cluster.
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Dictionary with network metric statistics
        """
        cluster_data = self.get_cluster_subreddits(cluster_label)
        
        network_cols = ['pagerank', 'betweenness', 'hub_score', 'authority_score',
                       'in_degree', 'out_degree', 'clustering']
        
        metrics = {}
        for col in network_cols:
            if col in cluster_data.columns:
                metrics[f'{col}_mean'] = cluster_data[col].mean()
                metrics[f'{col}_max'] = cluster_data[col].max()
                metrics[f'{col}_std'] = cluster_data[col].std()
            else:
                metrics[f'{col}_mean'] = 0
                metrics[f'{col}_max'] = 0
                metrics[f'{col}_std'] = 0
        
        return metrics
    
    def calculate_role_distribution(self, cluster_label: str) -> Dict:
        """
        Calculate percentage of subreddits with each role in cluster.
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Dictionary with role percentages
        """
        cluster_data = self.get_cluster_subreddits(cluster_label)
        
        n_total = len(cluster_data)
        if n_total == 0:
            return {
                'pct_role_influential': 0,
                'pct_role_supportive': 0,
                'pct_role_critical': 0,
                'pct_role_controversial': 0
            }
        
        role_cols = ['role_influential', 'role_supportive', 'role_critical', 'role_controversial']
        role_pcts = {}
        
        for role_col in role_cols:
            if role_col in cluster_data.columns:
                pct = cluster_data[role_col].sum() / n_total
                role_pcts[f'pct_{role_col}'] = pct
            else:
                role_pcts[f'pct_{role_col}'] = 0
        
        return role_pcts
    
    def calculate_community_overlap(self, cluster_label: str) -> Tuple[int, int]:
        """
        Calculate how many network communities overlap with this cluster.
        
        Args:
            cluster_label: Label of the cluster
            
        Returns:
            Tuple of (n_communities_represented, dominant_community_id)
        """
        cluster_data = self.get_cluster_subreddits(cluster_label)
        
        if 'community' not in cluster_data.columns or len(cluster_data) == 0:
            return 0, -1
        
        communities = cluster_data['community'].dropna()
        n_communities = communities.nunique()
        dominant_community = int(communities.mode()[0]) if len(communities) > 0 else -1
        
        return n_communities, dominant_community
    
    def calculate_derived_metrics(self, liwc_scores: Dict) -> Dict:
        """
        Calculate derived psychological metrics.
        
        Args:
            liwc_scores: Dictionary of LIWC scores
            
        Returns:
            Dictionary with derived metrics
        """
        toxicity = (
            liwc_scores.get('LIWC_Anger_mean', 0) +
            liwc_scores.get('LIWC_Swear_mean', 0) -
            liwc_scores.get('LIWC_Posemo_mean', 0)
        )
        
        analytical = np.mean([
            liwc_scores.get('LIWC_Insight_mean', 0),
            liwc_scores.get('LIWC_Certain_mean', 0),
            liwc_scores.get('LIWC_CogMech_mean', 0)
        ])
        
        emotional = np.mean([
            liwc_scores.get('LIWC_Anger_mean', 0),
            liwc_scores.get('LIWC_Anx_mean', 0),
            liwc_scores.get('LIWC_Sad_mean', 0)
        ])
        
        return {
            'toxicity_score': toxicity,
            'analytical_score': analytical,
            'emotional_score': emotional
        }
    
    def aggregate_to_clusters(self) -> pd.DataFrame:
        """
        Main aggregation function to create cluster master dataset.
        
        Returns:
            DataFrame with one row per cluster and all aggregated metrics
        """
        print("\nAggregating to cluster level...")
        
        cluster_records = []
        
        unique_labels = sorted(self.df_final['topic_cluster_label'].dropna().unique())
        
        for cluster_label in unique_labels:
            cluster_ids = self.df_final[
                self.df_final['topic_cluster_label'] == cluster_label
            ]['topic_cluster'].value_counts()
            
            cluster_id = int(cluster_ids.index[0]) if len(cluster_ids) > 0 else -1
            
            cluster_subs = self.get_cluster_subreddits(cluster_label)
            n_subreddits = len(cluster_subs)
            
            if n_subreddits == 0:
                print(f"    Warning: No subreddits found for {cluster_label}")
                continue
            
            print(f"  Processing: {cluster_label} (n={n_subreddits})")
            
            record = {
                'cluster_id': cluster_id,
                'cluster_label': cluster_label,
                'n_subreddits': n_subreddits
            }

            
            link_metrics = self.calculate_link_metrics(cluster_label)
            record.update(link_metrics)
            
            record['sentiment_asymmetry'] = (
                link_metrics['avg_in_sentiment'] - link_metrics['avg_out_sentiment']
            )
            
            record['insularity'] = self.calculate_insularity(cluster_label)
            
            record['top5_by_pagerank'] = self.extract_exemplars(cluster_label, 'pagerank', 5)
            record['top5_by_betweenness'] = self.extract_exemplars(cluster_label, 'betweenness', 5)
            record['top5_by_indegree'] = self.extract_exemplars(cluster_label, 'in_degree', 5)
            
            liwc_scores = self.aggregate_liwc_scores(cluster_label)
            record.update(liwc_scores)
            
            network_metrics = self.aggregate_network_metrics(cluster_label)
            record.update(network_metrics)
            
            derived_metrics = self.calculate_derived_metrics(liwc_scores)
            record.update(derived_metrics)
            
            role_dist = self.calculate_role_distribution(cluster_label)
            record.update(role_dist)
            
            n_communities, dominant_comm = self.calculate_community_overlap(cluster_label)
            record['n_communities_represented'] = n_communities
            record['dominant_community'] = dominant_comm
            
            cluster_records.append(record)
        
        df_cluster_master = pd.DataFrame(cluster_records)
        
        print(f"\nCluster master dataset created: {len(df_cluster_master)} clusters")
        print(f"Total columns: {len(df_cluster_master.columns)}")
        
        return df_cluster_master
    
    def save_cluster_master(self, df_cluster_master: pd.DataFrame) -> Path:
        """
        Save cluster master dataset to CSV.
        
        Args:
            df_cluster_master: Cluster master DataFrame
            
        Returns:
            Path to saved file
        """
        output_path = self.data_dir / "cluster_master_dataset.csv"
        df_cluster_master.to_csv(output_path, index=False)
        print(f"\nCluster master dataset saved to: {output_path}")
        return output_path
    
    def run(self) -> pd.DataFrame:
        """
        Run full aggregation pipeline.
        
        Returns:
            Cluster master DataFrame
        """
        self.load_data()
        df_cluster_master = self.aggregate_to_clusters()
        self.save_cluster_master(df_cluster_master)
        return df_cluster_master


def create_cluster_master_dataset(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Main function to create cluster master dataset.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Cluster master DataFrame
    """
    aggregator = ClusterAggregator(data_dir)
    return aggregator.run()


if __name__ == "__main__":
    df_cluster_master = create_cluster_master_dataset()
    print("\nFirst 3 clusters:")
    print(df_cluster_master.head(3))