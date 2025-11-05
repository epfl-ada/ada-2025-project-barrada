import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import community as community_louvain

class NetworkAnalyzer:
    """Analyze the structure of Reddit's inter-community network"""
    
    def __init__(self, hyperlinks_df, output_dir="data/processed"):
        self.df = hyperlinks_df
        ROOT = Path(__file__).resolve().parent.parent
        self.output_dir = Path(ROOT / output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.G = None 
        self.G_pos = None 
        self.G_neg = None 
        self.G_undirected = None 
        
    def build_networks(self):
        """Construct network graphs"""        
        print("\n Building main directed network...")
        
        # Main network - aggregate multiple links between same pair
        self.G = nx.DiGraph()
        
        # Group by source-target pairs
        edge_data = self.df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).agg({
            'POST_ID': 'count',
            'sentiment_numeric': ['mean'],
            'is_positive': 'sum',
            'is_negative': 'sum'
        }).reset_index() 
        
        edge_data.columns = ['source', 'target', 'n_links', 'avg_sentiment', 
                            'n_positive', 'n_negative']
        
        # Add edges with attributes
        for _, row in edge_data.iterrows():
            self.G.add_edge(
                row['source'],
                row['target'],
                weight=row['n_links'],
                avg_sentiment=row['avg_sentiment'],
                n_positive=row['n_positive'],
                n_negative=row['n_negative']
            )
        
        print(f" Nodes: {self.G.number_of_nodes():>10,}")
        print(f" Edges: {self.G.number_of_edges():>10,}")
        
        # Positive-only network
        print(f"\n Building positive-only network...")
        pos_edges = self.df[self.df['is_positive'] == 1].groupby(
            ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']
        ).size().reset_index(name='weight')
        
        self.G_pos = nx.DiGraph()
        for _, row in pos_edges.iterrows():
            self.G_pos.add_edge(row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT'], 
                               weight=row['weight'])
        
        print(f"Nodes: {self.G_pos.number_of_nodes():>10,}")
        print(f"Edges: {self.G_pos.number_of_edges():>10,}")
        
        # Negative-only network
        print(f"\n Building negative-only network...")
        neg_edges = self.df[self.df['is_negative'] == 1].groupby(
            ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']
        ).size().reset_index(name='weight')
        
        self.G_neg = nx.DiGraph()
        for _, row in neg_edges.iterrows():
            self.G_neg.add_edge(row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT'],
                               weight=row['weight'])
        
        print(f"Nodes: {self.G_neg.number_of_nodes():>10,}")
        print(f"Edges: {self.G_neg.number_of_edges():>10,}")
        
        # Undirected version for community detection
        print(f"Building undirected network for community detection...")
        self.G_undirected = self.G.to_undirected()
        
        return self.G
    
    def compute_basic_stats(self):
        """Compute basic network statistics"""
        print("BASIC NETWORK STATISTICS")
        
        stats = {}
        
        # Size
        stats['n_nodes'] = self.G.number_of_nodes()
        stats['n_edges'] = self.G.number_of_edges()
        stats['density'] = nx.density(self.G)
        
        # Connectivity
        stats['n_weakly_connected_components'] = nx.number_weakly_connected_components(self.G)
        stats['n_strongly_connected_components'] = nx.number_strongly_connected_components(self.G)
        
        # Get largest component
        largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
        stats['largest_component_size'] = len(largest_wcc)
        stats['largest_component_fraction'] = len(largest_wcc) / stats['n_nodes']
        
        # Reciprocity
        stats['reciprocity'] = nx.reciprocity(self.G)
        
        print(f"\n Network Overview:")
        print(f"    Nodes:              {stats['n_nodes']:>10,}")
        print(f"    Edges:              {stats['n_edges']:>10,}")
        print(f"    Density:            {stats['density']:>10.6f}")
        print(f"    Reciprocity:        {stats['reciprocity']:>10.3f}")
        print(f"\n  Weakly connected:   {stats['n_weakly_connected_components']:>10,}")
        print(f"    Strongly connected: {stats['n_strongly_connected_components']:>10,}")
        print(f"    Largest component:  {stats['largest_component_size']:>10,} "
              f"({stats['largest_component_fraction']:>6.1%})")
        
        # Degree distributions
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        
        print(f"\n Degree Statistics:")
        print(f"    In-degree  - Mean: {np.mean(in_degrees):>8.2f}, "
              f"Median: {np.median(in_degrees):>6.0f}, "
              f"Max: {np.max(in_degrees):>6.0f}")
        print(f"    Out-degree - Mean: {np.mean(out_degrees):>8.2f}, "
              f"Median: {np.median(out_degrees):>6.0f}, "
              f"Max: {np.max(out_degrees):>6.0f}")
        
        return stats
    
    def compute_node_metrics(self):
        """Compute centrality and node-level metrics"""
        print("COMPUTING NODE METRICS")
        
        metrics = []
        nodes = list(self.G.nodes())
        
        print(f"\n Computing metrics for {len(nodes):,} nodes...")
        
        # Degree centrality
        print("Degree centrality...")
        in_degree_cent = nx.in_degree_centrality(self.G)
        out_degree_cent = nx.out_degree_centrality(self.G)
        
        print("PageRank...")
        pagerank = nx.pagerank(self.G, alpha=0.85)
        
        print("HITS (hubs and authorities)...")
        hits_h, hits_a = nx.hits(self.G, max_iter=100)
        
        # Betweenness (sample for large networks)
        print("Betweenness centrality...")
        if len(nodes) > 50:
            print(f"Sampling 50 nodes for betweenness (for p2 we choose to test with only 50 nodes)")
            betweenness = nx.betweenness_centrality(self.G, k=50)
        else:
            betweenness = nx.betweenness_centrality(self.G)
        
        # Clustering coefficient
        print("Clustering coefficient...")
        clustering = nx.clustering(self.G_undirected)
        
        # Assemble per-node metrics
        print("\n Assembling per-node metrics...")
        for node in nodes:
            # Basic degree info
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            
            # Weighted degrees
            in_weight = sum(self.G[u][node]['weight'] for u in self.G.predecessors(node))
            out_weight = sum(self.G[node][v]['weight'] for v in self.G.successors(node))
            
            # Sentiment of connections
            in_edges = [(u, node) for u in self.G.predecessors(node)]
            out_edges = [(node, v) for v in self.G.successors(node)]
            
            avg_in_sentiment = np.mean([self.G[u][v]['avg_sentiment'] 
                                        for u, v in in_edges]) if in_edges else 0
            avg_out_sentiment = np.mean([self.G[u][v]['avg_sentiment'] 
                                         for u, v in out_edges]) if out_edges else 0
            
            # Count positive/negative 
            n_pos_in = sum(self.G[u][node]['n_positive'] for u in self.G.predecessors(node))
            n_neg_in = sum(self.G[u][node]['n_negative'] for u in self.G.predecessors(node))
            n_pos_out = sum(self.G[node][v]['n_positive'] for v in self.G.successors(node))
            n_neg_out = sum(self.G[node][v]['n_negative'] for v in self.G.successors(node))
            
            metrics.append({
                'subreddit': node,
                'in_degree': in_deg,
                'out_degree': out_deg,
                'total_degree': in_deg + out_deg,
                'in_weight': in_weight,
                'out_weight': out_weight,
                'total_weight': in_weight + out_weight,
                'in_degree_centrality': in_degree_cent[node],
                'out_degree_centrality': out_degree_cent[node],
                'pagerank': pagerank[node],
                'hub_score': hits_h[node],
                'authority_score': hits_a[node],
                'betweenness': betweenness.get(node, 0), 
                'clustering': clustering.get(node, 0), 
                'avg_in_sentiment': avg_in_sentiment,
                'avg_out_sentiment': avg_out_sentiment,
                'n_positive_in': n_pos_in,
                'n_negative_in': n_neg_in,
                'n_positive_out': n_pos_out,
                'n_negative_out': n_neg_out,
                'neg_in_ratio': n_neg_in / in_weight if in_weight > 0 else 0,
                'neg_out_ratio': n_neg_out / out_weight if out_weight > 0 else 0
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        print(f"\n Computed {len(metrics_df.columns)} metrics for {len(metrics_df):,} nodes")
        
        # Show top nodes by different metrics
        print(f"\n Top 5 nodes by key metrics:")
        
        for metric in ['pagerank', 'hub_score', 'authority_score', 'betweenness']:
            top5 = metrics_df.nlargest(5, metric)
            print(f"\n  {metric.replace('_', ' ').title()}:")
            for _, row in top5.iterrows():
                print(f"{row['subreddit']:<25} {row[metric]:>10.6f}")
        
        return metrics_df
    
    def detect_communities(self, resolution=1.0):
        """Detect communities using Louvain algorithm"""
        print("COMMUNITY DETECTION")
        
        print(f"\n Running Louvain algorithm (resolution={resolution})...")
        
        # Use undirected graph
        partition = community_louvain.best_partition(
            self.G_undirected,
            resolution=resolution,
            random_state=42
        )
        
        # Add to dataframe
        community_df = pd.DataFrame([
            {'subreddit': node, 'community': comm}
            for node, comm in partition.items()
        ])
        
        community_sizes = community_df['community'].value_counts().sort_index()
        n_communities = len(community_sizes)
        
        print(f"\n Found {n_communities} communities")
        print(f"\n Community size distribution:")
        print(f"Largest:  {community_sizes.max():>6,} subreddits")
        print(f"Median:   {community_sizes.median():>6.0f} subreddits")
        print(f"Smallest: {community_sizes.min():>6,} subreddits")
        
        # Show largest communities
        print(f"\n Top 10 largest communities:")
        for comm_id, size in community_sizes.head(10).items():
            members = community_df[community_df['community'] == comm_id]['subreddit'].tolist()
            sample = ', '.join(members[:5])
            print(f"Community {comm_id:>3}: {size:>5,} members - {sample}...")
        
        modularity = community_louvain.modularity(partition, self.G_undirected)
        print(f"\n Modularity: {modularity:.4f}")
        
        return community_df, modularity
    
    def analyze_sentiment_structure(self):
        """Analyze how sentiment relates to network structure"""
        print("SENTIMENT-STRUCTURE ANALYSIS")
        
        metrics_to_compare = [
            ('Nodes', 'number_of_nodes'),
            ('Edges', 'number_of_edges'),
            ('Density', 'density'),
            ('Avg degree', 'avg_degree')
        ]
        
        for name, attr in metrics_to_compare:
            if attr == 'density':
                pos_val = nx.density(self.G_pos)
                neg_val = nx.density(self.G_neg)
                print(f"    {name:<30} {pos_val:<15.6f} {neg_val:<15.6f}")
            elif attr == 'avg_degree':
                pos_val = np.mean([d for n, d in self.G_pos.degree()])
                neg_val = np.mean([d for n, d in self.G_neg.degree()])
                print(f"    {name:<30} {pos_val:<15.2f} {neg_val:<15.2f}")
            else:
                pos_val = getattr(self.G_pos, attr)()
                neg_val = getattr(self.G_neg, attr)()
                print(f"    {name:<30} {pos_val:<15,} {neg_val:<15,}")
        
        # Edge sentiment distribution
        sentiments = [data['avg_sentiment'] for u, v, data in self.G.edges(data=True)]
        
        print(f"\n Edge sentiment distribution:")
        print(f"Mean:   {np.mean(sentiments):>8.3f}")
        print(f"Median: {np.median(sentiments):>8.3f}")
        print(f"Std:    {np.std(sentiments):>8.3f}")
        print(f"Min:    {np.min(sentiments):>8.3f}")
        print(f"Max:    {np.max(sentiments):>8.3f}")
        
        return {
            'pos_density': nx.density(self.G_pos),
            'neg_density': nx.density(self.G_neg),
            'sentiment_mean': np.mean(sentiments),
            'sentiment_std': np.std(sentiments)
        }
    
    def save_results(self):
        """Save all network analysis results"""
        print("SAVING NETWORK RESULTS")
        
        print("\n Saving basic statistics...")
        stats = self.compute_basic_stats()
        pd.DataFrame([stats]).to_csv(
            self.output_dir / "network_basic_stats.csv",
            index=False
        )
        
        print(" Saving node metrics...")
        metrics_df = self.compute_node_metrics()
        metrics_df.to_csv(
            self.output_dir / "network_node_metrics.csv",
            index=False
        )
        
        print(" Detecting and saving communities...")
        community_df, _ = self.detect_communities()
        community_df.to_csv(
            self.output_dir / "network_communities.csv",
            index=False
        )
        
        print(" Analyzing sentiment structure...")
        sentiment_stats = self.analyze_sentiment_structure()
        pd.DataFrame([sentiment_stats]).to_csv(
            self.output_dir / "network_sentiment_stats.csv",
            index=False
        )
        
        print(" Saving edge list...")
        edges_data = []
        for u, v, data in self.G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data['weight'],
                'avg_sentiment': data['avg_sentiment'],
                'n_positive': data['n_positive'],
                'n_negative': data['n_negative']
            })
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(
            self.output_dir / "network_edges.csv",
            index=False
        )
        
        print(f"\n All network results saved to: {self.output_dir}")
        
        return {
            'stats': stats,
            'metrics': metrics_df,
            'communities': community_df,
            'edges': edges_df
        }


def analyze_network(hyperlinks_df):
    """Main function to run network analysis"""
    print("NETWORK STRUCTURE ANALYSIS")
    
    analyzer = NetworkAnalyzer(hyperlinks_df)
    analyzer.build_networks()
    results = analyzer.save_results()
    
    print(" NETWORK ANALYSIS COMPLETE")
    
    return results, analyzer.G


if __name__ == "__main__":
    pass