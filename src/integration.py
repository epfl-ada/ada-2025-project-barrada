import pandas as pd
import numpy as np
from pathlib import Path

class DataIntegrator:
    """Merge network, LIWC, and embedding features into unified dataset"""
    
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = Path(processed_dir)
        self.final_df = None
        
    def load_all_features(self):
        """Load all processed feature files"""
        print("LOADING....")
        
        features = {}
        
        # 1. Network metrics
        print("\n Loading network metrics...")
        features['network'] = pd.read_csv(
            self.processed_dir / "network_node_metrics.csv"
        )
        # Check for duplicates
        dups = features['network']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f" Found {dups} duplicate subreddits - removing...")
            features['network'] = features['network'].drop_duplicates(subset='subreddit', keep='first')
        print(f"  {len(features['network']):,} nodes * {len(features['network'].columns)} features")
        
        # 2. LIWC source features
        print(" Loading LIWC source features...")
        features['liwc_source'] = pd.read_csv(
            self.processed_dir / "subreddit_features_source.csv"
        )
        dups = features['liwc_source']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f"   Found {dups} duplicate subreddits - removing...")
            features['liwc_source'] = features['liwc_source'].drop_duplicates(subset='subreddit', keep='first')
        print(f" {len(features['liwc_source']):,} subreddits * {len(features['liwc_source'].columns)} features")
        
        # 3. LIWC target features
        print("Loading LIWC target features...")
        features['liwc_target'] = pd.read_csv(
            self.processed_dir / "subreddit_features_target.csv"
        )
        dups = features['liwc_target']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f"  Found {dups} duplicate subreddits - removing...")
            features['liwc_target'] = features['liwc_target'].drop_duplicates(subset='subreddit', keep='first')
        print(f"  {len(features['liwc_target']):,} subreddits * {len(features['liwc_target'].columns)} features")
        
        # 4. Embeddings
        print(" Loading embeddings...")
        features['embeddings'] = pd.read_csv(
            self.processed_dir / "embeddings_processed.csv"
        )
        dups = features['embeddings']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f"  Found {dups} duplicate subreddits - removing...")
            features['embeddings'] = features['embeddings'].drop_duplicates(subset='subreddit', keep='first')
        print(f"  {len(features['embeddings']):,} subreddits * {len(features['embeddings'].columns)} features")
        
        # 5. Communities
        print(" Loading community assignments...")
        features['communities'] = pd.read_csv(
            self.processed_dir / "network_communities.csv"
        )
        dups = features['communities']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f" Found {dups} duplicate subreddits - removing...")
            features['communities'] = features['communities'].drop_duplicates(subset='subreddit', keep='first')
        print(f" {len(features['communities']):,} subreddits")
        
        # 6. Roles
        print(" Loading psychological roles...")
        features['roles'] = pd.read_csv(
            self.processed_dir / "subreddit_roles.csv"
        )
        dups = features['roles']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f" Found {dups} duplicate subreddits - removing...")
            features['roles'] = features['roles'].drop_duplicates(subset='subreddit', keep='first')
        print(f"  {len(features['roles']):,} subreddits * {len(features['roles'].columns)} features")
        print(f"  Roles columns: {list(features['roles'].columns)}") 

        # 7. Topic clusters (human-readable) 
        print(" Loading topic clusters...")
        features['topic_clusters'] = pd.read_csv(
            self.processed_dir / "embeddings_kmeans_40.csv"
        )
        dups = features['topic_clusters']['subreddit'].duplicated().sum()
        if dups > 0:
            print(f" Found {dups} duplicate subreddits - removing...")
            features['topic_clusters'] = features['topic_clusters'].drop_duplicates(subset='subreddit', keep='first')
        print(f" {len(features['topic_clusters']):,} subreddits * {len(features['topic_clusters'].columns)} features")
        
        return features
    
    def merge_features(self, features):
        """Merge all feature datasets on subreddit"""
        print("MERGING ALL FEATURES")
        
        # Start with network metrics as base
        merged = features['network'].copy()
        merged = merged.rename(columns={'subreddit': 'subreddit'})
        
        print(f"\n Starting with network metrics: {len(merged):,} subreddits")
    
        # Merge LIWC source features
        print(" Merging LIWC source features...")
        liwc_source_subset = features['liwc_source'][
            ['subreddit'] + [col for col in features['liwc_source'].columns 
                            if col != 'subreddit' and col not in merged.columns]
        ]
        merged = merged.merge(
            liwc_source_subset,
            on='subreddit',
            how='left',
            suffixes=('', '_liwc_src')
        )
        print(f" Now have {len(merged.columns)} columns")
        
        # Merge LIWC target features
        print(" Merging LIWC target features...")
        liwc_target_subset = features['liwc_target'][
            ['subreddit'] + [col for col in features['liwc_target'].columns 
                            if col != 'subreddit' and col not in merged.columns]
        ]

        # Add suffix to distinguish incoming vs outgoing
        liwc_target_subset = liwc_target_subset.rename(
            columns={col: f"{col}_target" for col in liwc_target_subset.columns 
                    if col != 'subreddit'}
        )
        merged = merged.merge(
            liwc_target_subset,
            on='subreddit',
            how='left'
        )
        print(f" Now have {len(merged.columns)} columns")
        
        # Merge embeddings (keep only PCA components to reduce dimensionality)
        print(" Merging embeddings (PCA components)...")
        emb_cols = ['subreddit'] + [col for col in features['embeddings'].columns 
                                     if col.startswith('pca_')]
        embeddings_subset = features['embeddings'][emb_cols]
        merged = merged.merge(
            embeddings_subset,
            on='subreddit',
            how='left'
        )
        print(f" Now have {len(merged.columns)} columns")

        # Merge topic clusters 
        print(" Merging topic clusters...")
        merged = merged.merge(features['topic_clusters'][['subreddit', 'topic_cluster', 'topic_cluster_label']],
                              on='subreddit', how='left')
        print(f" Now have {len(merged.columns)} columns after topic clusters")
                
        # Merge communities
        print(" Merging community assignments...")
        merged = merged.merge(
            features['communities'],
            on='subreddit',
            how='left'
        )
        print(f" Now have {len(merged.columns)} columns")

        # Check what columns already exist from network metrics
        existing_ratio_cols = [col for col in merged.columns 
                              if col in ['neg_out_ratio', 'neg_in_ratio', 'pos_out_ratio', 'pos_in_ratio']]
        
        if existing_ratio_cols:
            print(f" Ratio columns already exist from network: {existing_ratio_cols}")
            print(f"Will use network versions and skip roles versions")
        
        # Get available ratio columns from roles that don't conflict
        available_ratio_cols = [col for col in ['neg_out_ratio', 'neg_in_ratio', 
                                                 'pos_out_ratio', 'pos_in_ratio']
                               if col in features['roles'].columns and col not in existing_ratio_cols]
        
        # Select columns: subreddit + role columns + non-conflicting ratios
        role_cols = ['subreddit'] + \
            [col for col in features['roles'].columns if col.startswith('role_')] + \
            available_ratio_cols + \
            ['total_links']
        
        # Only keep columns that actually exist
        role_cols = [col for col in role_cols if col in features['roles'].columns]
        
        print(f" Merging {len(role_cols)-1} role features")
        
        roles_subset = features['roles'][role_cols]
        merged = merged.merge(
            roles_subset,
            on='subreddit',
            how='left'
        )
        
        print(f"\n Final merged dataset: {len(merged):,} subreddits × {len(merged.columns)} features")
        
        return merged
    
        
    def save_final_dataset(self, df, filename="final_dataset.csv"):
        """Save the integrated dataset"""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
                
        print(f"\ Saved final dataset: {output_path}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return output_path
    
    
    def run(self):
        """Execute full integration pipeline"""
        print("Integration Pipeline...")
        
        features = self.load_all_features()
        merged = self.merge_features(features)
        
        print(" DATA INTEGRATION COMPLETE")
        self.save_final_dataset(merged)
        return merged


def integrate_all_data():
    """Main function to run data integration"""
    integrator = DataIntegrator()
    final_df = integrator.run()
    return final_df


if __name__ == "__main__":
    final_df = integrate_all_data()
    print(f"Final dataset: {final_df.shape}")