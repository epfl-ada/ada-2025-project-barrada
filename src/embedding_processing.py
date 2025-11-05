import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class EmbeddingProcessor:
    """Process subreddit embeddings"""
    
    def __init__(self, embeddings_path, output_dir="data/processed"):
        self.embeddings_path = embeddings_path
        ROOT = Path(__file__).resolve().parent.parent
        self.output_dir = Path(ROOT / output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.embedding_dim = 300
        
    def load_embeddings(self):
        """Load embeddings with column handling"""
        print("Loading.....")
        
        df = pd.read_csv(self.embeddings_path, header=None)
        
        print(f"Raw shape: {df.shape}")
        
        column_names = ['subreddit'] + [f'emb_{i}' for i in range(self.embedding_dim)]
        
        df.columns = column_names
        print(f" Columns: 1 subreddit + {self.embedding_dim} embedding dimensions")

                
        n_before = len(df)
        df = df.drop_duplicates(subset='subreddit', keep='first')
        n_removed = n_before - len(df)
        print(f" Removed {n_removed} duplicate subreddits")
        
        print(f"\n Loaded embeddings for {len(df):,} subreddits")
        
        return df
    
    def validate_embeddings(self, df):
        """Check embedding quality"""
        print("Embedding validation")
        
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        
        n_missing = df[emb_cols].isnull().sum().sum()
        if n_missing > 0:
            print(f"Found {n_missing:,} missing values in embeddings")
        else:
            print(" No missing values")
        
        emb_matrix = df[emb_cols].values
        
        print(f"\n Embedding statistics:")
        print(f"    Mean:   {emb_matrix.mean():.6f}")
        print(f"    Std:    {emb_matrix.std():.6f}")
        print(f"    Min:    {emb_matrix.min():.6f}")
        print(f"    Max:    {emb_matrix.max():.6f}")
        
        return df
    
    def compute_pca(self, df, n_components=50):
        """Reduce dimensionality with PCA"""
        print("PCA Dimensionality Reduction....")
        
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        X = df[emb_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\n Computing PCA ({n_components} components)...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_cols = [f'pca_{i}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        df = pd.concat([df, pca_df], axis=1)
        
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        print(f"\n Variance explained:")
        print(f"All {n_components} components: {cumsum_var[-1]:.1%}")
        
        variance_df = pd.DataFrame({
            'component': range(1, n_components + 1),
            'explained_variance': explained_var,
            'cumulative_variance': cumsum_var
        })
        variance_df.to_csv(self.output_dir / "pca_variance.csv", index=False)
        
        return df, pca
    
    def save_processed(self, df, filename="embeddings_processed.csv"):
        """Save processed embeddings"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n Saved processed embeddings: {output_path}")
        
        return output_path
    
    def run(self, n_pca=50):
        """Execute embedding processing pipeline"""
        print("Embedding Processing....")
        
        df = self.load_embeddings()
        df = self.validate_embeddings(df)
        df, pca = self.compute_pca(df, n_components=n_pca)

        self.save_processed(df)
        
        print(" Embedding Processing Complete")        
        return df, pca

def process_embeddings(
    embeddings_path=None,
    output_dir=None
    ):
    """Main function to run the embedding processing pipeline"""
    if embeddings_path is None or output_dir is None:
        try:
            HERE = Path(__file__).resolve().parent.parent
        except NameError:
            HERE = Path('.').resolve()
            
        DATA_DIR = HERE / "data"
        embeddings_path = embeddings_path or DATA_DIR / "subreddit_embeddings" / "web-redditEmbeddings-subreddits.csv"
        output_dir = output_dir or HERE / "data" / "processed"

    processor = EmbeddingProcessor(
        embeddings_path=embeddings_path,
        output_dir=str(output_dir)
    )
    df, pca = processor.run(n_pca=50)
    return df, pca 

if __name__ == "__main__":
    embeddings_df, pca = process_embeddings()
    print(f"Processed embeddings for {len(embeddings_df):,} subreddits")