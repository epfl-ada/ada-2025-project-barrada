import pandas as pd
import numpy as np
from pathlib import Path

class HyperlinkDataProcessor:
    """Process and combine Reddit hyperlink data from body and title sources"""
    
    def __init__(self, body_path, title_path, output_dir="data/processed"):
        self.body_path = body_path
        self.title_path = title_path 
        ROOT = Path(__file__).resolve().parent.parent
        self.output_dir = Path(ROOT / output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.raw_columns = [
            'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 
            'TIMESTAMP', 'LINK_SENTIMENT', 'PROPERTIES'
        ]
        
        # LIWC column (22-86)
        self.liwc_start_idx = 21
        self.liwc_end_idx = 86
        
        self.liwc_columns = [
            'LIWC_Funct', 'LIWC_Pronoun', 'LIWC_Ppron', 'LIWC_I', 'LIWC_We',
            'LIWC_You', 'LIWC_SheHe', 'LIWC_They', 'LIWC_Ipron', 'LIWC_Article',
            'LIWC_Verbs', 'LIWC_AuxVb', 'LIWC_Past', 'LIWC_Present', 'LIWC_Future',
            'LIWC_Adverbs', 'LIWC_Prep', 'LIWC_Conj', 'LIWC_Negate', 'LIWC_Quant',
            'LIWC_Numbers', 'LIWC_Swear', 'LIWC_Social', 'LIWC_Family', 'LIWC_Friends',
            'LIWC_Humans', 'LIWC_Affect', 'LIWC_Posemo', 'LIWC_Negemo', 'LIWC_Anx',
            'LIWC_Anger', 'LIWC_Sad', 'LIWC_CogMech', 'LIWC_Insight', 'LIWC_Cause',
            'LIWC_Discrep', 'LIWC_Tentat', 'LIWC_Certain', 'LIWC_Inhib', 'LIWC_Incl',
            'LIWC_Excl', 'LIWC_Percept', 'LIWC_See', 'LIWC_Hear', 'LIWC_Feel',
            'LIWC_Bio', 'LIWC_Body', 'LIWC_Health', 'LIWC_Sexual', 'LIWC_Ingest',
            'LIWC_Relativ', 'LIWC_Motion', 'LIWC_Space', 'LIWC_Time', 'LIWC_Work',
            'LIWC_Achiev', 'LIWC_Leisure', 'LIWC_Home', 'LIWC_Money', 'LIWC_Relig',
            'LIWC_Death', 'LIWC_Assent', 'LIWC_Dissent', 'LIWC_Nonflu', 'LIWC_Filler'
        ]
        
        # Text property column (indices 1-21)
        self.text_property_columns = [
            'n_characters', 'n_characters_no_space', 'frac_alpha', 'frac_digits',
            'frac_uppercase', 'frac_whitespace', 'frac_special', 'n_words',
            'n_unique_words', 'n_long_words', 'avg_word_length', 'n_unique_stopwords',
            'frac_stopwords', 'n_sentences', 'n_long_sentences', 'avg_chars_per_sentence',
            'avg_words_per_sentence', 'readability_index', 'vader_positive',
            'vader_negative', 'vader_compound'
        ]
        
    def parse_post_properties(self, properties_str):
        """Parse PROPERTIES string into individual features"""
        try:
            return [float(x) for x in properties_str.split(',')]
        except (AttributeError, ValueError, TypeError):
            return None
    
    def load_and_combine(self):
        """Load both TSV files and combine them"""
        print("LOADING HYPERLINK DATA")
        
        print("\n Loading files...")
        body_df = pd.read_csv(
            self.body_path, 
            sep='\t', 
            header=0, 
            names=self.raw_columns
        )
        title_df = pd.read_csv(
            self.title_path, 
            sep='\t', 
            header=0, 
            names=self.raw_columns
        )
        
        print(f"Body links:  {len(body_df):>8,} rows")
        print(f"Title links: {len(title_df):>8,} rows")
        
        body_df['LINK_SOURCE'] = 'body'
        title_df['LINK_SOURCE'] = 'title'
        
        combined = pd.concat([body_df, title_df], ignore_index=True)
        print(f"Combined: {len(combined):>8,} rows")
        
        return combined
    
    def expand_post_properties(self, df):
        """Extract LIWC and text features"""
        print("\nExtracting features from PROPERTIES...")
        
        # Parse POST_PROPERTIES
        df['properties_parsed'] = df['PROPERTIES'].apply(self.parse_post_properties)
        
        valid_mask = df['properties_parsed'].notna()
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"Removed {n_invalid:,} rows with invalid PROPERTIES")
            df = df[valid_mask].copy()
                
        props_df = pd.DataFrame(
            df['properties_parsed'].tolist(), 
            index=df.index
        )
        
        # Extract text properties
        text_cols_map = {i: name for i, name in enumerate(self.text_property_columns)}
        df = df.join(props_df[list(text_cols_map.keys())].rename(columns=text_cols_map))
        
        # Extract LIWC
        liwc_cols_map = {self.liwc_start_idx + i: name for i, name in enumerate(self.liwc_columns)}
        df = df.join(props_df[list(liwc_cols_map.keys())].rename(columns=liwc_cols_map))

        df = df.drop(columns=['properties_parsed'])
        
        print(f"Extracted {len(self.text_property_columns)} text properties")
        print(f"Extracted {len(self.liwc_columns)} LIWC features")
        
        return df
    
    def validate_data(self, df):
        """DATA VALIDATION"""
        print("DATA VALIDATION")
        
        # Check required columns
        required_cols = self.raw_columns + ['timestamp', 'is_positive', 'is_negative']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after cleaning: {missing}")
        print("All required columns present")
        
        # Check for nulls in critical columns
        critical_cols = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT', 'TIMESTAMP']
        null_counts = df[critical_cols].isnull().sum()
        if null_counts.any():
            print(f"\n Null values in critical columns:")
            for col, count in null_counts[null_counts > 0].items():
                print(f" {col}: {count:,}")
        else:
            print("No nulls in critical columns")
        
        # Sentiment distribution
        print(f"\nSentiment distribution (LINK_SENTIMENT):")
        sentiment_dist = df['LINK_SENTIMENT'].value_counts().sort_index()
        for label, count in sentiment_dist.items():
            pct = 100 * count / len(df)
            print(f" {label:>2}: {count:>8,} ({pct:>5.1f}%)")
        
        # Network size
        n_sources = df['SOURCE_SUBREDDIT'].nunique()
        n_targets = df['TARGET_SUBREDDIT'].nunique()
        all_subs = set(df['SOURCE_SUBREDDIT']).union(set(df['TARGET_SUBREDDIT']))
        
        print(f"\nNetwork composition:")
        print(f"Unique subreddits:  {len(all_subs):>8,}")
        print(f"As sources:         {n_sources:>8,}")
        print(f"As targets:         {n_targets:>8,}")
        print(f"Unique links:       {len(df):>8,}")
      
        print(f"\n Time range:")
        print(f"From: {df['timestamp'].min()}")
        print(f"To:   {df['timestamp'].max()}")
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print("DATA CLEANING")
        
        original_len = len(df)
        
        # Remove self-loops
        df = df[df['SOURCE_SUBREDDIT'] != df['TARGET_SUBREDDIT']].copy()
        n_self_loops = original_len - len(df)
        if n_self_loops > 0:
            print(f"Removed {n_self_loops:,} self-loops")
        
        # Standardize names
        df['SOURCE_SUBREDDIT'] = df['SOURCE_SUBREDDIT'].str.lower().str.strip()
        df['TARGET_SUBREDDIT'] = df['TARGET_SUBREDDIT'].str.lower().str.strip()
        print(" Standardized subreddit names (lowercase, trimmed)")
        
        df['sentiment_numeric'] = df['LINK_SENTIMENT']
        df['is_positive'] = (df['LINK_SENTIMENT'] == 1).astype(int)
        df['is_negative'] = (df['LINK_SENTIMENT'] == -1).astype(int)
        print("Created binary sentiment indicators")
        
        if 'TIMESTAMP' in df.columns:
            df['timestamp'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['date'] = df['timestamp'].dt.date
            print("Parsed timestamp into datetime, year, month, date")
        
        # Create aggregated LIWC categories
        df = self.create_liwc_aggregates(df)
        
        print(f"\n Final dataset: {len(df):,} rows × {len(df.columns)} columns")
        
        return df
    
    def create_liwc_aggregates(self, df):
        """Create higher-level LIWC category aggregates"""
        
        # Emotional categories
        emotion_cols = ['LIWC_Affect', 'LIWC_Posemo', 'LIWC_Negemo']
        if all(col in df.columns for col in emotion_cols):
            df['LIWC_Emotion_Total'] = df[emotion_cols].mean(axis=1)
        
        neg_emotion_cols = ['LIWC_Anx', 'LIWC_Anger', 'LIWC_Sad']
        if all(col in df.columns for col in neg_emotion_cols):
            df['LIWC_Negemo_Specific'] = df[neg_emotion_cols].mean(axis=1)
        
        cog_cols = ['LIWC_CogMech', 'LIWC_Insight', 'LIWC_Cause', 'LIWC_Discrep', 
                    'LIWC_Tentat', 'LIWC_Certain']
        if all(col in df.columns for col in cog_cols):
            df['LIWC_Cognitive_Total'] = df[cog_cols].mean(axis=1)
         
        social_cols = ['LIWC_Social', 'LIWC_Family', 'LIWC_Friends', 'LIWC_Humans']
        if all(col in df.columns for col in social_cols):
            df['LIWC_Social_Total'] = df[social_cols].mean(axis=1)
        
        pronoun_cols = ['LIWC_I', 'LIWC_We', 'LIWC_You', 'LIWC_They']
        if all(col in df.columns for col in pronoun_cols):
            df['LIWC_Pronoun_Total'] = df[pronoun_cols].mean(axis=1)
        
        return df
    
    def save_processed(self, df, filename="combined_hyperlinks.csv"):
        """Save processed data"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
                
        print(f"\n Saved to: {output_path}")
        return output_path
    
    def run(self):
        """Execute full preprocessing pipeline"""
        print("Data Preprocessing....")
        
        df = self.load_and_combine()
        df = self.expand_post_properties(df)
        df = self.clean_data(df)
        df = self.validate_data(df)
        self.save_processed(df)
        
        print("PREPROCESSING COMPLETE")       
        return df, self.liwc_columns


def process_hyperlinks(
    body_path=Path("data/hyperlink_network/soc-redditHyperlinks-body.tsv"),
    title_path=Path("data/hyperlink_network/soc-redditHyperlinks-title.tsv"),
    output_dir=Path("data/processed")
    ):
    """
    Main function to run the preprocessing pipeline.
    """

    ROOT = Path(__file__).resolve().parent.parent
    body_path = ROOT / body_path
    title_path = ROOT / title_path
    output_dir = ROOT / output_dir

    processor = HyperlinkDataProcessor(
        body_path=body_path,
        title_path=title_path,
        output_dir=str(output_dir)
    )
    df, liwc_cols = processor.run()
    return df, liwc_cols

if __name__ == "__main__":

    df, liwc_columns = process_hyperlinks()
    print(f"Loaded {len(df):,} hyperlinks")
    print(f"LIWC columns: {len(liwc_columns)}")