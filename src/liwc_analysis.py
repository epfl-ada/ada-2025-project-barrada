from typing import List, Dict, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

class LIWCAnalyzer:
    """Analyze psychological patterns in inter-subreddit communication"""
    
    def __init__(self, 
                 hyperlinks_df: pd.DataFrame, 
                 liwc_columns: List[str], 
                 output_dir: Union[str, Path] = "data/processed"
                 ) -> None:
        
        self.df = hyperlinks_df
        self.liwc_columns = liwc_columns
        ROOT = Path(__file__).resolve().parent.parent
        self.output_dir = Path(ROOT / output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Key LIWC dimension groups
        self.liwc_groups = {
            'Emotion': ['LIWC_Affect', 'LIWC_Posemo', 'LIWC_Negemo', 'LIWC_Anx', 
                       'LIWC_Anger', 'LIWC_Sad'],
            'Social': ['LIWC_Social', 'LIWC_Family', 'LIWC_Friends', 'LIWC_Humans'],
            'Cognitive': ['LIWC_CogMech', 'LIWC_Insight', 'LIWC_Cause', 
                         'LIWC_Discrep', 'LIWC_Tentat', 'LIWC_Certain'],
            'Pronouns': ['LIWC_I', 'LIWC_We', 'LIWC_You', 'LIWC_They'],
            'Drives': ['LIWC_Work', 'LIWC_Achiev', 'LIWC_Leisure', 'LIWC_Home', 
                      'LIWC_Money', 'LIWC_Relig', 'LIWC_Death']
        }
    
    def aggregate_by_subreddit(self, role: str = 'source') -> pd.DataFrame:
        """
        Aggregate LIWC features at subreddit level
        """

        print(f"AGGREGATING LIWC BY {role.upper()} SUBREDDIT")
        
        group_col = 'SOURCE_SUBREDDIT' if role == 'source' else 'TARGET_SUBREDDIT'
        
        # Basic link statistics
        print(f"\n Computing basic statistics...")
        basic_stats = self.df.groupby(group_col).agg({
            'POST_ID': 'count',
            'sentiment_numeric': ['mean'],
            'is_positive': 'sum',
            'is_negative': 'sum',
            'TARGET_SUBREDDIT' if role == 'source' else 'SOURCE_SUBREDDIT': 'nunique'
        }).reset_index()
        
        basic_stats.columns = [
            'subreddit',
            'n_links',
            'avg_sentiment',
            'n_positive_links', 'n_negative_links',
            'n_unique_targets' if role == 'source' else 'n_unique_sources'
        ]
        
        # LIWC aggregation
        print(f" Aggregating {len(self.liwc_columns)} LIWC features...")
        
        liwc_stats = self.df.groupby(group_col)[self.liwc_columns].agg(
            ['mean']
        ).reset_index()
        
        # Flatten column names
        liwc_stats.columns = ['subreddit'] + [
            f"{col}_{stat}" for col, stat in liwc_stats.columns[1:]
        ]
        
        # Merge
        result = basic_stats.merge(liwc_stats, on='subreddit', how='left')
        
        # Derived metrics
        result['negativity_ratio'] = result['n_negative_links'] / result['n_links']
        result['positivity_ratio'] = result['n_positive_links'] / result['n_links']
        result['sentiment_balance'] = result['positivity_ratio'] - result['negativity_ratio']
        
        # Add aggregated LIWC group scores
        for group_name, features in self.liwc_groups.items():
            mean_cols = [f"{f}_mean" for f in features if f in self.liwc_columns]
            if mean_cols:
                result[f'{group_name}_score'] = result[mean_cols].mean(axis=1)
        
        print(f"\n Aggregated {len(result):,} subreddits")
        print(f" Features per subreddit: {len(result.columns)}")
        
        return result
    
    def compare_sentiment_groups(self) -> pd.DataFrame:
        """Compare LIWC profiles of positive vs negative links"""
        print("COMPARING POSITIVE VS NEGATIVE LINKS")
        
        pos_links = self.df[self.df['is_positive'] == 1]
        neg_links = self.df[self.df['is_negative'] == 1]
        
        print(f"\n Sample sizes:")
        print(f"    Positive links: {len(pos_links):>10,}")
        print(f"    Negative links: {len(neg_links):>10,}")
        
        # Compare LIWC means
        comparison = []
        
        print(f"\n Computing differences across {len(self.liwc_columns)} features...")
        
        for feature in self.liwc_columns:
            pos_mean = pos_links[feature].mean()
            neg_mean = neg_links[feature].mean()

            
            # T-test
            try:
                t_stat, p_value = stats.ttest_ind(
                    neg_links[feature].dropna(),
                    pos_links[feature].dropna()
                )
            except:
                t_stat, p_value = np.nan, np.nan
            
            comparison.append({
                'feature': feature,
                'positive_mean': pos_mean,
                'negative_mean': neg_mean,
                'difference': neg_mean - pos_mean,
                'abs_difference': abs(neg_mean - pos_mean),
                't_statistic': t_stat,
                'p_value': p_value
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
        
        # Display top differences
        print(f"\nTop 15 most discriminating features:")
        display_cols = ['feature', 'positive_mean', 'negative_mean', 'difference']
        display_df = comparison_df.head(15)[display_cols]
        
        for _, row in display_df.iterrows():
            direction = "↑" if row['difference'] > 0 else "↓"
            print(f"  {row['feature']:<20} "
                  f"Pos: {row['positive_mean']:>6.2f}  "
                  f"Neg: {row['negative_mean']:>6.2f}  "
                  f"{direction} Δ: {abs(row['difference']):>6.2f}")
        
        return comparison_df
    
    def analyze_psychological_roles(self, top_n: int = 50) -> pd.DataFrame:
        """
        Identify psychological roles of communities
        (Critics, Targets, Amplifiers, etc.)
        """
        print("Identify psychological roles")
        
        # Get source and target features
        source_features = self.aggregate_by_subreddit(role='source')
        target_features = self.aggregate_by_subreddit(role='target')
        
        # Merge on subreddit
        merged = source_features.merge(
            target_features,
            on='subreddit',
            how='outer',
            suffixes=('_out', '_in')
        )
        
        fill_cols = {
            'n_links_out': 0, 'n_links_in': 0,
            'n_positive_links_out': 0, 'n_positive_links_in': 0,
            'n_negative_links_out': 0, 'n_negative_links_in': 0,
            'n_unique_targets': 0, 'n_unique_sources': 0,
            'avg_sentiment_out': 0, 'avg_sentiment_in': 0
        }
        
        fill_cols_exist = {k: v for k, v in fill_cols.items() if k in merged.columns}
        merged = merged.fillna(fill_cols_exist)
        
        print(f" Merged {len(merged):,} subreddits (source + target)")
        print(f"  Only sources: {(merged['n_links_in'] == 0).sum()}")
        print(f"  Only targets: {(merged['n_links_out'] == 0).sum()}")
        print(f"  Both: {((merged['n_links_in'] > 0) & (merged['n_links_out'] > 0)).sum()}")
                
        merged['total_links'] = merged['n_links_out'] + merged['n_links_in']
        
        # Calculate ratios with safe division TO avoid divide by zero
        merged['neg_out_ratio'] = np.where(
            merged['n_links_out'] > 0,
            merged['n_negative_links_out'] / merged['n_links_out'],
            0.0
        )
        merged['neg_in_ratio'] = np.where(
            merged['n_links_in'] > 0,
            merged['n_negative_links_in'] / merged['n_links_in'],
            0.0
        )
        merged['pos_out_ratio'] = np.where(
            merged['n_links_out'] > 0,
            merged['n_positive_links_out'] / merged['n_links_out'],
            0.0
        )
        merged['pos_in_ratio'] = np.where(
            merged['n_links_in'] > 0,
            merged['n_positive_links_in'] / merged['n_links_in'],
            0.0
        )
        
        # Role classification
        merged['role_critical'] = (merged['neg_out_ratio'] > 0.2) & (merged['n_links_out'] >= 10)
        merged['role_controversial'] = (merged['neg_in_ratio'] > 0.2) & (merged['n_links_in'] >= 10)
        merged['role_supportive'] = (merged['pos_in_ratio'] > 0.2) & (merged['n_links_in'] >= 10)
        merged['role_influential'] = (merged['pos_out_ratio'] > 0.2) & (merged['n_links_out'] >= 10)
        
        # Most active
        most_active = merged.nlargest(top_n, 'total_links')
        
        print(f"\n Role distribution (top {top_n} most active):")
        print(f"Critical (high neg out):    {most_active['role_critical'].sum()}")
        print(f"controversial (high neg in):     {most_active['role_controversial'].sum()}")
        print(f"influential (high pos out):  {most_active['role_influential'].sum()}")
        print(f"supportive (high pos in):  {most_active['role_supportive'].sum()}")
        
        # Show examples
        print(f"\n Example roles:")
        
        if most_active['role_critical'].any():
            critics = most_active[most_active['role_critical']].nlargest(5, 'n_negative_links_out')
            print(f"\n  Critical (negative outgoing):")
            for _, row in critics.iterrows():
                print(f"{row['subreddit']}: {row['n_negative_links_out']:.0f} neg out "
                      f"({row['neg_out_ratio']:.1%})")
        
        if most_active['role_controversial'].any():
            targets = most_active[most_active['role_controversial']].nlargest(5, 'n_negative_links_in')
            print(f"\n  Controversial (negative incoming):")
            for _, row in targets.iterrows():
                print(f"{row['subreddit']}: {row['n_negative_links_in']:.0f} neg in "
                      f"({row['neg_in_ratio']:.1%})")
        
        return merged
    
    def get_psychological_profiles(self, top_n: int = 100) -> pd.DataFrame:
        """Create psychological profiles for active subreddits"""
        print(f"Create psychological profiles (top {top_n})")
        
        top_subs = self.df['SOURCE_SUBREDDIT'].value_counts().head(top_n).index
        
        profiles = []
        print(f"\n Profiling {len(top_subs)} subreddits...")
        
        for sub in top_subs:
            sub_data = self.df[self.df['SOURCE_SUBREDDIT'] == sub]
            
            profile = {
                'subreddit': sub,
                'n_posts': len(sub_data),
                'avg_sentiment': sub_data['sentiment_numeric'].mean(),
                'negativity_ratio': sub_data['is_negative'].mean()
            }
            
            # Aggregate LIWC group scores
            for group_name, features in self.liwc_groups.items():
                available = [f for f in features if f in self.liwc_columns]
                if available:
                    profile[f'{group_name.lower()}_score'] = sub_data[available].mean().mean()
            
            # Individual key features
            key_features = ['LIWC_Anger', 'LIWC_Posemo', 'LIWC_We', 'LIWC_They',
                          'LIWC_Certain', 'LIWC_Tentat', 'LIWC_Insight']
            for feature in key_features:
                if feature in self.liwc_columns:
                    profile[feature] = sub_data[feature].mean()
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        print(f" Created profiles with {len(profiles_df.columns)} features")
        
        return profiles_df
    
    def save_results(self) -> Dict[str, pd.DataFrame]:
        """Save all LIWC analysis results"""
        print("SAVING RESULTS")
        
        # 1. Source subreddit features
        print("\Saving source subreddit features...")
        source_features = self.aggregate_by_subreddit(role='source')
        source_features.to_csv(
            self.output_dir / "subreddit_features_source.csv", 
            index=False
        )
        
        # 2. Target subreddit features
        print("Saving target subreddit features...")
        target_features = self.aggregate_by_subreddit(role='target')
        target_features.to_csv(
            self.output_dir / "subreddit_features_target.csv",
            index=False
        )
        
        # 3. Sentiment comparison
        print("Saving sentiment comparison...")
        sentiment_comparison = self.compare_sentiment_groups()
        sentiment_comparison.to_csv(
            self.output_dir / "liwc_sentiment_comparison.csv",
            index=False
        )
        
        # 4. Psychological roles
        print("Saving psychological roles...")
        roles = self.analyze_psychological_roles(top_n=100)
        roles.to_csv(
            self.output_dir / "subreddit_roles.csv",
            index=False
        )
        
        # 5. Psychological profiles
        print("Saving psychological profiles...")
        profiles = self.get_psychological_profiles(top_n=100)
        profiles.to_csv(
            self.output_dir / "psychological_profiles.csv",
            index=False
        )
        
        print("\n All LIWC analysis results saved to:", self.output_dir)
        
        return {
            'source_features': source_features,
            'target_features': target_features,
            'sentiment_comparison': sentiment_comparison,
            'roles': roles,
            'profiles': profiles
        }


def analyze_liwc(hyperlinks_df: pd.DataFrame, liwc_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """Main function to run LIWC analysis"""
    print("LIWC PSYCHOLOGICAL ANALYSIS")
    
    analyzer = LIWCAnalyzer(hyperlinks_df, liwc_columns)
    results = analyzer.save_results()
    
    print(" LIWC ANALYSIS COMPLETE")    
    return results


if __name__ == "__main__":
    pass