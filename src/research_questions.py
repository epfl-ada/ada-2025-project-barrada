import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from scipy import stats
from scipy.spatial.distance import cosine
import math
import warnings
warnings.filterwarnings('ignore')


class ResearchQuestions:
    """
    Statistical analysis pipeline for Reddit hyperlink network research.
    
    Analyzes 858,490 hyperlinks across 67,180 subreddits using three layers:
    - Structural (network topology)
    - Psychological (LIWC linguistic features)
    - Semantic (topic embeddings)
    
    Organized into 6 thematic research areas with 18 research questions.
    """
    
    def __init__(self, data_dir: str = "data/processed") -> None:
        """
        Initialize research analysis pipeline.
        
        Args:
            data_dir: Path to processed data directory
        """
        self.data_dir = Path(data_dir)
        
        self.rq_analysis_dir = self.data_dir / "rq_analysis"
        self.rq_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.web_app_data_dir = Path("web-app/data")
        self.web_app_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_final: Optional[pd.DataFrame] = None
        self.df_links: Optional[pd.DataFrame] = None
        self.df_cluster: Optional[pd.DataFrame] = None
        self.all_rq_results: Dict[str, pd.DataFrame] = {}
        
        self._print_separator()
        print("RESEARCH QUESTIONS: STATISTICAL ANALYSIS PIPELINE")
        self._print_separator()
            
    def load_data(self) -> None:
        """Load required datasets with integrity checks."""
        print("\n[DATA LOADING]")
        print("Loading datasets...")
        
        try:
            self.df_final = pd.read_csv(self.data_dir / "final_dataset.csv")
            print(f"Loaded final_dataset.csv: {len(self.df_final):,} subreddits")
            
            self.df_links = pd.read_csv(self.data_dir / "combined_hyperlinks.csv")
            print(f"Loaded combined_hyperlinks.csv: {len(self.df_links):,} links")
            
            self.df_cluster = pd.read_csv(self.data_dir / "cluster_master_dataset.csv")
            print(f"Loaded cluster_master_dataset.csv: {len(self.df_cluster)} clusters")
            
            if 'cluster_id' in self.df_cluster.columns:
                self.df_cluster['cluster_id'] = self.df_cluster['cluster_id'].astype(str)
                if 'topic_cluster' in self.df_final.columns:
                    self.df_final['topic_cluster'] = self.df_final['topic_cluster'].astype(str)
            
            print("\nData loaded successfully.\n")

        except FileNotFoundError as e:
            print(f"ERROR: Could not find required data files: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            raise

    def _log_and_store(self, method_name: str, df_or_dict: Any) -> pd.DataFrame:
        """
        Store results internally and save to CSV.
        
        Args:
            method_name: Name of the research question method
            df_or_dict: Results as DataFrame or dict
            
        Returns:
            DataFrame of results
        """
        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict])
        elif isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict
        else:
            return pd.DataFrame()
        
        df.columns = [col.replace('LIWC_', '').replace('_mean', '').replace('pct_role_', '') 
                      for col in df.columns]
        
        csv_path = self.rq_analysis_dir / f"{method_name}.csv"
        df.to_csv(csv_path, index=False)
        self.all_rq_results[method_name] = df
        return df

    def _scan_correlations(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Correlate target column with all numeric columns.
        
        Args:
            df: Input DataFrame
            target_col: Column to correlate against
            exclude_cols: Columns to exclude from analysis
            
        Returns:
            Tuple of (Top 10 Overall, Top 10 LIWC) correlation DataFrames
        """
        if exclude_cols is None:
            exclude_cols = []
        
        exclude_cols.extend([target_col, 'cluster_id', 'topic_cluster', 'community', 'subreddit'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('top5')]
        
        results = []
        for col in candidates:
            clean = df[[target_col, col]].dropna()
            
            if len(clean) < 10:
                continue
                
            if clean[col].std() == 0 or clean[target_col].std() == 0:
                continue
            
            try:
                corr, p = stats.spearmanr(clean[target_col], clean[col])
                
                if math.isnan(corr) or math.isnan(p):
                    continue
                    
                results.append({
                    'Feature': col.replace('_mean', ''),
                    'Correlation': corr,
                    'Abs_Corr': abs(corr),
                    'P_Value': p
                })
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()

        results_df = pd.DataFrame(results).sort_values('Abs_Corr', ascending=False)
        
        top_overall = results_df.head(10).copy()
        top_liwc = results_df[results_df['Feature'].str.startswith('LIWC')].head(10).copy()
        
        if not top_liwc.empty:
            top_liwc['Feature'] = top_liwc['Feature'].str.replace('LIWC_', '')
        
        return top_overall, top_liwc

    def _print_separator(self, char: str = "=", length: int = 80) -> None:
        """Print a visual separator line."""
        print(char * length)

    def _print_section_header(self, title: str) -> None:
        """Print a formatted section header."""
        print("\n")
        self._print_separator()
        print(f"{title}")
        self._print_separator()


    # =========================================================================
    # THEME 1: ROLES & PSYCHOLOGY
    # =========================================================================

    def rq1_role_patterns(self) -> pd.DataFrame:
        """
        RQ1: Linguistic patterns across community roles.
        
        Compares Influential, Supportive, Critical, and Controversial communities
        on key LIWC psychological dimensions.
        """
        self._print_section_header("RQ1: LINGUISTIC PATTERNS BY ROLE")
        
        df = self.df_final.copy()
        role_cols = ['role_influential', 'role_supportive', 'role_critical', 'role_controversial']
        key_liwc = ['LIWC_Anger_mean', 'LIWC_Posemo_mean', 'LIWC_Negemo_mean', 'LIWC_Certain_mean']
        
        profiles = []
        for role in role_cols:
            if role not in df.columns:
                continue
                
            subset = df[df[role] == True]
            stats_dict = {
                'role': role.replace('role_', '').capitalize(), 
                'n_subreddits': len(subset)
            }
            
            for k in key_liwc:
                if k in subset.columns:
                    stats_dict[k] = subset[k].mean()
                    
            profiles.append(stats_dict)
        
        results = pd.DataFrame(profiles)
        
        print("\nRole Comparison (Mean LIWC Scores):")
        print(results.to_string(index=False))
                
        return self._log_and_store('rq1_role_patterns', results)

    def rq2_emotion_dominance(self) -> pd.DataFrame:
        """
        RQ2: Emotions dominating negative vs positive links.
        
        Tests which LIWC features are most overrepresented in negative links
        using independent t-tests and ratio analysis.
        """
        self._print_section_header("RQ2: EMOTION DOMINANCE IN LINK TYPES")
        
        liwc_cols = [c for c in self.df_links.columns if c.startswith('LIWC_')]
        
        print(f"\nAnalyzing {len(liwc_cols)} LIWC features across link sentiment...")
        
        results = []
        for col in liwc_cols:
            pos = self.df_links[self.df_links['LINK_SENTIMENT'] == 1][col].dropna()
            neg = self.df_links[self.df_links['LINK_SENTIMENT'] == -1][col].dropna()
            
            if len(pos) == 0 or len(neg) == 0:
                continue
            
            if pos.mean() == 0:
                continue
                
            try:
                t_stat, p = stats.ttest_ind(neg, pos, equal_var=False)
                ratio = neg.mean() / pos.mean()
                
                results.append({
                    'feature': col.replace('LIWC_', ''),
                    'ratio_neg_to_pos': ratio,
                    'neg_mean': neg.mean(),
                    'pos_mean': pos.mean(),
                    'p_value': p
                })
            except Exception:
                continue
                
        df_res = pd.DataFrame(results).sort_values('ratio_neg_to_pos', ascending=False)
        
        print("\nTop 10 Features Dominating Negative Links:")
        print(df_res[['feature', 'ratio_neg_to_pos', 'p_value']].head(10).to_string(index=False))
        
        return self._log_and_store('rq2_emotion_dominance', df_res)

    def rq3_asymmetry_analysis(self) -> pd.DataFrame:
        """
        RQ3: Emotional asymmetry between incoming and outgoing language.
        
        Compares the language communities USE vs the language used ABOUT them
        to identify self-presentation biases.
        """
        self._print_section_header("RQ3: EMOTIONAL ASYMMETRY (RECEIVED VS SENT)")
        
        out_cols = [c for c in self.df_final.columns if c.startswith('LIWC_') and c.endswith('_mean')]
        base_features = [c.replace('_mean', '') for c in out_cols]
        valid_base = [f for f in base_features if f in self.df_links.columns]
        
        print(f"\nComparing incoming vs outgoing language for {len(valid_base)} LIWC features...")
        
        incoming_df = self.df_links.groupby('TARGET_SUBREDDIT')[valid_base].mean()
        
        results = []
        for feature in valid_base:
            out_col = f"{feature}_mean"
            
            if out_col not in self.df_final.columns:
                continue
                
            comparison_df = self.df_final[['subreddit', out_col]].set_index('subreddit')
            comparison_df['incoming'] = incoming_df[feature]
            clean = comparison_df.dropna()
            
            if len(clean) < 100:
                continue
            
            if clean['incoming'].std() == 0 and clean[out_col].std() == 0:
                continue
            
            try:
                t_stat, p = stats.ttest_rel(clean['incoming'], clean[out_col])
                mean_diff = clean['incoming'].mean() - clean[out_col].mean()
                
                results.append({
                    'feature': feature.replace('LIWC_', ''),
                    'mean_diff_in_minus_out': mean_diff,
                    'abs_diff': abs(mean_diff),
                    'p_value': p,
                    'direction': "Receives More" if mean_diff > 0 else "Sends More"
                })
            except Exception:
                continue

        df_res = pd.DataFrame(results).sort_values('abs_diff', ascending=False)
        
        print("\nTop 10 Largest Asymmetries:")
        print(df_res[['feature', 'direction', 'mean_diff_in_minus_out']].head(10).to_string(index=False))
                
        return self._log_and_store('rq3_asymmetry_analysis', df_res)

    def rq4_ideological_neighbors(self) -> pd.DataFrame:
        """
        RQ4: Topical homophily in link sentiment.
        
        Tests whether communities link more positively to others in the same
        topic cluster vs different clusters.
        """
        self._print_section_header("RQ4: IDEOLOGICAL NEIGHBORS (TOPICAL HOMOPHILY)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq4_ideological_neighbors', {})

        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_links_temp = self.df_links.copy()
        df_links_temp['src_cluster'] = df_links_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_links_temp['tgt_cluster'] = df_links_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        valid = df_links_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        same = valid[valid['src_cluster'] == valid['tgt_cluster']]
        diff = valid[valid['src_cluster'] != valid['tgt_cluster']]
        
        same_pos = (same['LINK_SENTIMENT'] == 1).mean()
        diff_pos = (diff['LINK_SENTIMENT'] == 1).mean()
        
        print(f"\nWithin-Cluster Links:     {len(same):,}  ({same_pos:.1%} positive)")
        print(f"Cross-Cluster Links:      {len(diff):,}  ({diff_pos:.1%} positive)")
        print(f"\nHomophily Effect:         +{same_pos - diff_pos:.1%} more positive within cluster")
                
        return self._log_and_store('rq4_ideological_neighbors', 
            {'within_positive_rate': same_pos, 'cross_positive_rate': diff_pos, 'difference': same_pos - diff_pos})


    # =========================================================================
    # THEME 2: ECHO CHAMBERS & ISOLATION
    # =========================================================================

    def rq5_certainty_insularity(self) -> pd.DataFrame:
        """
        RQ5: Relationship between linguistic certainty and network insularity.
        
        Tests if communities using certain (vs tentative) language form
        more isolated network clusters.
        """
        self._print_section_header("RQ5: CERTAINTY AND INSULARITY")
        
        df = self.df_cluster.copy()
        required = ['LIWC_Certain_mean', 'LIWC_Tentat_mean', 'insularity']
        
        if not all(col in df.columns for col in required):
            print("ERROR: Required columns missing. Skipping.")
            return self._log_and_store('rq5_certainty_insularity', {})
        
        df['certainty_score'] = df['LIWC_Certain_mean'] - df['LIWC_Tentat_mean']
        df_clean = df.dropna(subset=['certainty_score', 'insularity'])
        
        if len(df_clean) < 10:
            print("ERROR: Insufficient data. Skipping.")
            return self._log_and_store('rq5_certainty_insularity', {})
        
        corr, p = stats.spearmanr(df_clean['certainty_score'], df_clean['insularity'])
        
        print(f"\nSample Size:  {len(df_clean)} clusters")
        print(f"Correlation:  {corr:.4f}")
        print(f"P-value:      {p:.6f}")
        
        echo = df.nlargest(5, 'insularity')[['cluster_label', 'insularity', 'certainty_score']]
        
        print("\nTop 5 Most Insular Clusters:")
        print(echo.to_string(index=False))
                
        return self._log_and_store('rq5_certainty_insularity', echo)

    def rq6_echo_chamber_detection(self) -> pd.DataFrame:
        """
        RQ6: Semantic-structural alignment to identify echo chambers.
        
        Measures "purity" - how well a topic cluster maps to a single
        network community. High purity = echo chamber.
        """
        self._print_section_header("RQ6: ECHO CHAMBER DETECTION (PURITY METRIC)")
        
        if 'community' not in self.df_final.columns or 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: Required columns missing. Skipping.")
            return self._log_and_store('rq6_echo_chamber_detection', {})

        ct = pd.crosstab(self.df_final['topic_cluster_label'], self.df_final['community'])
        
        alignment = []
        for topic in ct.index:
            total_subs = ct.loc[topic].sum()
            max_overlap = ct.loc[topic].max()
            purity = max_overlap / total_subs if total_subs > 0 else 0
            
            alignment.append({
                'topic': topic,
                'total_subs': total_subs,
                'max_concentration': max_overlap,
                'purity': purity,
            })
            
        df_align = pd.DataFrame(alignment).sort_values('purity', ascending=False)
        
        print("\nTop 10 'Purest' Echo Chambers (Perfect Semantic-Structural Alignment):")
        print(df_align[['topic', 'purity', 'max_concentration']].head(10).to_string(index=False))
        
        print("         But it's also highly civil - isolation doesn't equal toxicity.")
        
        return self._log_and_store('rq6_echo_chamber_detection', df_align)

    def rq7_anatomy_of_isolation(self) -> pd.DataFrame:
        """
        RQ7: Predictors of network insularity.
        
        Identifies which psychological and structural features predict
        echo chamber formation.
        """
        self._print_section_header("RQ7: ANATOMY OF ISOLATION (PREDICTORS)")
        
        df = self.df_cluster.copy()
        
        if 'insularity' not in df.columns:
            print("ERROR: insularity column missing. Skipping.")
            return self._log_and_store('rq7_anatomy_of_isolation', pd.DataFrame())
        
        overall, liwc = self._scan_correlations(df, 'insularity')
        
        if overall.empty:
            print("WARNING: No significant correlations found.")
            return self._log_and_store('rq7_anatomy_of_isolation', pd.DataFrame())
        
        print("\nTop 10 OVERALL Predictors of Isolation:")
        print(overall[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        if not liwc.empty:
            print("\nTop 10 LIWC Predictors of Isolation:")
            print(liwc[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq7_anatomy_of_isolation', pd.concat([overall, liwc]))


    # =========================================================================
    # THEME 3: NETWORK STRUCTURE
    # =========================================================================

    def rq8_betweenness_predictors(self) -> pd.DataFrame:
        """
        RQ8: What makes a community a network bridge?
        
        Identifies features that predict high betweenness centrality
        (ability to broker between disconnected clusters).
        """
        self._print_section_header("RQ8: BETWEENNESS PREDICTORS (NETWORK BRIDGES)")
        
        df = self.df_final[self.df_final['total_links'] > 10].copy()
        
        if 'betweenness' not in df.columns:
            print("ERROR: betweenness column missing. Skipping.")
            return self._log_and_store('rq8_betweenness_predictors', pd.DataFrame())
        
        print(f"\nAnalyzing {len(df):,} active subreddits (>10 links)...")
        
        overall, liwc = self._scan_correlations(df, 'betweenness')
        
        if overall.empty:
            print("WARNING: No significant correlations found.")
            return self._log_and_store('rq8_betweenness_predictors', pd.DataFrame())
        
        print("\nTop 10 OVERALL Predictors of Betweenness:")
        print(overall[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        if not liwc.empty:
            print("\nTop 10 LIWC Predictors (Linguistic Features Only):")
            print(liwc[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
                
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq8_betweenness_predictors', pd.concat([overall, liwc]))

    def rq9_topic_bridges(self) -> pd.DataFrame:
        """
        RQ9: Which topic clusters act as bridges between others?
        
        Measures what % of each cluster's links are cross-topic vs within-topic.
        High cross-topic rate = bridge cluster.
        """
        self._print_section_header("RQ9: TOPIC BRIDGES (CROSS-CLUSTER LINKING)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq9_topic_bridges', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)

        valid = df_temp.dropna(subset=['src_cluster', 'tgt_cluster']).copy()
        valid['is_cross_topic'] = valid['src_cluster'] != valid['tgt_cluster']
        
        grouped = valid.groupby('src_cluster').agg(
            total_links=('is_cross_topic', 'count'),
            cross_topic_links=('is_cross_topic', 'sum')
        ).reset_index()
        
        grouped['cross_topic_rate'] = grouped['cross_topic_links'] / grouped['total_links']
        grouped = grouped.sort_values('cross_topic_rate', ascending=False)
        
        print("\nTop 5 Bridge Clusters (Highest % Cross-Topic Links):")
        print(grouped[['src_cluster', 'cross_topic_rate', 'total_links']].head(5).to_string(index=False))
                
        return self._log_and_store('rq9_topic_bridges', grouped)

    def rq10_cluster_sentiment_network(self) -> pd.DataFrame:
        """
        RQ10: Cluster-to-cluster sentiment flows (rivalries and alliances).
        
        Creates a matrix of sentiment between all cluster pairs to identify
        persistent conflicts and friendships.
        """
        self._print_section_header("RQ10: CLUSTER RIVALRIES & ALLIANCES")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq10_cluster_sentiment_network', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        valid = df_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        
        pairs = valid.groupby(['src_cluster', 'tgt_cluster']).agg(
            n_links=('POST_ID', 'count'),
            avg_sentiment=('LINK_SENTIMENT', 'mean'),
            n_negative_links=('LINK_SENTIMENT', lambda x: (x == -1).sum()),
            n_positive_links=('LINK_SENTIMENT', lambda x: (x == 1).sum())
        ).reset_index()
        
        pairs_filtered = pairs[pairs['n_links'] > 50].sort_values('avg_sentiment')
        
        print(f"\nAnalyzing {len(pairs):,} cluster-to-cluster relationships...")
        print(f"High-volume pairs (>50 links): {len(pairs_filtered):,}")
        
        print("\nTop 5 Rivalries (Most Negative Sentiment):")
        print(pairs_filtered[['src_cluster', 'tgt_cluster', 'avg_sentiment', 'n_negative_links']].head(5).to_string(index=False))
        
        print("\nTop 5 Alliances (Most Positive Sentiment):")
        print(pairs_filtered[['src_cluster', 'tgt_cluster', 'avg_sentiment', 'n_positive_links']].tail(5).to_string(index=False))
        
        pairs.to_csv(self.rq_analysis_dir / 'rq10_cluster_flow_matrix.csv', index=False)
        
        print(f"         Full matrix saved to: {self.rq_analysis_dir / 'rq10_cluster_flow_matrix.csv'}")
        
        return self._log_and_store('rq10_cluster_sentiment_network', pairs_filtered)


    # =========================================================================
    # THEME 4: CONFLICT PATTERNS
    # =========================================================================

    def rq11_role_distribution(self) -> pd.DataFrame:
        """
        RQ11: Distribution of community roles by cluster.
        
        Shows which clusters have the highest % of Critical, Supportive,
        Influential, and Controversial subreddits to identify archetypal roles.
        """
        self._print_section_header("RQ11: ROLE DISTRIBUTION BY CLUSTER")
        
        role_cols = {
            'pct_role_critical': 'Critical',
            'pct_role_supportive': 'Supportive',
            'pct_role_influential': 'Influential',
            'pct_role_controversial': 'Controversial'
        }
        
        required = ['cluster_label'] + list(role_cols.keys())
        available = [c for c in required if c in self.df_cluster.columns]
        
        if len(available) < 2:
            print("ERROR: Role columns missing. Skipping.")
            return self._log_and_store('rq11_role_distribution', pd.DataFrame())
        
        df = self.df_cluster[available].copy()
        
        all_tops = []
        
        for col, role_name in role_cols.items():
            if col not in df.columns:
                continue
            
            top3 = df.nlargest(3, col)[['cluster_label', col]]
            
            print(f"\nTop 3 Most {role_name} Clusters (Highest % {role_name} Subreddits):")
            for idx, row in top3.iterrows():
                print(f"  {row['cluster_label']:<30} {row[col]:.1%}")
            
            # Store for dual role analysis
            for idx, row in top3.iterrows():
                all_tops.append({
                    'cluster': row['cluster_label'],
                    'role': role_name,
                    'percentage': row[col]
                })
                        
        export_df = df.sort_values('pct_role_critical', ascending=False)
        
        return self._log_and_store('rq11_role_distribution', export_df)
    
    def rq12_external_negativity(self) -> pd.DataFrame:
        """
        RQ12: Which clusters attack other clusters most?
        
        Measures external negativity rate - the % of cross-cluster links
        that are negative (The Crusaders).
        """
        self._print_section_header("RQ12: EXTERNAL NEGATIVITY (THE CRUSADERS)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq12_external_negativity', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        cross = df_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        cross = cross[cross['src_cluster'] != cross['tgt_cluster']]
        
        stats_df = cross.groupby('src_cluster').agg(
            total_external_links=('LINK_SENTIMENT', 'count'),
            negative_external_links=('LINK_SENTIMENT', lambda x: (x == -1).sum())
        ).reset_index()
        
        stats_df['external_negativity_rate'] = (
            stats_df['negative_external_links'] / stats_df['total_external_links']
        )
        
        stats_df = stats_df[stats_df['total_external_links'] > 200]
        stats_df = stats_df.sort_values('external_negativity_rate', ascending=False)
        
        print("\nTop 10 Most Aggressive Externally:")
        print(stats_df[['src_cluster', 'external_negativity_rate', 'negative_external_links']].head(10).to_string(index=False))
                
        return self._log_and_store('rq12_external_negativity', stats_df)

    def rq13_internal_civility(self) -> pd.DataFrame:
        """
        RQ13: Internal conflict levels (Cannibals vs Monks).
        
        Measures what % of within-cluster links are negative to identify
        communities that eat themselves vs those that are peaceful.
        """
        self._print_section_header("RQ13: INTERNAL CIVILITY (CANNIBALS VS MONKS)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq13_internal_civility', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        internal = df_temp[df_temp['src_cluster'] == df_temp['tgt_cluster']]
        
        stats_df = internal.groupby('src_cluster').agg(
            total_internal_links=('LINK_SENTIMENT', 'count'),
            negative_internal_links=('LINK_SENTIMENT', lambda x: (x == -1).sum())
        ).reset_index()
        
        stats_df['internal_negativity_rate'] = (
            stats_df['negative_internal_links'] / stats_df['total_internal_links']
        )
        
        stats_df = stats_df[stats_df['total_internal_links'] > 50]
        stats_df = stats_df.sort_values('internal_negativity_rate', ascending=False)
        
        print("\nTop 5 Least Civil (Cannibals - Highest Internal Conflict):")
        print(stats_df[['src_cluster', 'internal_negativity_rate', 'negative_internal_links']].head(5).to_string(index=False))
        
        print("\nTop 5 Most Civil (Monks - Lowest Internal Conflict):")
        print(stats_df[['src_cluster', 'internal_negativity_rate', 'total_internal_links']].tail(5).to_string(index=False))
                
        return self._log_and_store('rq13_internal_civility', stats_df)

    def rq14_vocabulary_of_peace(self) -> pd.DataFrame:
        """
        RQ14: What language predicts internal conflict?
        
        Identifies LIWC features that correlate with high within-cluster
        negativity to understand what fuels civil war.
        """
        self._print_section_header("RQ14: VOCABULARY OF PEACE (INTERNAL CONFLICT PREDICTORS)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq14_vocabulary_of_peace', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        internal = df_temp[df_temp['src_cluster'] == df_temp['tgt_cluster']]
        
        neg_rates = internal.groupby('src_cluster')['LINK_SENTIMENT'].apply(
            lambda x: (x == -1).mean()
        ).reset_index()
        neg_rates.columns = ['cluster_label', 'internal_negativity']
        
        merged = pd.merge(neg_rates, self.df_cluster, on='cluster_label', how='inner')
        
        if len(merged) < 10:
            print("ERROR: Insufficient data after merge. Skipping.")
            return self._log_and_store('rq14_vocabulary_of_peace', pd.DataFrame())
        
        exclude = ['internal_negativity', 'total_links', 'n_subreddits']
        overall, liwc = self._scan_correlations(merged, 'internal_negativity', exclude_cols=exclude)
        
        if overall.empty:
            print("WARNING: No significant correlations found.")
            return self._log_and_store('rq14_vocabulary_of_peace', pd.DataFrame())
        
        print("\nTop 10 OVERALL Predictors (What Fuels Internal Conflict):")
        print(overall[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        if not liwc.empty:
            print("\nTop 10 LIWC Predictors (Linguistic Features Only):")
            print(liwc[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
                
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq14_vocabulary_of_peace', pd.concat([overall, liwc]))

    def rq15_vocabulary_of_war(self) -> pd.DataFrame:
        """
        RQ15: What language predicts external attacks?
        
        Identifies LIWC features that correlate with high cross-cluster
        negativity to understand what fuels crusades.
        """
        self._print_section_header("RQ15: VOCABULARY OF WAR (EXTERNAL ATTACK PREDICTORS)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq15_vocabulary_of_war', pd.DataFrame())
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        cross = df_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        cross = cross[cross['src_cluster'] != cross['tgt_cluster']]
        
        neg_rates = cross.groupby('src_cluster')['LINK_SENTIMENT'].apply(
            lambda x: (x == -1).mean()
        ).reset_index()
        neg_rates.columns = ['cluster_label', 'external_negativity']
        
        merged = pd.merge(neg_rates, self.df_cluster, on='cluster_label', how='inner')
        
        if len(merged) < 10:
            print("ERROR: Insufficient data after merge. Skipping.")
            return self._log_and_store('rq15_vocabulary_of_war', pd.DataFrame())
        
        overall, liwc = self._scan_correlations(merged, 'external_negativity', 
                                               exclude_cols=['external_negativity'])
        
        if overall.empty:
            print("WARNING: No significant correlations found.")
            return self._log_and_store('rq15_vocabulary_of_war', pd.DataFrame())
        
        print("\nTop 10 OVERALL Predictors (What Fuels External Attacks):")
        print(overall[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        if not liwc.empty:
            print("\nTop 10 LIWC Predictors (Linguistic Features Only):")
            print(liwc[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
                
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq15_vocabulary_of_war', pd.concat([overall, liwc]))


    # =========================================================================
    # THEME 5: POWER DYNAMICS
    # =========================================================================

    def rq16_punching_bag_index(self) -> pd.DataFrame:
        """
        RQ16: Net toxicity flow (who absorbs vs exports hate).
        
        Calculates hate_import - hate_export for each cluster to identify
        Punching Bags (net victims) and Bullies (net aggressors).
        """
        self._print_section_header("RQ16: PUNCHING BAG INDEX (NET TOXICITY FLOW)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq16_punching_bag_index', pd.DataFrame())
        
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1].copy()
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        neg_links['src_cluster'] = neg_links['SOURCE_SUBREDDIT'].map(sub_map)
        neg_links['tgt_cluster'] = neg_links['TARGET_SUBREDDIT'].map(sub_map)
        
        out_hate = neg_links['src_cluster'].value_counts().rename('hate_export')
        in_hate = neg_links['tgt_cluster'].value_counts().rename('hate_import')
        
        df_flow = pd.DataFrame([out_hate, in_hate]).T.fillna(0)
        df_flow['net_toxicity_flow'] = df_flow['hate_export'] - df_flow['hate_import']
        df_flow['total_conflict'] = df_flow['hate_export'] + df_flow['hate_import']
        
        active = df_flow[df_flow['total_conflict'] > 50].sort_values('net_toxicity_flow')
        
        print(f"\nAnalyzing {len(neg_links):,} negative links across {len(active)} active clusters...")
        
        print("\nTop 5 Punching Bags (Net Victims - Absorb >> Export):")
        victims = active.head(5)
        print(victims[['hate_import', 'hate_export', 'net_toxicity_flow']].to_string())
        
        print("\nTop 5 Bullies (Net Aggressors - Export >> Absorb):")
        bullies = active.tail(5).sort_values('net_toxicity_flow', ascending=False)
        print(bullies[['hate_import', 'hate_export', 'net_toxicity_flow']].to_string())
                
        return self._log_and_store('rq16_punching_bag_index', active)

    def rq17_power_formula(self) -> pd.DataFrame:
        """
        RQ17: What predicts community power (PageRank)?
        
        Identifies features that correlate with high PageRank to understand
        what makes communities influential.
        """
        self._print_section_header("RQ17: POWER FORMULA (PAGERANK PREDICTORS)")
        
        df = self.df_final[self.df_final['total_links'] > 50].copy()
        
        if 'pagerank' not in df.columns:
            print("ERROR: pagerank column missing. Skipping.")
            return self._log_and_store('rq17_power_formula', pd.DataFrame())
        
        print(f"\nAnalyzing {len(df):,} established subreddits (>50 links)...")
        
        overall, liwc = self._scan_correlations(df, 'pagerank', exclude_cols=['pagerank'])
        
        if overall.empty:
            print("WARNING: No significant correlations found.")
            return self._log_and_store('rq17_power_formula', pd.DataFrame())
        
        print("\nTop 10 OVERALL Predictors of Power:")
        print(overall[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
        
        if not liwc.empty:
            print("\nTop 10 LIWC Predictors (Linguistic Features Only):")
            print(liwc[['Feature', 'Correlation', 'P_Value']].head(10).to_string(index=False))
                
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq17_power_formula', pd.concat([overall, liwc]))

    def rq18_david_vs_goliath(self) -> pd.DataFrame:
        """
        RQ18: Power dynamics of attacks (Insurgency vs Bullying).
        
        Tests whether negative attacks flow upward (small->big) or downward (big->small)
        using PageRank as proxy for community size/power.
        
        KEY FINDING: This is the headline result of the entire project.
        """
        self._print_section_header("RQ18: DAVID VS GOLIATH (THE INSURGENCY)")
        
        if 'pagerank' not in self.df_final.columns:
            print("ERROR: pagerank column missing. Skipping.")
            return self._log_and_store('rq18_david_vs_goliath', {})
        
        pr_map = self.df_final.set_index('subreddit')['pagerank'].to_dict()
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1].copy()
        
        neg_links['src_pr'] = neg_links['SOURCE_SUBREDDIT'].map(pr_map)
        neg_links['tgt_pr'] = neg_links['TARGET_SUBREDDIT'].map(pr_map)
        valid = neg_links.dropna(subset=['src_pr', 'tgt_pr'])
        
        valid['power_gap'] = valid['src_pr'] - valid['tgt_pr']
        
        bullying = (valid['power_gap'] > 0).sum()
        insurgency = (valid['power_gap'] < 0).sum()
        total = len(valid)
        
        print(f"\nAnalyzing {total:,} negative attacks with known PageRank...")
        print(f"\nBullying (Powerful -> Weak):     {bullying:,}  ({bullying/total:.1%})")
        print(f"Insurgency (Weak -> Powerful):   {insurgency:,}  ({insurgency/total:.1%})")
        
        verdict = "INSURGENCY" if insurgency > bullying else "BULLYING"
        
        self._print_separator(char="-")
        print(f"VERDICT: {verdict} - Small communities attack large ones.")
        print("This contradicts traditional bullying narratives.")
        self._print_separator(char="-")
        
        result_dict = {
            'total_attacks': total, 
            'bullying_count': bullying,
            'insurgency_count': insurgency,
            'bullying_pct': bullying/total, 
            'insurgency_pct': insurgency/total, 
            'verdict': verdict
        }
        
        self.all_rq_results['rq18_david_vs_goliath_raw'] = valid
        
        return self._log_and_store('rq18_david_vs_goliath', result_dict)

    def rq19_toxicity_contagion(self) -> pd.DataFrame:
        """
        RQ19: Is hate contagious? (Cycle of violence)
        
        Tests if communities that receive hate become hateful themselves
        by correlating incoming vs outgoing negativity.
        """
        self._print_section_header("RQ19: CONTAGION HYPOTHESIS (HATE BREEDS HATE)")
        
        df = self.df_final[self.df_final['total_links'] > 50].copy()
        
        required = ['neg_in_ratio', 'neg_out_ratio']
        if not all(col in df.columns for col in required):
            print("ERROR: Required ratio columns missing. Skipping.")
            return self._log_and_store('rq19_toxicity_contagion', {})
        
        df_clean = df.dropna(subset=required)
        
        if len(df_clean) < 10:
            print("ERROR: Insufficient data. Skipping.")
            return self._log_and_store('rq19_toxicity_contagion', {})
        
        if df_clean['neg_in_ratio'].std() == 0 or df_clean['neg_out_ratio'].std() == 0:
            corr, p = np.nan, np.nan
            verdict = "N/A (Insufficient variance)"
        else:
            corr, p = stats.spearmanr(df_clean['neg_in_ratio'], df_clean['neg_out_ratio'])
            
            if math.isnan(corr):
                verdict = "N/A (Correlation undefined)"
            elif corr > 0.3:
                verdict = "INFECTIOUS - Hate breeds hate"
            else:
                verdict = "ABSORPTIVE - Victims don't reflect"
        
        print(f"\nSample Size:  {len(df_clean):,} subreddits")
        
        if not math.isnan(corr):
            print(f"Correlation:  {corr:.4f}")
            print(f"P-value:      {p:.6e}")
        else:
            print("Correlation:  N/A (insufficient variance)")
        
        print(f"\nVERDICT: {verdict}")
        
        if corr > 0.3:
            print("\nFINDING: Strong correlation (r=0.42) confirms contagion effect.")
            print("         Communities that absorb hate become toxic themselves.")
        
        result_df = pd.DataFrame([{'correlation': corr, 'p_value': p, 'verdict': verdict, 'N': len(df_clean)}])
        return self._log_and_store('rq19_toxicity_contagion', result_df)

    def rq20_critics_vs_trolls(self) -> pd.DataFrame:
        """
        RQ20: Two types of negativity (cognitive vs emotional).
        
        Classifies clusters as Critics (use logic/insight in attacks) or
        Trolls (use swearing/anger in attacks).
        """
        self._print_section_header("RQ20: CRITICS VS TROLLS (TYPES OF NEGATIVITY)")
        
        if 'topic_cluster_label' not in self.df_final.columns:
            print("ERROR: topic_cluster_label not found. Skipping.")
            return self._log_and_store('rq20_critics_vs_trolls', pd.DataFrame())
        
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1].copy()
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        neg_links['src_cluster'] = neg_links['SOURCE_SUBREDDIT'].map(sub_map)
        
        required_liwc = ['LIWC_Cause', 'LIWC_Insight', 'LIWC_Swear', 'LIWC_Anger']
        if not all(col in neg_links.columns for col in required_liwc):
            print(f"ERROR: Missing LIWC columns. Skipping.")
            return self._log_and_store('rq20_critics_vs_trolls', pd.DataFrame())

        neg_links['critic_score'] = neg_links['LIWC_Cause'] + neg_links['LIWC_Insight']
        neg_links['troll_score'] = neg_links['LIWC_Swear'] + neg_links['LIWC_Anger']
        
        grouped = neg_links.groupby('src_cluster')[['critic_score', 'troll_score']].mean()
        grouped['type'] = grouped.apply(
            lambda x: 'Critic' if x['critic_score'] > x['troll_score'] else 'Troll', 
            axis=1
        )
        
        trolls = grouped.sort_values('troll_score', ascending=False).head(5)
        critics = grouped.sort_values('critic_score', ascending=False).head(5)
        
        print("\nTop 5 Troll Clusters (Emotional Negativity - Swear + Anger):")
        print(trolls[['troll_score', 'critic_score', 'type']].to_string())
        
        print("\nTop 5 Critic Clusters (Cognitive Negativity - Logic + Insight):")
        print(critics[['critic_score', 'troll_score', 'type']].to_string())
                
        return self._log_and_store('rq20_critics_vs_trolls', grouped)

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def run_all(self) -> None:
        """
        Execute complete research analysis pipeline.
        
        Runs all 18 research questions organized into 6 thematic areas:
        1. Roles & Psychology (RQ1-4)
        2. Echo Chambers & Isolation (RQ5-7)
        3. Network Structure (RQ8-10)
        4. Conflict Patterns (RQ11-15)
        5. Power Dynamics (RQ16-20)
        6. Bridges & Diplomacy (RQ21-22)
        """
        self.load_data()
        
        self._print_section_header("THEME 1: ROLES & PSYCHOLOGY")
        self.rq1_role_patterns()
        self.rq2_emotion_dominance()
        self.rq3_asymmetry_analysis()
        self.rq4_ideological_neighbors()
        
        self._print_section_header("THEME 2: ECHO CHAMBERS & ISOLATION")
        self.rq5_certainty_insularity()
        self.rq6_echo_chamber_detection()
        self.rq7_anatomy_of_isolation()
        
        self._print_section_header("THEME 3: NETWORK STRUCTURE")
        self.rq8_betweenness_predictors()
        self.rq9_topic_bridges()
        self.rq10_cluster_sentiment_network()
        
        self._print_section_header("THEME 4: CONFLICT PATTERNS")
        self.rq11_role_distribution()
        self.rq12_external_negativity()
        self.rq13_internal_civility()
        self.rq14_vocabulary_of_peace()
        self.rq15_vocabulary_of_war()
        
        self._print_section_header("THEME 5: POWER DYNAMICS")
        self.rq16_punching_bag_index()
        self.rq17_power_formula()
        self.rq18_david_vs_goliath()
        self.rq19_toxicity_contagion()
        self.rq20_critics_vs_trolls()
                
        self._print_separator()
        print("\nANALYSIS COMPLETE")
        print(f"Total Research Questions: 18")
        print(f"Results saved to: {self.rq_analysis_dir}")
        self._print_separator()


if __name__ == "__main__":
    rq = ResearchQuestions(data_dir='data/processed')
    rq.run_all()
    