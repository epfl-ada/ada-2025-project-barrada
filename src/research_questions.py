import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy import stats
import math

class ResearchQuestions:
    """
    Final Statistical Pipeline for P3, optimized for web deployment.
    
    Features:
    - Runs all Research Questions (RQs) and logs results to CSV.
    - Exports key metrics to JSON format.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        
        # Directories for analytical log files (CSV)
        self.rq_analysis_dir = self.data_dir / "rq_analysis"
        self.rq_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories for web app JSON files
        self.web_app_data_dir = Path("web-app/data")
        self.web_app_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_final: Optional[pd.DataFrame] = None
        self.df_links: Optional[pd.DataFrame] = None
        self.df_cluster: Optional[pd.DataFrame] = None
        self.all_rq_results: Dict[str, pd.DataFrame] = {}
            
    def load_data(self) -> None:
        """Load required datasets with integrity checks."""
        print("Loading datasets...")
        try:
            self.df_final = pd.read_csv(self.data_dir / "final_dataset.csv")
            self.df_links = pd.read_csv(self.data_dir / "combined_hyperlinks.csv")
            self.df_cluster = pd.read_csv(self.data_dir / "cluster_master_dataset.csv")
            
            # Ensure topic_cluster_label is string for merging
            if 'cluster_id' in self.df_cluster.columns:
                 self.df_cluster['cluster_id'] = self.df_cluster['cluster_id'].astype(str)
                 if 'topic_cluster' in self.df_final.columns:
                     self.df_final['topic_cluster'] = self.df_final['topic_cluster'].astype(str)

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    def _log_and_store(self, method_name: str, df_or_dict) -> pd.DataFrame:
        """Helper to store results internally and save to CSV."""
        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict])
        elif isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict
        else:
            return pd.DataFrame()
            
        df.columns = [col.replace('LIWC_', '').replace('_mean', '').replace('pct_role_', '') for col in df.columns]

        # Log to CSV
        csv_path = self.rq_analysis_dir / f"{method_name}.csv"
        df.to_csv(csv_path, index=False)
        self.all_rq_results[method_name] = df
        return df

    def _scan_correlations(self, df: pd.DataFrame, target_col: str, 
                          exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper: Correlates target_col with ALL other numeric columns.
        Returns two DataFrames: (Top 10 Overall, Top 10 LIWC).
        """
        if exclude_cols is None:
            exclude_cols = []
        
        exclude_cols.extend([target_col, 'cluster_id', 'topic_cluster', 'community'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('top5')]
        
        res = []
        for col in candidates:
            clean = df[[target_col, col]].dropna()
            
            if len(clean) > 10:
                if clean[col].std() == 0 or clean[target_col].std() == 0:
                    continue
                    
                corr, p = stats.spearmanr(clean[target_col], clean[col])
                
                if math.isnan(corr):
                    continue
                    
                res.append({
                    'Feature': col.replace('_mean', ''),
                    'Correlation': corr,
                    'Abs_Corr': abs(corr),
                    'P_Value': p
                })
        
        if not res:
            return pd.DataFrame(), pd.DataFrame()

        results_df = pd.DataFrame(res).sort_values('Abs_Corr', ascending=False)
        
        # Split into Overall and LIWC
        top_overall = results_df.head(10)
        top_liwc = results_df[results_df['Feature'].str.startswith('LIWC')].head(10)
        
        top_liwc = top_liwc.copy()
        top_liwc['Feature'] = top_liwc['Feature'].str.replace('LIWC_', '')
        
        return top_overall, top_liwc

    # =========================================================================
    # THEME 1: NETWORK STRUCTURE & ROLES
    # =========================================================================

    def rq1a_analytical_bridges(self) -> pd.DataFrame:
        print("RQ1a: Analytical Communities as Bridges")
        df = self.df_final.copy()
        required_cols = ['LIWC_Insight_mean', 'LIWC_Certain_mean', 'LIWC_CogMech_mean', 
                        'LIWC_Tentat_mean', 'betweenness']
        df_clean = df.dropna(subset=required_cols).copy()
        
        df_clean['analytical_score'] = (
            df_clean['LIWC_Insight_mean'] + 
            df_clean['LIWC_Certain_mean'] + 
            df_clean['LIWC_CogMech_mean'] - 
            df_clean['LIWC_Tentat_mean']
        ) / 4
        
        correlation, p_value = stats.spearmanr(df_clean['analytical_score'], df_clean['betweenness'])
        
        print(f"  N={len(df_clean):,}, Correlation={correlation:.4f}, p={p_value:.6f}")
        return self._log_and_store('rq1a_analytical_bridges', 
            {'test': 'spearman', 'correlation': correlation, 'p_value': p_value, 'N': len(df_clean)})
    
    def rq1b_role_patterns(self) -> pd.DataFrame:
        print("RQ1b: Linguistic Patterns by Role")
        df = self.df_final.copy()
        role_cols = ['role_influential', 'role_supportive', 'role_critical', 'role_controversial']
        
        profiles = []
        key_liwc = ['LIWC_Anger_mean', 'LIWC_Posemo_mean', 'LIWC_Negemo_mean', 'LIWC_Certain_mean']
        
        for role in role_cols:
            if role in df.columns:
                subset = df[df[role] == True]
                stats_dict = {'role': role.replace('role_', '').capitalize(), 'n_subreddits': len(subset)}
                for k in key_liwc:
                    if k in subset.columns:
                        stats_dict[k] = subset[k].mean()
                profiles.append(stats_dict)
        
        results = pd.DataFrame(profiles)
        print(results.to_string(index=False))
        return self._log_and_store('rq1b_role_patterns', results)

    # =========================================================================
    # THEME 2: EMOTIONAL GEOGRAPHY
    # =========================================================================
    
    def rq2a_emotion_dominance(self) -> pd.DataFrame:
        print("RQ2a: Emotion Dominance in Link Types")
        
        # Test ALL LIWC features in links
        liwc_cols = [c for c in self.df_links.columns if c.startswith('LIWC_')]
        
        results = []
        for col in liwc_cols:
            pos = self.df_links[self.df_links['LINK_SENTIMENT'] == 1][col].dropna()
            neg = self.df_links[self.df_links['LINK_SENTIMENT'] == -1][col].dropna()
            
            if len(pos) > 0 and len(neg) > 0:
                t_stat, p = stats.ttest_ind(neg, pos, equal_var=False)
                ratio = neg.mean() / pos.mean() if pos.mean() > 0 else 0
                
                results.append({
                    'feature': col.replace('LIWC_', ''),
                    'ratio_neg_to_pos': ratio,
                    'p_value': p
                })
                
        df_res = pd.DataFrame(results).sort_values('ratio_neg_to_pos', ascending=False)
        print("Top 10 Features Dominating Negative Links (highest ratio):")
        print(df_res[['feature', 'ratio_neg_to_pos', 'p_value']].head(10).to_string(index=False))
        return self._log_and_store('rq2a_emotion_dominance', df_res)

    def rq2b_asymmetry_analysis(self) -> pd.DataFrame:
        print("RQ2b: Emotional Asymmetry (Incoming vs Outgoing)")
                
        out_cols = [c for c in self.df_final.columns if c.startswith('LIWC_') and c.endswith('_mean')]
        base_features = [c.replace('_mean', '') for c in out_cols]
        valid_base = [f for f in base_features if f in self.df_links.columns]
        
        incoming_df = self.df_links.groupby('TARGET_SUBREDDIT')[valid_base].mean()
        
        results = []
        for feature in valid_base:
            out_col = f"{feature}_mean"
            comparison_df = self.df_final[['subreddit', out_col]].set_index('subreddit')
            comparison_df['incoming'] = incoming_df[feature]
            clean = comparison_df.dropna()
            
            if len(clean) > 100:
                # Skip if no variance for t-test
                if clean['incoming'].std() == 0 and clean[out_col].std() == 0:
                     continue
                     
                t_stat, p = stats.ttest_rel(clean['incoming'], clean[out_col])
                mean_diff = clean['incoming'].mean() - clean[out_col].mean()
                
                results.append({
                    'feature': feature.replace('LIWC_', ''),
                    'mean_diff_in_minus_out': mean_diff,
                    'abs_diff': abs(mean_diff),
                    'p_value': p,
                    'direction': "Receives More" if mean_diff > 0 else "Sends More"
                })

        df_res = pd.DataFrame(results).sort_values('abs_diff', ascending=False)
        print("Top 10 Largest Asymmetries (Language Received vs Sent):")
        print(df_res[['feature', 'direction', 'mean_diff_in_minus_out']].head(10).to_string(index=False))
        return self._log_and_store('rq2b_asymmetry_analysis', df_res)

    # =========================================================================
    # THEME 3: SOCIAL DYNAMICS
    # =========================================================================

    def rq3a_similarity_sentiment(self) -> pd.DataFrame:
        print("RQ3a: Psychological Similarity & Sentiment")
        
        sample = self.df_links.sample(min(10000, len(self.df_links)), random_state=42)
        pca_cols = [c for c in self.df_final.columns if c.startswith('pca_')]
        vectors = self.df_final.set_index('subreddit')[pca_cols].to_dict('index')
        
        sims, sents = [], []
        for _, row in sample.iterrows():
            s, t = row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT']
            if s in vectors and t in vectors:
                v1 = list(vectors[s].values())
                v2 = list(vectors[t].values())
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                sim = dot / norm if norm > 0 else 0
                sims.append(sim)
                sents.append(row['LINK_SENTIMENT'])
        
        df_sim = pd.DataFrame({'sim': sims, 'sent': sents})
        
        try:
            df_sim['quartile'] = pd.qcut(df_sim['sim'], 4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
        except ValueError:
            df_sim['quartile'] = pd.cut(df_sim['sim'], bins=np.linspace(df_sim['sim'].min(), df_sim['sim'].max(), 5), include_lowest=True)

        
        stats_df = df_sim.groupby('quartile', observed=True)['sent'].apply(lambda x: (x==1).mean()).reset_index(name='positive_rate')
        
        low_sim_rate = stats_df.iloc[0]['positive_rate'] if not stats_df.empty else np.nan
        high_sim_rate = stats_df.iloc[-1]['positive_rate'] if not stats_df.empty else np.nan

        print(f"  Positive Link Rate - Low Sim: {low_sim_rate:.1%}, High Sim: {high_sim_rate:.1%}")
        return self._log_and_store('rq3a_similarity_sentiment', stats_df)

    def rq3b_ideological_neighbors(self) -> pd.DataFrame:
        print("RQ3b: Topical Homophily (Same vs Different Cluster)")
        if 'topic_cluster_label' not in self.df_final.columns: return self._log_and_store('rq3b_ideological_neighbors', {})

        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_links_temp = self.df_links.copy()
        df_links_temp['src_cluster'] = df_links_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_links_temp['tgt_cluster'] = df_links_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        valid = df_links_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        same = valid[valid['src_cluster'] == valid['tgt_cluster']]
        diff = valid[valid['src_cluster'] != valid['tgt_cluster']]
        
        same_pos = (same['LINK_SENTIMENT'] == 1).mean()
        diff_pos = (diff['LINK_SENTIMENT'] == 1).mean()
        
        print(f"  Within-Cluster Positive Rate: {same_pos:.1%}")
        print(f"  Cross-Cluster Positive Rate:  {diff_pos:.1%}")
        print(f"  Result: Neighbors are {same_pos - diff_pos:+.1%} more positive.")
        return self._log_and_store('rq3b_ideological_neighbors', 
            {'within_positive_rate': same_pos, 'cross_positive_rate': diff_pos})

    # =========================================================================
    # THEME 4: ECHO CHAMBERS & POLARIZATION
    # =========================================================================

    def rq4a_certainty_insularity(self) -> pd.DataFrame:
        print("RQ4a: Certainty and Insularity")
        df = self.df_cluster.copy()
        required = ['LIWC_Certain_mean', 'LIWC_Tentat_mean', 'insularity']
        
        if not all(col in df.columns for col in required): return self._log_and_store('rq4a_certainty_insularity', {})
            
        df['certainty_score'] = df['LIWC_Certain_mean'] - df['LIWC_Tentat_mean']
        corr, p = stats.spearmanr(df['certainty_score'], df['insularity'])
        
        print(f"  N={len(df)}, Correlation={corr:.4f}, p={p:.6f}")
        
        echo = df.nlargest(5, 'insularity')[['cluster_label', 'insularity', 'certainty_score']]
        print("\nTop 5 Most Insular Clusters:")
        print(echo.to_string(index=False))
        return self._log_and_store('rq4a_certainty_insularity', echo)

    def rq4b_semantic_structural_interaction(self) -> pd.DataFrame:
        print("RQ4b: Echo Chamber Detection (Semantic vs Structural Alignment)")
        if 'community' not in self.df_final.columns: return self._log_and_store('rq4b_semantic_structural_interaction', {})

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
        print("Top 10 Most 'Pure' Echo Chambers (Perfect Alignment):")
        print(df_align[['topic', 'purity', 'max_concentration']].head(10).to_string(index=False))
        return self._log_and_store('rq4b_semantic_structural_interaction', df_align)
    
    # =========================================================================
    # THEME 5: EXTENDED ANALYSIS
    # =========================================================================

    def rq5_topic_bridges(self) -> pd.DataFrame:
        print("RQ5: Topic Clusters as Bridges")
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)

        valid = df_temp.dropna(subset=['src_cluster', 'tgt_cluster']).copy()
        valid['is_cross_topic'] = valid['src_cluster'] != valid['tgt_cluster']
        
        grouped = valid.groupby('src_cluster')['is_cross_topic'].mean().reset_index(name='cross_topic_rate')
        grouped = grouped.sort_values('cross_topic_rate', ascending=False)
        
        print("Top 5 Bridging Clusters (Highest % of cross-topic links):")
        print(grouped.head(5).to_string(index=False))
        return self._log_and_store('rq5_topic_bridges', grouped)

    def rq6_betweenness_predictors(self) -> pd.DataFrame:
        print("RQ6: Predictors of Network Influence (Betweenness)")
        df = self.df_final[self.df_final['total_links'] > 10].copy()
        overall, liwc = self._scan_correlations(df, 'betweenness')
        
        print("Top 10 OVERALL Predictors:")
        print(overall[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        print("\nTop 10 LIWC Predictors (Linguistic only):")
        print(liwc[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        
        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq6_betweenness_predictors', pd.concat([overall, liwc]))

    def rq7_cluster_sentiment_network(self) -> pd.DataFrame:
        print("RQ7: Cluster-to-Cluster Sentiment (Rivalries & Alliances)")
        
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
        
        pairs_filtered = pairs[pairs['n_links'] > self.LINK_VOLUME_THRESHOLD].sort_values('avg_sentiment')
        
        print("Top 5 Rivalries (Most Negative Links):")
        print(pairs_filtered.head(5).to_string(index=False))
        print("\nTop 5 Alliances (Most Positive Links):")
        print(pairs_filtered.tail(5).to_string(index=False))
        
        pairs.to_csv(self.rq_analysis_dir / 'rq7_cluster_flow_matrix.csv', index=False)
        return self._log_and_store('rq7_cluster_sentiment_network', pairs_filtered)

    def rq8_role_distribution(self) -> pd.DataFrame:
        print("RQ8: Role Distribution (Cluster Composition)")
        cols = ['cluster_label', 'pct_role_critical', 'pct_role_supportive']
        available = [c for c in cols if c in self.df_cluster.columns]
        
        if not available: return self._log_and_store('rq8_role_distribution', {})
            
        df = self.df_cluster.sort_values('pct_role_critical', ascending=False)
        print("Top 5 Clusters by % Critical Subreddits (The Aggressors):")
        print(df[available].head(5).to_string(index=False))
        return self._log_and_store('rq8_role_distribution', df[['cluster_label', 'pct_role_critical', 'pct_role_supportive']])

    def rq9_external_negativity(self) -> pd.DataFrame:
        print("RQ9: External Negativity Ranking (The 'Crusaders')")
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        cross = df_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        cross = cross[cross['src_cluster'] != cross['tgt_cluster']]
        
        stats_series = cross.groupby('src_cluster')['LINK_SENTIMENT'].apply(lambda x: (x == -1).mean())
        df_res = stats_series.sort_values(ascending=False).reset_index()
        df_res.columns = ['cluster', 'external_negativity_rate']
        
        counts = cross['src_cluster'].value_counts()
        df_res = df_res[df_res['cluster'].map(counts) > 200]
        
        print("Top 10 Most Aggressive Externally:")
        print(df_res.head(10).to_string(index=False))
        return self._log_and_store('rq9_external_negativity', df_res)

    def rq10_internal_civility(self) -> pd.DataFrame:
        print("RQ10: Internal Civility Ranking (The 'Cannibals' vs 'Monks')")
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        internal = df_temp[df_temp['src_cluster'] == df_temp['tgt_cluster']]
        
        stats_series = internal.groupby('src_cluster')['LINK_SENTIMENT'].apply(lambda x: (x == -1).mean())
        df_res = stats_series.sort_values(ascending=False).reset_index()
        df_res.columns = ['cluster', 'internal_negativity_rate']
        
        counts = internal['src_cluster'].value_counts()
        df_res = df_res[df_res['cluster'].map(counts) > 50]
        
        print("Top 5 Least Civil (Highest Internal Conflict):")
        print(df_res.head(5).to_string(index=False))
        print("\nTop 5 Most Civil (Lowest Internal Conflict):")
        print(df_res.tail(5).to_string(index=False))
        return self._log_and_store('rq10_internal_civility', df_res)

    def rq11_vocabulary_of_peace(self) -> pd.DataFrame:
        print("RQ11: The Vocabulary of Peace (Predictors of Internal Conflict)")
        
        # Calculate Internal Negativity
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        internal = df_temp[df_temp['src_cluster'] == df_temp['tgt_cluster']]
        
        neg_rates = internal.groupby('src_cluster')['LINK_SENTIMENT'].apply(lambda x: (x == -1).mean()).reset_index()
        neg_rates.columns = ['cluster_label', 'internal_negativity']
        
        merged = pd.merge(neg_rates, self.df_cluster, on='cluster_label')
        
        exclude = ['internal_negativity', 'negative_ratio', 'total_links', 'n_subreddits']
        overall, liwc = self._scan_correlations(merged, 'internal_negativity', exclude_cols=exclude)
        
        print("Top 10 OVERALL Predictors (Fuel Internal Conflict):")
        print(overall[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        print("\nTop 10 LIWC Predictors (Linguistic):")
        print(liwc[['Feature', 'Correlation', 'P_Value']].to_string(index=False))

        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq11_vocabulary_of_peace', pd.concat([overall, liwc]))

    def rq12_vocabulary_of_war(self) -> pd.DataFrame:
        print("RQ12: The Vocabulary of War (Predictors of External Attacks)")
        
        # Calculate External Negativity
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        df_temp = self.df_links.copy()
        df_temp['src_cluster'] = df_temp['SOURCE_SUBREDDIT'].map(sub_map)
        df_temp['tgt_cluster'] = df_temp['TARGET_SUBREDDIT'].map(sub_map)
        
        cross = df_temp.dropna(subset=['src_cluster', 'tgt_cluster'])
        cross = cross[cross['src_cluster'] != cross['tgt_cluster']]
        
        neg_rates = cross.groupby('src_cluster')['LINK_SENTIMENT'].apply(lambda x: (x == -1).mean()).reset_index()
        neg_rates.columns = ['cluster_label', 'external_negativity']
        
        merged = pd.merge(neg_rates, self.df_cluster, on='cluster_label')
        
        overall, liwc = self._scan_correlations(merged, 'external_negativity', exclude_cols=['external_negativity'])
        
        print("Top 10 OVERALL Predictors (Fuel External Attacks):")
        print(overall[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        print("\nTop 10 LIWC Predictors (Linguistic):")
        print(liwc[['Feature', 'Correlation', 'P_Value']].to_string(index=False))

        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq12_vocabulary_of_war', pd.concat([overall, liwc]))

    def rq13_anatomy_of_isolation(self) -> pd.DataFrame:
        print("RQ13: The Anatomy of Isolation (What predicts Echo Chamber formation?)")
        
        df = self.df_cluster.copy()
        overall, liwc = self._scan_correlations(df, 'insularity')
        
        print("Top 10 OVERALL Predictors of Isolation:")
        print(overall[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        print("\nTop 10 LIWC Predictors of Isolation:")
        print(liwc[['Feature', 'Correlation', 'P_Value']].to_string(index=False))

        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq13_anatomy_of_isolation', pd.concat([overall, liwc]))
    
    # =========================================================================
    # THEME 6: POWER DYNAMICS
    # =========================================================================
    
    def rq14_punching_bag_index(self) -> pd.DataFrame:
        print("RQ14: The Punching Bag Index")
        
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1]
        
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        neg_links = neg_links.copy()
        neg_links['src_cluster'] = neg_links['SOURCE_SUBREDDIT'].map(sub_map)
        neg_links['tgt_cluster'] = neg_links['TARGET_SUBREDDIT'].map(sub_map)
        
        out_hate = neg_links['src_cluster'].value_counts().rename('hate_export')
        in_hate = neg_links['tgt_cluster'].value_counts().rename('hate_import')
        
        df_flow = pd.DataFrame([out_hate, in_hate]).T.fillna(0)
        df_flow['net_toxicity_flow'] = df_flow['hate_export'] - df_flow['hate_import']
        df_flow['total_conflict'] = df_flow['hate_export'] + df_flow['hate_import']
        
        active = df_flow[df_flow['total_conflict'] > 50].sort_values('net_toxicity_flow')
        
        print("\nTop 5 'Punching Bags' (Absorb Hate, Don't Give It):")
        print(active[['hate_import', 'hate_export', 'net_toxicity_flow']].head(5).to_string())
        
        print("\nTop 5 'Bullies' (Export Hate, Don't Take It):")
        print(active[['hate_import', 'hate_export', 'net_toxicity_flow']].tail(5).sort_values('net_toxicity_flow', ascending=False).to_string())
        return self._log_and_store('rq14_punching_bag_index', active)

    def rq15_mainstream_curse(self) -> pd.DataFrame:
        print("RQ15: The Mainstream Curse (Correlation: Power vs Toxicity)")
        
        df = self.df_final[self.df_final['total_links'] > 50].copy()
        if 'neg_out_ratio' not in df.columns:
             df['neg_out_ratio'] = df['n_negative_links_out'] / df['total_links']

        power_metrics = ['pagerank', 'in_degree', 'total_links']
        tox_metrics = ['neg_out_ratio', 'LIWC_Anger_mean', 'LIWC_Swear_mean']
        
        res = []
        for p_metric in power_metrics:
            for t_metric in tox_metrics:
                if p_metric in df.columns and t_metric in df.columns:
                    if df[p_metric].std() > 0 and df[t_metric].std() > 0:
                        corr, p = stats.spearmanr(df[p_metric], df[t_metric])
                        
                        if not math.isnan(corr):
                            res.append({
                                'Power_Metric': p_metric,
                                'Toxicity_Metric': t_metric,
                                'Correlation': corr,
                                'P_Value': p,
                                'Verdict': "Power breeds Toxicity" if corr > 0.1 else "No Curse"
                            })
                    
        df_res = pd.DataFrame(res)
        print(df_res.to_string(index=False))
        return self._log_and_store('rq15_mainstream_curse', df_res)

    def rq16_power_formula(self) -> pd.DataFrame:
        print("RQ16: The Power Formula (What predicts PageRank?)")
        
        df = self.df_final[self.df_final['total_links'] > 50].copy()
        overall, liwc = self._scan_correlations(df, 'pagerank', exclude_cols=['pagerank'])
        
        print("Top 10 OVERALL Predictors of Power:")
        print(overall[['Feature', 'Correlation', 'P_Value']].to_string(index=False))
        print("\nTop 10 LIWC Predictors of Power:")
        print(liwc[['Feature', 'Correlation', 'P_Value']].to_string(index=False))

        overall['type'] = 'overall'
        liwc['type'] = 'liwc'
        return self._log_and_store('rq16_power_formula', pd.concat([overall, liwc]))

    def rq17_david_vs_goliath(self) -> Dict:
        print("RQ17: David vs. Goliath (Power Dynamics of Attacks)")
        
        pr_map = self.df_final.set_index('subreddit')['pagerank'].to_dict()
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1].copy()
        
        neg_links['src_pr'] = neg_links['SOURCE_SUBREDDIT'].map(pr_map)
        neg_links['tgt_pr'] = neg_links['TARGET_SUBREDDIT'].map(pr_map)
        valid = neg_links.dropna(subset=['src_pr', 'tgt_pr'])
        
        valid['power_gap'] = valid['src_pr'] - valid['tgt_pr']
        bullying = (valid['power_gap'] > 0).sum()
        insurgency = (valid['power_gap'] < 0).sum()
        total = len(valid)
        
        print(f"  Analyzed {total:,} negative attacks.")
        print(f"  Bullying (Big -> Small):   {bullying:,} ({bullying/total:.1%})")
        print(f"  Insurgency (Small -> Big): {insurgency:,} ({insurgency/total:.1%})")
        
        verdict = "Insurgency (Punching Up)" if insurgency > bullying else "Bullying (Punching Down)"
        print(f"  VERDICT: {verdict}.")
        
        result_dict = {'total_attacks': total, 'bullying_pct': bullying/total, 'insurgency_pct': insurgency/total, 'verdict': verdict}
        self.all_rq_results['rq17_david_vs_goliath_raw'] = valid # Store raw for animation JSON
        return self._log_and_store('rq17_david_vs_goliath', result_dict)
    
    def rq18_toxicity_contagion(self) -> pd.DataFrame:
        print("RQ18: The Contagion Hypothesis (Cycle of Violence)")
        
        df = self.df_final[self.df_final['total_links'] > 50].copy()
        
        if df['neg_in_ratio'].std() == 0 or df['neg_out_ratio'].std() == 0:
            corr, p = np.nan, np.nan
        else:
            corr, p = stats.spearmanr(df['neg_in_ratio'], df['neg_out_ratio'])
        
        print(f"  N={len(df):,}")
        
        if not math.isnan(corr):
            print(f"  Correlation between In-Hate and Out-Hate: {corr:.4f} (p={p:.6e})")
            verdict = "Infectious (Hate breeds Hate)" if corr > 0.3 else "Absorptive (Victims don't reflect)"
        else:
            print("  Correlation is NaN (Insufficient variance).")
            verdict = "N/A"
        
        print(f"  VERDICT: {verdict}")
        result_df = pd.DataFrame([{'correlation': corr, 'p_value': p, 'verdict': verdict}])
        return self._log_and_store('rq18_toxicity_contagion', result_df)

    def rq19_critics_vs_trolls(self) -> pd.DataFrame:
        print("RQ19: Critics vs. Trolls (Types of Negativity by Cluster)")
        
        neg_links = self.df_links[self.df_links['LINK_SENTIMENT'] == -1].copy()
        sub_map = self.df_final.set_index('subreddit')['topic_cluster_label'].to_dict()
        neg_links['src_cluster'] = neg_links['SOURCE_SUBREDDIT'].map(sub_map)
                
        required_liwc = ['LIWC_Cause', 'LIWC_Insight', 'LIWC_Swear', 'LIWC_Anger']
        if not all(col in neg_links.columns for col in required_liwc):
            print(f"  Missing LIWC columns: {required_liwc}. Skipping.")
            return self._log_and_store('rq19_critics_vs_trolls', {})

        neg_links['critic_score'] = neg_links['LIWC_Cause'] + neg_links['LIWC_Insight']
        neg_links['troll_score'] = neg_links['LIWC_Swear'] + neg_links['LIWC_Anger']
        
        grouped = neg_links.groupby('src_cluster')[['critic_score', 'troll_score']].mean()
        grouped['type'] = grouped.apply(lambda x: 'Critic' if x['critic_score'] > x['troll_score'] else 'Troll', axis=1)
        grouped = grouped.sort_values('troll_score', ascending=False)
        
        print("Top 5 'Troll' Clusters (Swearing/Anger in Negative Links):")
        print(grouped.head(5)[['troll_score', 'critic_score']].to_string())
        
        print("\nTop 5 'Critic' Clusters (Logic/Insight in Negative Links):")
        print(grouped.sort_values('critic_score', ascending=False).head(5)[['troll_score', 'critic_score']].to_string())
        return self._log_and_store('rq19_critics_vs_trolls', grouped)

    def run_all(self) -> None:
        """Run the full consolidated analysis pipeline."""
        print("STATISTICAL ANALYSIS: RESEARCH QUESTIONS\n")
        self.load_data()
        
        # Theme 1
        self.rq1a_analytical_bridges()
        self.rq1b_role_patterns()
        
        # Theme 2
        self.rq2a_emotion_dominance()
        self.rq2b_asymmetry_analysis()
        
        # Theme 3
        self.rq3a_similarity_sentiment()
        self.rq3b_ideological_neighbors()
        
        # Theme 4
        self.rq4a_certainty_insularity()
        self.rq4b_semantic_structural_interaction()
        
        # Theme 5
        self.rq5_topic_bridges()
        self.rq6_betweenness_predictors()
        self.rq7_cluster_sentiment_network()
        self.rq8_role_distribution()
        self.rq9_external_negativity()
        self.rq10_internal_civility()
        self.rq11_vocabulary_of_peace()
        self.rq12_vocabulary_of_war()
        self.rq13_anatomy_of_isolation()
        
        # Theme 6
        self.rq14_punching_bag_index()
        self.rq15_mainstream_curse()
        self.rq16_power_formula()
        self.rq17_david_vs_goliath()
        self.rq18_toxicity_contagion()
        self.rq19_critics_vs_trolls()
        
        print("\nALL ANALYSES COMPLETE")

if __name__ == "__main__":
    rq = ResearchQuestions(data_dir='data/processed')
    
    rq.run_all() 
    