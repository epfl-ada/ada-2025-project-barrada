import pandas as pd
import json
import numpy as np
from pathlib import Path

class VisualizationDataPrep:
    """
    Complete data preparation for all Reddit network visualizations.
    Generates JSON files for: Network, Insurgency, Toxicity, Echo, Roles, Power
    """
    
    def __init__(self, data_dir="data/processed", output_dir="docs/assets/data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_nodes_json(self):
        """Create nodes.json from cluster_master_dataset.csv"""
        print("Creating nodes.json...")
        
        df = pd.read_csv(self.data_dir / "cluster_master_dataset.csv")
        
        nodes = []
        for _, row in df.iterrows():
            internal_neg_pct = (1 - row.get('pct_positive_out', 0)) * 100
            
            node = {
                "name": row['cluster_label'],
                "pagerank": float(row.get('pagerank_mean', 0)),
                "internal_negativity": float(internal_neg_pct),
                "size": int(row['n_subreddits']),
                "insularity": float(row.get('insularity', 0) * 100),
                "top_subreddits": row.get('top5_by_pagerank', ''),
                
                # LIWC scores
                "Anger": float(row.get('LIWC_Anger_mean', 0)),
                "Negemo": float(row.get('LIWC_Negemo_mean', 0)),
                "Posemo": float(row.get('LIWC_Posemo_mean', 0)),
                "Certain": float(row.get('LIWC_Certain_mean', 0)),
                "CogMech": float(row.get('LIWC_CogMech_mean', 0)),
                "Social": float(row.get('LIWC_Social_mean', 0)),
                
                # Role data for filtering
                "pct_critical": float(row.get('pct_role_critical', 0) * 100),
                "pct_supportive": float(row.get('pct_role_supportive', 0) * 100),
                "pct_influential": float(row.get('pct_role_influential', 0) * 100),
                "pct_controversial": float(row.get('pct_role_controversial', 0) * 100),
                
                # Echo chamber data
                "semantic_purity": float(row.get('semantic_purity', 0) * 100) if 'semantic_purity' in row else 0,
            }
            
            nodes.append(node)
        
        output_path = self.output_dir / "nodes.json"
        with open(output_path, 'w') as f:
            json.dump(nodes, f, indent=2)
        
        print(f"Created {output_path} with {len(nodes)} nodes")
        return nodes
    
    def create_edges_json(self):
        """
        Create edges with sentiment-based filtering
        Strategy: Top 2 per cluster + edges >500 links
        """
        print("\nCreating edges.json...")
        
        flow_file = self.data_dir / "rq_analysis" / "rq10_cluster_flow_matrix.csv"
        if not flow_file.exists():
            print(f"WARNING: {flow_file} not found. Skipping edges.")
            return []
        
        df = pd.read_csv(flow_file)
        
        selected_edges = set()
        
        # Top 2 edges per cluster
        for cluster in df['src_cluster'].unique():
            cluster_edges = df[df['src_cluster'] == cluster].nlargest(2, 'n_links')
            for idx in cluster_edges.index:
                selected_edges.add(idx)
        
        # High volume edges
        high_volume = df[df['n_links'] > 500]
        for idx in high_volume.index:
            selected_edges.add(idx)
        
        df_filtered = df.loc[list(selected_edges)]
        
        edges = []
        for _, row in df_filtered.iterrows():
            edge = {
                "source": row['src_cluster'],
                "target": row['tgt_cluster'],
                "value": int(row['n_links']),
                "sentiment": float(row['avg_sentiment'])
            }
            edges.append(edge)
        
        output_path = self.output_dir / "edges.json"
        with open(output_path, 'w') as f:
            json.dump(edges, f, indent=2)
        
        print(f"Created {output_path} with {len(edges)} edges")
        return edges
    
    def create_rivalry_json(self):
        """Top 1 alliance + top 1 rival per cluster"""
        print("\nCreating rivalry.json...")
        
        flow_file = self.data_dir / "rq_analysis" / "rq10_cluster_flow_matrix.csv"
        if not flow_file.exists():
            print(f"WARNING: {flow_file} not found.")
            return {}
        
        df = pd.read_csv(flow_file)
        df_filtered = df[df['n_links'] >= 15].copy()
        
        all_alliances = []
        all_rivalries = []
        
        clusters = set(df_filtered['src_cluster'].unique()) | set(df_filtered['tgt_cluster'].unique())
        
        for cluster in clusters:
            cluster_links = df_filtered[df_filtered['src_cluster'] == cluster]
            
            if len(cluster_links) == 0:
                continue
            
            # Top 1 alliance
            alliance = cluster_links[cluster_links['avg_sentiment'] > 0.8].nlargest(1, 'avg_sentiment')
            if len(alliance) > 0:
                row = alliance.iloc[0]
                all_alliances.append({
                    "source_name": row['src_cluster'],
                    "target_name": row['tgt_cluster'],
                    "sentiment": float(row['avg_sentiment']),
                    "links": int(row['n_links'])
                })
            
            # Top 1 rivalry
            rivalry = cluster_links.nsmallest(1, 'avg_sentiment')
            if len(rivalry) > 0:
                row = rivalry.iloc[0]
                all_rivalries.append({
                    "source_name": row['src_cluster'],
                    "target_name": row['tgt_cluster'],
                    "sentiment": float(row['avg_sentiment']),
                    "links": int(row['n_links'])
                })
        
        output = {
            "alliances": all_alliances,
            "rivalries": all_rivalries
        }
        
        output_path = self.output_dir / "rivalry.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created {output_path}")
        return output
    
    def create_insurgency_json(self):
        """Create insurgency data"""
        print("\nCreating insurgency.json...")
        
        rq18_file = self.data_dir / "rq_analysis" / "rq18_david_vs_goliath.csv"
        if not rq18_file.exists():
            print(f"WARNING: {rq18_file} not found.")
            return {}
        
        df = pd.read_csv(rq18_file)
        
        output = {
            "total_attacks": int(df['total_attacks'].iloc[0]),
            "bullying_count": int(df['bullying_count'].iloc[0]),
            "insurgency_count": int(df['insurgency_count'].iloc[0]),
            "bullying_pct": float(df['bullying_pct'].iloc[0]),
            "insurgency_pct": float(df['insurgency_pct'].iloc[0]),
            "verdict": str(df['verdict'].iloc[0])
        }
        
        output_path = self.output_dir / "insurgency.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created {output_path}")
        return output
    
    def create_toxicity_json(self):
        """Create toxicity flow data for Sankey"""
        print("\nCreating toxicity.json...")
        
        rq16_file = self.data_dir / "rq_analysis" / "rq16_punching_bag_index.csv"
        if not rq16_file.exists():
            print(f"WARNING: {rq16_file} not found.")
            return {}
        
        df = pd.read_csv(rq16_file, index_col=0)  # ← Read cluster names as index
        
        # Top 10 victims (negative net flow)
        victims = df.nsmallest(10, 'net_toxicity_flow')
        victims_list = []
        for cluster_name in victims.index:  # ← Iterate over index (cluster names)
            row = victims.loc[cluster_name]
            victims_list.append({
                "cluster": str(cluster_name),
                "hate_import": int(row['hate_import']),
                "hate_export": int(row['hate_export']),
                "net_flow": int(row['net_toxicity_flow'])
            })
        
        # Top 10 bullies (positive net flow)
        bullies = df.nlargest(10, 'net_toxicity_flow')
        bullies_list = []
        for cluster_name in bullies.index:  # ← Iterate over index (cluster names)
            row = bullies.loc[cluster_name]
            bullies_list.append({
                "cluster": str(cluster_name),
                "hate_import": int(row['hate_import']),
                "hate_export": int(row['hate_export']),
                "net_flow": int(row['net_toxicity_flow'])
            })
        
        output = {
            "victims": victims_list,
            "bullies": bullies_list
        }
        
        output_path = self.output_dir / "toxicity.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created {output_path}")
        return output
        
    def create_echo_json(self):
        """Create echo chamber scatter data (RQ6)"""
        print("\nCreating echo.json...")
        
        cluster_file = self.data_dir / "cluster_master_dataset.csv"
        df = pd.read_csv(cluster_file)
        
        echo_data = []
        for _, row in df.iterrows():
            internal_neg_pct = (1 - row.get('pct_positive_out', 0)) * 100
            
            echo_data.append({
                "name": row['cluster_label'],
                "size": int(row['n_subreddits']),
                "purity": float(row.get('semantic_purity', 0) * 100) if 'semantic_purity' in row else 0,
                "civility": float(100 - internal_neg_pct),
                "pagerank": float(row.get('pagerank_mean', 0))
            })
        
        output_path = self.output_dir / "echo.json"
        with open(output_path, 'w') as f:
            json.dump(echo_data, f, indent=2)
        
        print(f"Created {output_path} with {len(echo_data)} clusters")
        return echo_data
    
    def create_roles_json(self):
        """Create role distribution data (RQ1)"""
        print("\nCreating roles.json...")
        
        cluster_file = self.data_dir / "cluster_master_dataset.csv"
        df = pd.read_csv(cluster_file)
        
        total_subs = df['n_subreddits'].sum()
        
        role_counts = {
            "Neutral": 0,
            "Influential": 0,
            "Supportive": 0,
            "Critical": 0,
            "Controversial": 0
        }
        
        for _, row in df.iterrows():
            n_subs = row['n_subreddits']
            role_counts["Critical"] += row.get('pct_role_critical', 0) * n_subs
            role_counts["Supportive"] += row.get('pct_role_supportive', 0) * n_subs
            role_counts["Influential"] += row.get('pct_role_influential', 0) * n_subs
            role_counts["Controversial"] += row.get('pct_role_controversial', 0) * n_subs
        
        # Neutral is the remainder
        role_counts["Neutral"] = total_subs - sum([role_counts[k] for k in role_counts if k != "Neutral"])
        
        role_pcts = {k: (v / total_subs * 100) for k, v in role_counts.items()}
        
        # Get top clusters for each role
        top_by_role = {}
        for role in ["Critical", "Supportive", "Influential", "Controversial"]:
            col_name = f'pct_role_{role.lower()}'
            if col_name in df.columns:
                top_clusters = df.nlargest(5, col_name)
                top_by_role[role] = [
                    {"name": row['cluster_label'], "pct": float(row[col_name] * 100)}
                    for _, row in top_clusters.iterrows()
                ]
        
        output = {
            "distribution": role_pcts,
            "top_by_role": top_by_role
        }
        
        output_path = self.output_dir / "roles.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created {output_path}")
        return output
    
    def create_power_json(self):
        """Create power correlation data (RQ17)"""
        print("\nCreating power.json...")
        
        power_data = {
            "overall_features": [
                {"feature": "Positive Links In", "correlation": 0.8785, "p_value": 0.0},
                {"feature": "Total In-Weight", "correlation": 0.8660, "p_value": 0.0},
                {"feature": "In-Degree Centrality", "correlation": 0.8320, "p_value": 0.0},
                {"feature": "In-Degree", "correlation": 0.8320, "p_value": 0.0},
                {"feature": "Unique Sources", "correlation": 0.8071, "p_value": 0.0},
                {"feature": "Betweenness", "correlation": 0.6354, "p_value": 0.0},
                {"feature": "Authority Score", "correlation": 0.6126, "p_value": 0.0},
                {"feature": "Clustering", "correlation": -0.5951, "p_value": 0.0},
                {"feature": "Total Degree", "correlation": 0.5944, "p_value": 0.0},
                {"feature": "Negative Links In", "correlation": 0.5743, "p_value": 0.0}
            ],
            "liwc_features": [
                {"feature": "Relativity", "correlation": 0.2317, "p_value": 9.17e-50},
                {"feature": "Tentative", "correlation": 0.1838, "p_value": 1.18e-31},
                {"feature": "Work", "correlation": 0.1613, "p_value": 1.14e-24},
                {"feature": "Inclusive", "correlation": 0.1586, "p_value": 6.80e-24},
                {"feature": "Negative Emotion", "correlation": -0.1496, "p_value": 2.03e-21},
                {"feature": "Anger", "correlation": -0.1466, "p_value": 1.30e-20},
                {"feature": "Prepositions", "correlation": 0.1421, "p_value": 1.91e-19},
                {"feature": "Time", "correlation": 0.1358, "p_value": 6.88e-18},
                {"feature": "Discrepancy", "correlation": 0.1332, "p_value": 2.96e-17},
                {"feature": "Motion", "correlation": 0.1297, "p_value": 1.95e-16}
            ],
            "toxicity_features": [
                {"feature": "% Positive Out", "correlation": -0.7627, "p_value": 4.10e-08},
                {"feature": "Avg Out Sentiment", "correlation": -0.7627, "p_value": 4.10e-08},
                {"feature": "% Critical Role", "correlation": 0.7545, "p_value": 6.93e-08},
                {"feature": "LIWC Negemo", "correlation": 0.7525, "p_value": 7.83e-08},
                {"feature": "Emotional Score", "correlation": 0.7404, "p_value": 1.62e-07},
                {"feature": "LIWC Anger", "correlation": 0.7039, "p_value": 1.17e-06},
                {"feature": "% Positive In", "correlation": -0.6752, "p_value": 4.56e-06},
                {"feature": "Avg In Sentiment", "correlation": -0.6752, "p_value": 4.56e-06},
                {"feature": "% Controversial", "correlation": 0.6147, "p_value": 5.18e-05},
                {"feature": "LIWC Religion", "correlation": 0.5654, "p_value": 0.000266}
            ]
        }
        
        output_path = self.output_dir / "power.json"
        with open(output_path, 'w') as f:
            json.dump(power_data, f, indent=2)
        
        print(f"Created {output_path}")
        return power_data
    

    def create_bridges_json(self):
        """Create betweenness predictor data (RQ8)"""
        print("\nCreating bridges.json...")
        
        # Data taken from your RQ8 Analysis Log
        bridges_data = {
            "overall_features": [
                {"feature": "Total Degree", "correlation": 0.7287, "category": "Structural"},
                {"feature": "In-Degree", "correlation": 0.7169, "category": "Structural"},
                {"feature": "Out-Degree", "correlation": 0.6628, "category": "Structural"},
                {"feature": "Unique Sources", "correlation": 0.6562, "category": "Structural"},
                {"feature": "Unique Targets", "correlation": 0.6500, "category": "Structural"},
                {"feature": "PageRank", "correlation": 0.6356, "category": "Structural"},
                {"feature": "Authority Score", "correlation": 0.6335, "category": "Structural"},
                {"feature": "In-Weight", "correlation": 0.6293, "category": "Structural"}
            ],
            "liwc_features": [
                {"feature": "Non-fluencies (uh, um)", "correlation": 0.2678, "category": "Linguistic"},
                {"feature": "Friends", "correlation": 0.2598, "category": "Linguistic"},
                {"feature": "Family", "correlation": 0.2455, "category": "Linguistic"},
                {"feature": "Assent (yes, ok)", "correlation": 0.2431, "category": "Linguistic"},
                {"feature": "Home", "correlation": 0.2300, "category": "Linguistic"},
                {"feature": "Filler (you know)", "correlation": 0.2272, "category": "Linguistic"},
                {"feature": "Anxiety", "correlation": 0.2132, "category": "Linguistic"},
                {"feature": "Ingest (Food/Drink)", "correlation": 0.2103, "category": "Linguistic"},
                {"feature": "Swear", "correlation": 0.2080, "category": "Linguistic"},
                {"feature": "Sexual", "correlation": 0.1965, "category": "Linguistic"}
            ]
        }
        
        output_path = self.output_dir / "bridges.json"
        with open(output_path, 'w') as f:
            json.dump(bridges_data, f, indent=2)
        
        print(f"Created {output_path}")
        return bridges_data
    
    def create_quadrant_json(self):
        """Create roles_scatter.json for Interactive Quadrant Map"""
        print("\nCreating roles_scatter.json...")
        
        final_df_path = self.data_dir / "final_dataset.csv"
        
        if not final_df_path.exists():
             print(f"WARNING: {final_df_path} not found. Skipping quadrant map.")
             return []

        df = pd.read_csv(final_df_path)
        
        df_top = df.nlargest(200, 'total_links').copy()
        
        export_df = df_top[[
            'subreddit', 
            'pos_out_ratio',     
            'neg_in_ratio',   
            'total_links',      
            'avg_out_sentiment', 
            'role_critical',      
            'role_controversial',
            'role_supportive',
            'role_influential'
        ]].copy()
        
        export_df.columns = [
            'name', 'x', 'y', 'size', 'sentiment', 
            'is_crit', 'is_cont', 'is_supp', 'is_inf'
        ]
        
        export_df = export_df.fillna(0)
        
        data = export_df.to_dict(orient='records')
        
        output_path = self.output_dir / "roles_scatter.json"
        with open(output_path, 'w') as f:
            json.dump(data, f)
            
        print(f"Created {output_path} with {len(data)} nodes")
        return data
    
    def run_all(self):
        """Run all JSON generation tasks"""
        print("COMPLETE VISUALIZATION DATA PREPARATION")
        
        self.create_nodes_json()
        self.create_edges_json()
        self.create_rivalry_json()
        self.create_insurgency_json()
        self.create_toxicity_json()
        self.create_echo_json()
        self.create_roles_json()
        self.create_power_json()
        self.create_bridges_json()
        self.create_quadrant_json()
        
        print("All JSON files created successfully!")
        print(f"Output directory: {self.output_dir}")

if __name__ == "__main__":
    prep = VisualizationDataPrep()
    prep.run_all()