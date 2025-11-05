import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.ticker as ticker
try:
    from adjustText import adjust_text
except ImportError:
    print("WARNING: 'adjustText' not found. Labels will overlap.")
    print("Run 'pip install adjustText' for cleaner plots.")
    def adjust_text(texts, **kwargs): pass

# Define Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
HTML_DIR = BASE_DIR / 'results' / 'html'

# Ensure all output directories exist
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
HTML_DIR.mkdir(exist_ok=True, parents=True)


# Helper Function

def load_hyperlinks_data(file_name="combined_hyperlinks.csv"):
    """Loads the processed hyperlinks CSV"""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    return df

def load_pca_variance_data(file_name="pca_variance.csv"):
    """Loads PCA variance data."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_embeddings_data(file_name="embeddings_processed.csv"):
    """Loads processed embeddings"""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    try:
        use_cols = ['subreddit', 'pca_0', 'pca_1']
        df = pd.read_csv(csv_path, usecols=lambda c: c in use_cols or c.startswith('pca_'))
        return df
    except ValueError:
        return pd.read_csv(csv_path)
    
def load_liwc_sentiment_comparison(file_name="liwc_sentiment_comparison.csv"):
    """Loads the LIWC sentiment comparison data."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_subreddit_roles(file_name="subreddit_roles.csv"):
    """Loads the merged subreddit roles and features data."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_psychological_profiles(file_name="psychological_profiles.csv"):
    """Loads the top-100 psychological profiles."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_network_metrics(file_name="network_node_metrics.csv"):
    """Loads the node-level network metrics (pagerank, centrality, etc.)."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_network_communities(file_name="network_communities.csv"):
    """Loads the community assignments for each subreddit."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_network_edges(file_name="network_edges.csv"):
    """Loads the aggregated network edge list."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_topic_clusters(file_name="embeddings_kmeans_40.csv"):
    """Loads the topic cluster assignment for each subreddit."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_cluster_labels(file_name="cluster_labels_40.csv"):
    """Loads the labels for the 40 topic clusters."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def load_final_dataset(file_name="final_dataset.csv"):
    """Loads the final, fully integrated dataset."""

    csv_path = PROCESSED_DIR / file_name
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        print("Please run the final integration step first.")
        return None
    
    print(f"Loading final dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} subreddits with {df.shape[1]} features.")
    return df

# =============================================================================
# STEP 1: HYPERLINK VISUALIZATIONS
# =============================================================================

def plot_sentiment_distribution(df, save_name="step1_sentiment_distribution.png"):
    """ Creates a bar chart of positive vs. negative link counts."""

    print(f"Generating {save_name}...")
    
    counts = df['LINK_SENTIMENT'].value_counts().reset_index()
    counts.columns = ['LINK_SENTIMENT', 'Count']
    counts = counts[counts['LINK_SENTIMENT'].isin([1, -1])].copy()
    counts['Sentiment'] = counts['LINK_SENTIMENT'].map({
         1: 'Positive (+1)', 
        -1: 'Negative (-1)'
    })
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x='Sentiment', 
        y='Count',  
        hue='Sentiment',
        data=counts, 
        palette={'Positive (+1)': '#54a0ff', 'Negative (-1)': '#ff6b6b'},
        order=['Positive (+1)', 'Negative (-1)'],
        legend=False           
)

    ax.set_title('Overall Sentiment Distribution of Links', fontsize=16, pad=15)
    ax.set_xlabel('Link Sentiment', fontsize=12)
    ax.set_ylabel('Total Count', fontsize=12)
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=12)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path


def plot_sentiment_volume_over_time(
    df,
    save_name="step1_sentiment_volume_over_time.png",
    rolling_window=3,
    dpi=150,
):
    print(f"Generating {save_name}...")
    if 'timestamp' not in df.columns:
        print(f"Cannot generate {save_name}: 'timestamp' column not found.")
        return

    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
    d = d.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    if d.empty:
        print("No timestamped rows to plot.")
        return

    # Monthly aggregates
    monthly_total = d.resample('ME').size().rename('Total')
    monthly_pos = d[d['LINK_SENTIMENT'] == 1].resample('ME').size().rename('Positive')
    monthly_neg = d[d['LINK_SENTIMENT'] == -1].resample('ME').size().rename('Negative')

    df_monthly = pd.concat([monthly_total, monthly_pos, monthly_neg], axis=1).fillna(0).astype(int)
    if df_monthly.empty:
        print("No monthly data after resampling.")
        return

    # 3-month rolling average of Total for trend
    df_monthly['Total_Roll'] = df_monthly['Total'].rolling(window=rolling_window, min_periods=1).mean()

    total_color = '#808080'
    pos_color = '#54a0ff'
    neg_color = '#ff6b6b'

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.stackplot(
        df_monthly.index,
        df_monthly['Positive'],
        df_monthly['Negative'],
        labels=['Positive', 'Negative'],
        colors=[pos_color, neg_color],
        alpha=0.85
    )

    # Bold total line and rolling average
    ax.plot(df_monthly.index, df_monthly['Total'], label='Total (monthly)', color=total_color,
            linewidth=2.0, linestyle='-', zorder=5)
    ax.plot(df_monthly.index, df_monthly['Total_Roll'], label=f'{rolling_window}-mo rolling avg',
            color='black', linewidth=2.2, linestyle='--', zorder=6)

    # Highlight the month with the largest number of Negative links
    if (df_monthly['Negative'].sum() > 0):
        peak_neg_idx = df_monthly['Negative'].idxmax()
        peak_neg_val = int(df_monthly['Negative'].max())
        ax.axvline(peak_neg_idx, color=neg_color, alpha=0.35, linewidth=1.2, linestyle=':')
        ax.annotate(
            f"Peak negative: {peak_neg_val:,}\n({peak_neg_idx.strftime('%Y-%m')})",
            xy=(peak_neg_idx, df_monthly.loc[peak_neg_idx, 'Negative']),
            xytext=(10, -60),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff6b6b", alpha=0.9)
        )

    # Formatting & labels
    ax.set_title('Link Volume Over Time by Sentiment', fontsize=18, pad=14)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Number of Links per Month', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # X-axis ticks
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=30)

    # Y-axis formatting with commas
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()

    save_path = FIGURES_DIR / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f"Saved PNG: {save_path}")
    return save_path

def plot_top_sources_sentiment(df, top_k=15, save_name="step1_top_sources.png"):
    """Similar top sources."""

    print(f"Generating {save_name}...")
    
    # Get counts of pos/neg per source
    counts = df[df['LINK_SENTIMENT'].isin([1, -1])].groupby('SOURCE_SUBREDDIT')['LINK_SENTIMENT'].value_counts().unstack(fill_value=0)
    counts.columns = ['Negative', 'Positive']
    
    # Calculate total and get top K 
    counts['Total'] = counts['Negative'] + counts['Positive']
    top_k_sources = counts.nlargest(top_k, 'Total').sort_values('Total', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_k_sources[['Positive', 'Negative']].plot(
        kind='barh', 
        stacked=True, 
        color={'Positive': '#54a0ff', 'Negative': '#ff6b6b'},
        ax=ax
    )
    
    ax.set_title(f'Top {top_k} Link Sources by Volume and Sentiment', fontsize=16, pad=15)
    ax.set_xlabel('Total Links Sent', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    ax.legend(title='Sentiment')
    
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

def plot_top_targets_sentiment(df, top_k=15, save_name="step1_top_targets.png"):
    """Graph: Top Targets"""

    print(f"Generating {save_name}...")

    if 'TARGET_SUBREDDIT' not in df.columns or 'LINK_SENTIMENT' not in df.columns:
        print("Cannot generate top targets plot: required columns missing.")
        return

    counts = (
        df[df['LINK_SENTIMENT'].isin([1, -1])]
        .groupby('TARGET_SUBREDDIT')['LINK_SENTIMENT']
        .value_counts()
        .unstack(fill_value=0)
    )

    # Normalize column names to 'Positive' and 'Negative'
    if 1 in counts.columns and -1 in counts.columns:
        counts = counts.rename(columns={1: 'Positive', -1: 'Negative'})
    else:
        counts = counts.rename(columns={c: ('Positive' if c == 1 else 'Negative') for c in counts.columns})

    counts = counts.reindex(columns=['Positive', 'Negative'], fill_value=0)
    counts['Total'] = counts['Positive'] + counts['Negative']

    top_k_targets = counts.nlargest(top_k, 'Total').sort_values('Total', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos_color = '#54a0ff'
    neg_color = '#ff6b6b'
    top_k_targets[['Positive', 'Negative']].plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=[pos_color, neg_color],
        width=0.75
    )

    ax.set_title(f'Top {top_k} Link Targets by Volume and Sentiment', fontsize=16, pad=15)
    ax.set_xlabel('Total Links', fontsize=12)
    ax.set_ylabel('Target Subreddit', fontsize=12)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend(title='Sentiment')

    # Annotate totals at end of bars
    for i, (idx, row) in enumerate(top_k_targets.iterrows()):
        total = int(row['Total'])
        ax.annotate(f"{total:,}", xy=(row['Total'], i), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

def plot_liwc_radar_profiles(df, sub_a, sub_b, label_a="Sub A", label_b="Sub B", min_links=100, save_name="step1_liwc_radar_case_study.png"):
    """Compares the LIWC profiles of two specific subreddits."""
    print(f"Generating Case Study: r/{sub_a} vs. r/{sub_b}...")

    # Get data for the two subreddits
    sub_a_data = df[df['SOURCE_SUBREDDIT'] == sub_a]
    sub_b_data = df[df['SOURCE_SUBREDDIT'] == sub_b]

    # Check if they have enough data
    if len(sub_a_data) < min_links:
        print(f"  Error: 'r/{sub_a}' has only {len(sub_a_data)} links (min {min_links}). Skipping.")
        return
    if len(sub_b_data) < min_links:
        print(f"  Error: 'r/{sub_b}' has only {len(sub_b_data)} links (min {min_links}). Skipping.")
        return
    
    print(f"  Comparing r/{sub_a} (n={len(sub_a_data)} links) vs. r/{sub_b} (n={len(sub_b_data)} links)")

    features = ['LIWC_Anger', 'LIWC_Swear', 'LIWC_Work', 'LIWC_Money', 'LIWC_I', 'LIWC_We','LIWC_Certain']
    labels = ['Anger', 'Swear', 'Work', 'Money', 'I (Self)', 'We (Group)','Certainty']    

    # Calculate raw means for these two subreddits
    vals_a_raw = sub_a_data[features].mean()
    vals_b_raw = sub_b_data[features].mean()
    
    plot_max = max(vals_a_raw.max(), vals_b_raw.max())
    upper_limit = max(0.05, plot_max * 1.2)
    print(f"  Setting plot upper limit to: {upper_limit:.2f}")
    
    # Prepare Radar Data
    vals_a = pd.concat([vals_a_raw, vals_a_raw.iloc[[0]]]).values
    vals_b = pd.concat([vals_b_raw, vals_b_raw.iloc[[0]]]).values
    angles = [n / float(len(features)) * 2 * np.pi for n in range(len(features))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    ax.plot(angles, vals_a, linewidth=2, linestyle='solid', label=f"r/{sub_a} ({label_a})", color='#54a0ff')
    ax.fill(angles, vals_a, '#54a0ff', alpha=0.25)
    
    ax.plot(angles, vals_b, linewidth=2, linestyle='solid', label=f"r/{sub_b} ({label_b})", color='#ff6b6b')
    ax.fill(angles, vals_b, '#ff6b6b', alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    
    ax.set_ylim(0, upper_limit)
    ax.set_rlabel_position(30)
    
    plt.title(f"Linguistic Profile Case Study: r/{sub_a} vs. r/{sub_b}\n(Raw LIWC % Scores)", size=15, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    print(f"Saved PNG: {save_path}")
    return save_path

def plot_attack_pattern_small_multiples(df, save_name="step1_attack_patterns.png"):
    print(f"Generating {save_name}...")
    
    neg_df = df[df['LINK_SENTIMENT'] == -1].copy()
    
    # Find top 6 most aggressive sources
    top_aggressors = neg_df['SOURCE_SUBREDDIT'].value_counts().head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=False)
    axes = axes.flatten()
    
    for i, aggressor in enumerate(top_aggressors):
        ax = axes[i]
        
        # Get this aggressor's top 5 targets
        targets = neg_df[neg_df['SOURCE_SUBREDDIT'] == aggressor]['TARGET_SUBREDDIT'].value_counts().head(5)
        
        sns.barplot(x=targets.values, y=targets.index, ax=ax, color='#ff6b6b')
        
        ax.set_title(f"r/{aggressor} attacks...", fontsize=12, fontweight='bold')
        ax.set_xlabel("Negative Links Sent")
        ax.set_ylabel("") 

    plt.suptitle("Top 6 Aggressor Communities and their Favorite Targets", fontsize=18, y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

def plot_liwc_diverging_lollipop(df, save_name="step1_liwc_diverging.png"):
    print(f"Generating {save_name}...")
    
    liwc_cats = [
        'LIWC_Anger', 'LIWC_Anx', 'LIWC_Sad', 'LIWC_Posemo', 'LIWC_Negemo',
        'LIWC_Insight', 'LIWC_Cause', 'LIWC_Tentat', 'LIWC_Certain',
        'LIWC_Social', 'LIWC_Family', 'LIWC_Friends', 'LIWC_Work',
        'LIWC_Money', 'LIWC_Relig', 'LIWC_Death', 'LIWC_Swear'
    ]
    
    valid_cats = [c for c in liwc_cats if c in df.columns]
    if not valid_cats:
        print("No valid LIWC categories found for diverging plot.")
        return

    # standardize
    std_df = df.copy()
    for col in valid_cats:
        std_df[col] = (std_df[col] - std_df[col].mean()) / std_df[col].std()

    # Calculate mean Z-score
    pos_means = std_df[std_df['LINK_SENTIMENT'] == 1][valid_cats].mean()
    neg_means = std_df[std_df['LINK_SENTIMENT'] == -1][valid_cats].mean()
    
    diff = pos_means - neg_means
    diff = diff.sort_values()

    fig, ax = plt.subplots(figsize=(10, 10))
    
    my_color = np.where(diff >= 0, '#54a0ff', '#ff6b6b')
    
    ax.hlines(y=diff.index, xmin=0, xmax=diff.values, color=my_color, alpha=0.8, linewidth=3)
    ax.scatter(x=diff.values, y=diff.index, color=my_color, s=100, alpha=1.0)
    
    ax.axvline(0, color='grey', linestyle='--', alpha=0.6)
    ax.set_title('What distinguishes Positive vs Negative links?\n(Difference in Standardized LIWC Scores)', fontsize=16, pad=15)
    ax.set_xlabel('← More typical of NEGATIVE links       More typical of POSITIVE links →', fontsize=12, fontweight='bold')
    
    clean_labels = [l.replace('LIWC_', '') for l in diff.index]
    ax.set_yticks(range(len(diff.index))) 
    ax.set_yticklabels(clean_labels, fontsize=11)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

# =============================================================================
# STEP 2: Psychological (LIWC) Analysis
# =============================================================================

def plot_role_quadrant_map(df_roles, top_n=200, save_name="step2_role_quadrant.png"):
    """Plots the social roles of the most active subreddits."""

    print(f"Generating {save_name}...")
    
    df_top = df_roles.nlargest(top_n, 'total_links').copy()
    
    fig, ax = plt.subplots(figsize=(15, 12)) 
    
    sns.scatterplot(
        data=df_top,
        x='pos_out_ratio',
        y='neg_in_ratio',
        size='total_links', 
        sizes=(30, 1500),
        alpha=0.6,
        hue='avg_sentiment_out', 
        palette='RdBu_r',
        legend=False,
        ax=ax
    )
    
    x_med = df_top['pos_out_ratio'].median()
    y_med = df_top['neg_in_ratio'].median()
    
    ax.axvline(x_med, color='grey', linestyle='--', alpha=0.5)
    ax.axhline(y_med, color='grey', linestyle='--', alpha=0.5)
    
    ax.text(0.98, 0.02, 'Controversial\nInflential', 
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=12, color='black', alpha=0.8)
    
    ax.text(0.02, 0.98, 'Critical\nControversial', 
            transform=ax.transAxes, ha='left', va='top', 
            fontsize=12, color='black', alpha=0.8)
    
    ax.text(0.98, 0.98, 'Supportive &\nInflential', 
            transform=ax.transAxes, ha='right', va='top', 
            fontsize=12, color='black', alpha=0.8)

    ax.text(0.02, 0.02, 'Niche &\nSelf-Contained', 
            transform=ax.transAxes, ha='left', va='bottom', 
            fontsize=12, color='black', alpha=0.8)

    ax.set_title(f'Role Map for Top {top_n} Most Active Subreddits', fontsize=16, pad=15)
    ax.set_xlabel('Positive Outgoing Ratio (← Less | More →)', fontsize=12)
    ax.set_ylabel('Negative Incoming Ratio (← Less | More →)', fontsize=12)
    
    df_label = df_top.nlargest(10, 'total_links') 
    texts = []
    for i, row in df_label.iterrows():
        texts.append(ax.text(row['pos_out_ratio'], row['neg_in_ratio'], row['subreddit'], fontsize=9, fontweight='bold'))
    
    adjust_text(texts, ax=ax, force_text=(0.5, 0.5))
    
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path


def plot_psychological_asymmetry(df_roles, top_n=25, save_name="step2_anger_asymmetry.png"):
    """Shows the gap between outgoing vs. incoming 'Anger'."""
    print(f"Generating {save_name}...")
    
    df = df_roles.copy()
    
    if 'LIWC_Anger_mean_out' not in df.columns or 'LIWC_Anger_mean_in' not in df.columns:
        print("SKIPPING {save_name}: Required LIWC_Anger columns not found.")
        return

    df_filt = df[(df['n_links_in'] > 100) & (df['n_links_out'] > 100)].copy()
    
    df_plot = df_filt.nlargest(top_n, 'LIWC_Anger_mean_in').sort_values('LIWC_Anger_mean_in')
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    ax.hlines(
        y=df_plot['subreddit'], 
        xmin=df_plot['LIWC_Anger_mean_out'], 
        xmax=df_plot['LIWC_Anger_mean_in'], 
        color='grey', 
        alpha=0.5
    )
    
    ax.scatter(df_plot['LIWC_Anger_mean_out'], df_plot['subreddit'], color='#54a0ff', s=60, label='Language They Use (Outgoing)')
    ax.scatter(df_plot['LIWC_Anger_mean_in'], df_plot['subreddit'], color='#ff6b6b', s=60, label='Language Used Against Them (Incoming)')
    
    ax.set_title(f'Anger Asymmetry: Language Used vs. Received\n(Top {top_n} by incoming "Anger" score)', fontsize=15, pad=15)
    ax.set_xlabel('Average "Anger" Score', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path


def plot_top_influential(roles_df, top_k=10, save_name="step2_top_influential.png"):
    """ Creates a bar chart of the top 'influential' subreddits."""
    print(f"Generating {save_name}...")
    
    champions = roles_df[roles_df['role_influential']].sort_values('n_positive_links_out', ascending=False).head(top_k)
    
    champions = champions.sort_values('n_positive_links_out', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.barh(champions['subreddit'], champions['n_positive_links_out'], color='#54a0ff')
    
    ax.set_title(f'Top {top_k} "Influential" Subreddits', fontsize=16, pad=15)
    ax.set_xlabel('Total Positive Outgoing Links', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    for i, (value, name) in enumerate(zip(champions['n_positive_links_out'], champions['subreddit'])):
        ax.text(value, i, f' {int(value):,}', va='center', ha='left', fontsize=10)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path


def plot_top_supported(roles_df, top_k=10, save_name="step2_top_supported.png"):
    """Creates a bar chart of the top 'supportive' subreddits."""
    print(f"Generating {save_name}...")
    
    amplifiers = roles_df[roles_df['role_supportive']].sort_values('n_positive_links_in', ascending=False).head(top_k)
    
    amplifiers = amplifiers.sort_values('n_positive_links_in', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.barh(amplifiers['subreddit'], amplifiers['n_positive_links_in'], color='#2ecc71')
    
    ax.set_title(f'Top {top_k} "Supported" Subreddits', fontsize=16, pad=15)
    ax.set_xlabel('Total Positive Incoming Links', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    for i, (value, name) in enumerate(zip(amplifiers['n_positive_links_in'], amplifiers['subreddit'])):
        ax.text(value, i, f' {int(value):,}', va='center', ha='left', fontsize=10)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

# =============================================================================
# STEP 3: Network Strcuture
# =============================================================================

def plot_centrality_grid(df_metrics, top_n=15, save_name="step3_centrality_grid.png"):
    """Shows the top N subreddits for four different centrality metrics in a 2x2 grid."""

    print(f"Generating {save_name}...")
    
    metrics_to_plot = {
        'PageRank': 'pagerank',
        'Betweenness': 'betweenness',
        'Hub Score': 'hub_score',
        'Authority Score': 'authority_score'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    for i, (title, metric) in enumerate(metrics_to_plot.items()):
        if metric not in df_metrics.columns:
            print(f"  Skipping {title}: column '{metric}' not found.")
            continue
            
        ax = axes[i]
        
        df_top = df_metrics.nlargest(top_n, metric).sort_values(metric, ascending=True)
        
        ax.barh(df_top['subreddit'], df_top[metric], color='#54a0ff')
        
        ax.set_title(f'Top {top_n} by {title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=10)
    
    plt.suptitle('Network Influencers: Top Subreddits by Centrality', fontsize=20, y=1.03)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

# =============================================================================
# STEP 4: EMBEDDING VISUALIZATIONS
# =============================================================================

def plot_pca_variance(df_var, save_name="step4_pca_variance.png"):
    """Combined bar and line plot for PCA variance."""
    print(f"Generating {save_name}...")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Bar chart for explained variance
    sns.barplot(x='component', y='explained_variance', data=df_var, color='#54a0ff', alpha=0.6, ax=ax1, zorder=1)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (per component)', fontsize=12, color='#54a0ff')
    ax1.tick_params(axis='y', labelcolor='#54a0ff')
    
    for i, label in enumerate(ax1.get_xticklabels()):
        if (i + 1) % 5 != 0 and i != 0:
            label.set_visible(False)

    # Line chart for cumulative variance
    ax2 = ax1.twinx()
    sns.lineplot(x=df_var.index, y='cumulative_variance', data=df_var, color='#ee5253', marker='o', linewidth=2, ax=ax2, zorder=2)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, color='#ee5253')
    ax2.tick_params(axis='y', labelcolor='#ee5253')
    ax2.set_ylim(0, 1.05)
    
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax2.text(len(df_var)-1, 0.94, '90% Threshold', color='gray', ha='center')

    plt.title('PCA Explained Variance (Scree Plot)', fontsize=16, pad=15)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    print(f"Saved PNG: {save_path}")
    return save_path

# =============================================================================
# STEP 5: Cluesters
# =============================================================================

def plot_topic_cluster_distribution(df_clusters, save_name="step5_topic_distribution.png"):
    """Creates a horizontal bar chart showing the size of each topic cluster."""
    print(f"Generating {save_name}...")
    
    size_counts = df_clusters['topic_cluster_label'].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    ax.barh(size_counts.index, size_counts.values, color='#54a0ff')
    
    ax.set_title('Topic Cluster Size Distribution (K=40)', fontsize=16, pad=15)
    ax.set_xlabel('Number of Subreddits', fontsize=12)
    ax.set_ylabel('Topic Cluster Label', fontsize=12)
    
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    for i, v in enumerate(size_counts.values):
        ax.text(v + 50, i, f'{v:,}', va='center', fontsize=9)
        
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

# =============================================================================
# FINAL STEP
# =============================================================================

def plot_topic_network_role(df_final, top_n=40, save_name="step6_topic_network_role.png"):
    """Dumbbell plot showing avg incoming vs. outgoing sentiment for each topic."""

    print(f"Generating {save_name}...")
    
    cols = ['topic_cluster_label', 'avg_in_sentiment', 'avg_out_sentiment', 'total_links']
    if not all(c in df_final.columns for c in cols):
        print(f"SKIPPING {save_name}: Missing required columns.")
        return

    df_agg = df_final.groupby('topic_cluster_label')[cols].mean(numeric_only=True)
    
    df_plot = df_agg.nlargest(top_n, 'total_links').sort_values('avg_in_sentiment')
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    ax.hlines(
        y=df_plot.index, 
        xmin=df_plot['avg_in_sentiment'], 
        xmax=df_plot['avg_out_sentiment'], 
        color='grey', 
        alpha=0.5
    )
    
    ax.scatter(df_plot['avg_in_sentiment'], df_plot.index, color='#ff6b6b', s=60, label='Avg. Incoming Sentiment')
    ax.scatter(df_plot['avg_out_sentiment'], df_plot.index, color='#54a0ff', s=60, label='Avg. Outgoing Sentiment')
    
    ax.set_title(f'Network Role of Top {top_n} Topics', fontsize=16, pad=15)
    ax.set_xlabel('Average Link Sentiment (-1 to +1)', fontsize=12)
    ax.set_ylabel('Topic Cluster', fontsize=12)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Neutral (0.0)')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path

def plot_semantic_structural_heatmap(df_final, top_n=15, save_name="step6_sem_struct_heatmap.png"):
    """Heatmap showing the intersection of Topic Clusters (Semantic) vs.Network Communities (Structural)."""

    print(f"Generating {save_name}...")
    
    cols = ['topic_cluster_label', 'community', 'subreddit']
    if not all(c in df_final.columns for c in cols):
        print(f"SKIPPING {save_name}: Missing required columns.")
        return
        
    df_clean = df_final[cols].dropna()

    top_topics = df_clean['topic_cluster_label'].value_counts().head(top_n).index
    top_comms = df_clean['community'].value_counts().head(top_n).index

    df_top = df_clean[
        df_clean['topic_cluster_label'].isin(top_topics) &
        df_clean['community'].isin(top_comms)
    ]
    
    df_pivot = pd.crosstab(df_top['topic_cluster_label'], df_top['community'])
    
    df_plot = np.log10(df_pivot + 1)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        df_plot,
        annot=df_pivot, 
        fmt=",",
        cmap='viridis',
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title('Semantic vs. Structural Alignment', fontsize=16, pad=15)
    ax.set_xlabel(f'Top {top_n} Network Communities (Structural)', fontsize=12)
    ax.set_ylabel(f'Top {top_n} Topic Clusters (Semantic)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path)
    plt.show()
    print(f"Saved PNG: {save_path}")
    return save_path


def plot_liwc_role_lift(
    df: pd.DataFrame,
    key_liwc=None,
    role_columns=None,
    aggregate_by_subreddit_col="SRC_SUBREDDIT",
    save_name="step6_emotion_profiles_lift.png",
    cmap="RdYlGn"
):
    """Plot a heatmap of LIWC relative lift by role"""

    if key_liwc is None:
        key_liwc = [
            'LIWC_Anger_mean', 'LIWC_Posemo_mean', 'LIWC_Anx_mean',
            'LIWC_Sad_mean', 'LIWC_We_mean', 'LIWC_They_mean',
            'LIWC_Certain_mean', 'LIWC_Tentat_mean'
        ]

    available_liwc = [c for c in key_liwc if c in df.columns]
    if not available_liwc:
        raise ValueError("No LIWC columns found. Checked: " + ", ".join(key_liwc))

    if not role_columns:
        role_columns = [c for c in df.columns if c.startswith("role_")]
    active_roles = [rc for rc in role_columns if rc in df.columns]
    if not active_roles:
        raise ValueError("No role_* columns found in df.")

    def _compute_role_means(frame: pd.DataFrame, liwc_cols, role_cols):
        role_means = {}
        for rc in role_cols:
            role_name = rc.replace("role_", "")
            mask = frame[rc] == True
            role_means[role_name] = (
                frame.loc[mask, liwc_cols].mean(numeric_only=True)
                if mask.any() else pd.Series(index=liwc_cols, dtype=float)
            )
        out = pd.DataFrame(role_means).T
        return out.reindex(columns=liwc_cols)

    if aggregate_by_subreddit_col in df.columns:
        by_sub_liwc = df.groupby(aggregate_by_subreddit_col, as_index=True)[available_liwc].mean(numeric_only=True)
        role_by_sub = df.groupby(aggregate_by_subreddit_col, as_index=True)[active_roles].any()
        merged = by_sub_liwc.join(role_by_sub, how='inner')
        emotion_df = _compute_role_means(merged, available_liwc, active_roles)
        gmean = by_sub_liwc.mean(numeric_only=True)
    else:
        emotion_df = _compute_role_means(df, available_liwc, active_roles)
        gmean = df[available_liwc].mean(numeric_only=True)

    eps = 1e-12
    lift = (emotion_df - gmean) / (gmean + eps)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        lift, annot=True, fmt=".1%",
        cmap=cmap, center=0,
        cbar_kws={'label': 'Relative lift vs global mean'}
    )
    plt.title("LIWC: Relative Lift by Role", fontsize=14, fontweight='bold')
    plt.xlabel("LIWC Feature")
    plt.ylabel("Role")
    plt.tight_layout()

    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path
