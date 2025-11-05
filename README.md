# Reddit’s Invisible Brain: The Web Hidden Beneath the Threads

## Abstract

Reddit is not just a collection of online forums; it is a living ecosystem of interconnected communities, each with its own emotional and cognitive identity. Through hyperlinks, subreddits comment on, critique, and reference one another, forming a complex web of inter-community relationships.

This project aims to uncover the hidden psychological structure underlying that web by integrating three perspectives:

- **Structural:** Who talks about whom? (Network topology, centrality, community structure)  
- **Psychological (LIWC):** How do they talk? (Emotional tone, cognitive style, "we" vs "they" linguistics)  
- **Semantic:** What are they about? (Topic embeddings)

By combining these layers, we seek to visualize Reddit as a “social MRI”, revealing how communities align, clash, or coexist—not only by what they discuss but how they think and feel. Our goal is to tell the story of Reddit’s invisible brain: the emotional and cognitive currents that shape the flow of ideas across its digital landscape.

## Research Questions

We organize the project around **four themes** that connect roles, psychology, topics, and structure. 

### **Network Structure & Roles**

* Do communities with specific psychological profiles (e.g., high **Anger**, high **We/They**) occupy distinct positions (central vs. peripheral; high PageRank/HITS vs. low)?
* Are “analytical” communities (high **Cognitive**: Certain/Tentative/Insight) more likely to act as **bridges** (high betweenness) between disparate clusters?
* Which subreddits emerge as influential amplifiers, supportive hubs, critical, or controversial targets, and what linguistic patterns characterize them?

### **Emotional Geography**

* Which emotions (e.g., **Anger**, **Anxiety**, **Certainty**/**Tentativeness**) dominate across **negative vs. positive** links?
* Do critical communities (high **outgoing negativity**) exhibit **depression markers** (high “I” usage + negative affect)?
* Where do we observe **asymmetries**: the language a subreddit uses vs. the language used **against** it (incoming vs. outgoing LIWC)?

### **Social Dynamics**

* Are **positive links** more common between **psychologically or topically similar** subreddits (homophily)?
* Do **ideological neighbors** criticize each other more than distant communities?

### **Echo Chambers & Polarization**

* Do communities with high **certainty** and low **tentativeness** (dogmatism) form **more insular clusters**?
* How do semantic and structural groupings interact to form echo chambers?

## Additional Datasets

Beyond the Reddit hyperlink network, we use:

- **Reddit User and Subreddit Embeddings (Stanford SNAP)** — [https://snap.stanford.edu/data/web-RedditEmbeddings.html]  
  Provides 300-dimensional vectors representing subreddit topics.

We process these vectors using PCA for dimensional reduction and K-Means Clustering to assign a categorical Topic Cluster ID to every subreddit.

This is the basis for our **Semantic (Topic) Layer**.

We enrich these datasets by calculating community roles based on network and sentiment metrics:

- **Critical:** high outgoing negativity  
- **Controversial:** high incoming negativity  
- **Supportive:** high outgoing positive/neutral links  
- **Influential:** high incoming positive/neutral links


**Data Note:** The raw `.tsv` and `.csv` files from SNAP are not included in this repository. To run the pipeline place them in the following directories:
* `data/hyperlink_network/soc-redditHyperlinks-body.tsv`
* `data/hyperlink_network/soc-redditHyperlinks-title.tsv`
* `data/subreddit_embeddings/web-redditEmbeddings-subreddits.csv`


## Methods

### Data & Cleaning (data_processing.py)

* We load the body and title hypernetwork (≈858k links). The PROPERTIES vector is split into 21 text attributes (e.g., readability, VADER sentiment) and 65 LIWC indicators per link. 
* We standardize subreddit names (lowercase, trimmed), drop self-loops, coerce timestamps to datetime, and create binary sentiment flags (is_positive, is_negative) plus convenience time fields (year/month). Invalid or malformed PROPERTIES rows are removed. 
* The output is **combined_hyperlinks.csv** , the foundation for all subsequent analysis.

### Linguistic / Psychological Layer (liwc_analysis.py)

* We aggregate LIWC scores per subreddit as both source and target
* We Compute interpretable composites e.g.:
    * **Emotion_Total**
    * **Negemo_Specific** = mean of Anger/Anxiety/Sad
    * **Cognitive_Total** = mean of Certain/Tentative/Insight/Discrepancy/Cause 
    * We derive **asymmetry** features (outgoing minus incoming) to capture “how I talk” vs “how others talk about me.” 
* These features feed role labels (**Critical**, **Controversial**, **Supportive**, **Influential**).
* The output is saved as `subreddit_features_{source/target}.csv`.


### Network Analysis (network_analysis.py)

* **Graph Construction:** We build a `networkx` directed graph `G` from the hyperlink data, where subreddits are nodes and aggregated hyperlinks are weighted, directed edges. The aggregated edge list is saved to `network_edges.csv`.
* **Centrality Analysis:** We compute key metrics for each node to identify influential subreddits:
    * **PageRank** to measure "prestige" or "importance."
    * **HITS** to find "hubs" (good linkers) and "authorities" (good content).
    * **Betweenness Centrality** (using `k=50` sampling) to find "bridges."
    All node metrics are saved in `network_node_metrics.csv`.
* **Community Detection:** We use the **Louvain method** on an undirected version of the graph to optimize for modularity, discovering structurally dense "communities" of subreddits. 
* The resulting assignments are saved in `network_communities.csv`.

### Semantic Layer (embedding_processing.py, topic_clustering.py)

* **PCA Dimensionality Reduction:** We load the 300-dimensional raw subreddit embeddings and apply **PCA**, reducing them to 50 components. This captures ~90% of the variance while reducing noise and improving clustering. The output is saved to `embeddings_processed.csv`.
* **K-Means Topic Clustering:** We run **K-Means clustering** (with K=40) on the 50 PCA components to group semantically similar subreddits. This assigns a "Topic Cluster ID" to each subreddit.
* **Topic Labeling:** We manually inspect the most central members of each cluster to assign a human-readable topic label (e.g., "Politics", "Gaming"). 
* The final mapping of subreddits to topic IDs and labels are saved in `embeddings_kmeans_40.csv`.


### Integration & Statistical Testing (integration.py)

* **Feature Consolidation:** We merge all previously generated datasets using the `subreddit` column as the primary key.
* **Layer Merging:** This step performs a series of joins to combine:
    * **Network Metrics** (`network_node_metrics.csv`)
    * **Community Assignments** (`network_communities.csv`)
    * **Semantic Topics** (`embeddings_kmeans_40.csv`)
    * **Psychological Profiles** (`subreddit_features_source.csv`, `subreddit_features_target.csv`)
    * **Derived Roles** (`subreddit_roles.csv`)
* **Final Dataset Creation:** The result is a single, comprehensive table, `final_dataset.csv`, where each row represents one subreddit and its complete structural, semantic, and psychological features.

### Visualization & Reporting

We use a 17-plot pipeline to tell our story. We use **bar charts** for example (`plot_top_sources`) to rank communities and **quadrant maps** (`plot_role_quadrant_map`) to define their social roles (e.g., "Critical," "Supportive").

One of our most important visual is the **`plot_semantic_structural_heatmap`**. This heatmap compares **Topics (semantic rows)** with **Network Groups (structural columns)**. A bright square on a row is the visual signature of an **echo chamber**: it proves that a specific topic cluster is also a highly insular network, linking almost exclusively to itself while dismissing other groups.

You can see all plots in the Jupyter Notebook. All visualizations are also saved as PNGs in the `results/figures/` directory.

## Repository Structure

* `/data/`            : Contains raw (ignored by .gitignore), processed, and analysis-ready data.
* `/src/`             : All Python source code, organized as modules for each step of the pipeline (processing, network analysis, clustering, etc.).
* `/results/`         : Contains generated figures (`/figures`).
* `results.ipynb`     : Contains the main Jupyter Notebook 
* `README.md`         : This file.
* `pip_requirements.txt`: All required libraries.


## Proposed Timeline (to P3)


* **Week 9:** Build 3D "Social MRI" map.
* **Week 10:** Analyze psychological asymmetry (outgoing vs. incoming LIWC).
* **Week 11:** Model topic homophily (allies) and inter-cluster interaction (antagonists).
* **Week 12:** Finalize visuals and write narratives.

## Organization Within the Team

* **Amer Lakrami — Network analysis**.
* **Hamza Barrada — Embeddings and clustering**.
* **Omar El Khyari — LIWC and roles**.
* **Omar Zakariya — Integration and visualization**.
* **Cesar Illanes — READme**.

## Questions for TAs

1.  Is it better to use the Tausczik and Pennebaker (2010) paper ([link](https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2010.pdf)) to justify and enhance our LIWC interpretations?
2.  Is manually labeling our 40 K-Means topic clusters sufficient, or is an NLP validation required?
3.  Is `k=50` sampling for betweenness centrality acceptable for P3, given the computation time?


---
