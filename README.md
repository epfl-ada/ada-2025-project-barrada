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
* Do **ideological neighbors** criticize each other more than distant communities ("narcissism of small differences")?
* How do **topic clusters** intersect with **network communities** (Louvain): alignment vs. cross‑topic corridors of support/attack?

### **Polarization Mechanisms**

* Do communities with high **certainty** and low **tentativeness** (dogmatism) form **more insular clusters**?
* Are **emotionally extreme** communities (high anger/anxiety) more or less likely to **engage across ideological boundaries**?

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


## Methods

### Data & Cleaning

We load the body and title hypernetwork (≈858k links). The PROPERTIES vector is split into 21 text attributes (e.g., readability, VADER sentiment) and 65 LIWC indicators per link. We standardize subreddit names (lowercase, trimmed), drop self-loops, coerce timestamps to datetime, and create binary sentiment flags (is_positive, is_negative) plus convenience time fields (year/month). Invalid or malformed PROPERTIES rows are removed.

### Network Layer (structure & communities)

- Construct directed weighted graph (nodes=subreddits, edges=aggregated links)
- Compute centrality metrics: PageRank, HITS (hubs/authorities), betweenness (sampled for computational efficiency), clustering coefficient
- Detect communities using Louvain algorithm on undirected projection
- Identify positive-only and negative-only subgraphs for sentiment-specific analysis

### Linguistic / Psychological Layer (LIWC aggregation)

- Aggregate LIWC scores per subreddit as both source (outgoing links) and target (incoming links)
- We also compute interpretable composites (e.g., Emotion_Total, Negemo_Specific = mean of Anger/Anxiety/Sad; Cognitive_Total = mean of Certain/Tentative/Insight/Discrepancy/Cause). We derive asymmetry features (outgoing minus incoming) to capture “how I talk” vs “how others talk about me.” These features feed role labels (Critical, Controversial, Supportive, Influential).


### Semantic Layer (topics & distances)

- We load 300-d subreddit embeddings, de-duplicate, validate, standardize, then apply PCA → 50 (we track explained variance and keep ≥~90%). 
- On PCA we run K-Means with K=40 and we sssign human-readable labels to clusters through manual inspection of top members and compute cluster quality metrics.

### Integration & Statistical Testing

We left-join all layers on subreddit to form final_dataset.csv, containing:

- Structure: centralities, degrees, community, modularity context
- Psychology: LIWC aggregates (source/target), composite indices, asymmetries, role flags
- Semantics: PCA coordinates, Topic Cluster ID


### Visualization & Reporting




## Proposed Timeline (to P3)

**Week 9:**



**Week 10:**



**Week 11:**


**Week 12 (P3 polish):**

* Refine narrative, finalize figures, write‑up.

## Organization Within the Team

* **Amer Lakrami — Network analysis**.
* **Hamza Barrada — Embeddings & clustering**.
* **Omar El Khyari — LIWC & roles**.
* **Omar Zakariya — Integration & viz**.
* **Cesar Illanes — QA & writing**.


## Questions for TAs (optional)

1. naming topic clusters.
2. For betweenness on large graphs, is node‑sampling (k=50–200) acceptable for P3 if we report the sampling scheme?

---
