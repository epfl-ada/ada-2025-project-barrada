# Reddit’s Invisible Brain: The Web Hidden Beneath the Threads

**Mapping the emotional and cognitive architecture of online communities**

---

## Abstract


Reddit is not just a collection of online forums; it is a living ecosystem of interconnected communities, each with its own emotional and cognitive identity. Through hyperlinks, subreddits comment on, critique, and reference one another, forming a complex web of inter-community relationships.

This project aims to uncover the hidden psychological structure underlying that web by integrating three perspectives:

- **Structural:** Who talks about whom? (Network topology, centrality, community structure)  
- **Psychological (LIWC):** How do they talk? (Emotional tone, cognitive style, "we" vs "they" linguistics)  
- **Semantic:** What are they about? (Topic embeddings)

By combining these layers, we seek to visualize Reddit as a “social MRI”, revealing how communities align, clash, or coexist—not only by what they discuss but how they think and feel. Our goal is to tell the story of Reddit’s invisible brain: the emotional and cognitive currents that shape the flow of ideas across its digital landscape.

---

## Additional Datasets

Beyond the Reddit hyperlink network, we use:

- **Reddit User and Subreddit Embeddings (Stanford SNAP)** — [https://snap.stanford.edu/data/web-RedditEmbeddings.html]  
  Provides 300-dimensional vectors representing subreddit topics.

We process these vectors using PCA for dimensional reduction (to 50 components) and K-Means Clustering (K=40) to assign a categorical Topic Cluster ID to every subreddit.

This is the basis for our **Semantic (Topic) Layer**.

We enrich these datasets by calculating community roles based on network and sentiment metrics:

- **Critical/Critic:** high outgoing negativity  
- **Controversial/Target:** high incoming negativity  
- **Supportive/Champion:** high number of outgoing positive/neutral links  
- **Useful/Amplifier:** high incoming positive/neutral links

---

## Research Questions

We structure our investigation around four interrelated themes:

1. **Network Structure:**  
   - Do communities with specific psychological profiles (e.g., high anger) occupy different positions in the network (central vs. peripheral)?  
   - Are "analytical" communities (high cognitive complexity) more likely to serve as bridges between disparate clusters?

2. **Emotional Geography:**  
   - Which emotions (anger, anxiety, certainty) dominate across negative vs. positive links?  
   - Do "critic" communities (high outgoing negativity) exhibit depression markers (high "I" usage + negative emotion)?

3. **Social Dynamics:**  
   - Are positive links more common between psychologically or topically similar subreddits?  
   - Do ideological “neighbors” criticize each other more than distant communities?

4. **Polarization Mechanisms:**  
   - Do communities with high certainty and low tentativeness (dogmatism) form more insular clusters?  
   - Are emotionally extreme communities (high anger or anxiety) more or less likely to engage across ideological boundaries?

---
