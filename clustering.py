import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("news_articles_100.csv")

# Step 2: TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Step 3: Compute cosine distance
#distance_matrix = cosine_distances(tfidf_matrix)
#distance_matrix = pairwise_distances(tfidf_matrix.toarray(), metric='euclidean')
# For cosine distance (works for TF-IDF)
distance_vector = pdist(tfidf_matrix.toarray(), metric='euclidean')

# Then use this in linkage
linkage_matrix = linkage(distance_vector, method='complete')

# Step 4: Apply hierarchical clustering
#linkage_matrix = linkage(distance_matrix, method='ward')  # try 'ward' or 'complete'

# Step 5: Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=df['title'].values, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Article Title")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Step 6: Assign clusters (e.g., 3 clusters)
df['cluster'] = fcluster(linkage_matrix, t=3, criterion='maxclust')
cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
#df['cluster'] = fcluster(linkage_matrix, t=1.5, criterion='distance')
score = silhouette_score(tfidf_matrix, df['cluster'], metric='cosine')
print(score)
# Step 7: View clusters
print(df[['title', 'category', 'cluster']])
print(cluster_labels)

