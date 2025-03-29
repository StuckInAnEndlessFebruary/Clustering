import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
selected_features = []
with open("selected_features.txt", "r") as file:
    for line in file:
        selected_features.append(int(line.strip()))

df = pd.read_csv("features.csv")
X = df.iloc[:, selected_features]
feature_names = [df.columns[i] for i in selected_features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluation function
def evaluate_clustering(X, labels):
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    return -1

k = 6  # Number of clusters

# Create figure for plots
plt.figure(figsize=(18, 12))

# 1. KMeans
plt.subplot(2, 2, 1)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = evaluate_clustering(X_scaled, kmeans_labels)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title(f'KMeans (Score: {kmeans_score:.3f})')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# 2. Agglomerative
plt.subplot(2, 2, 2)
agglo = AgglomerativeClustering(n_clusters=k)
agglo_labels = agglo.fit_predict(X_scaled)
agglo_score = evaluate_clustering(X_scaled, agglo_labels)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_labels, cmap='plasma')
plt.title(f'Agglomerative (Score: {agglo_score:.3f})')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# 3. DBSCAN
plt.subplot(2, 2, 3)
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_score = evaluate_clustering(X_scaled, dbscan_labels)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='cool')
plt.title(f'DBSCAN (Score: {dbscan_score:.3f})')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# 4. MeanShift
plt.subplot(2, 2, 4)
meanshift = MeanShift(bandwidth=0.5)
meanshift_labels = meanshift.fit_predict(X_scaled)
meanshift_score = evaluate_clustering(X_scaled, meanshift_labels)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=meanshift_labels, cmap='spring')
plt.title(f'MeanShift (Score: {meanshift_score:.3f})')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

plt.tight_layout()
plt.show()

# Print scores
print(f"\nKMeans Silhouette Score: {kmeans_score:.3f}")
print(f"Agglomerative Silhouette Score: {agglo_score:.3f}")
print(f"DBSCAN Silhouette Score: {dbscan_score:.3f}")
print(f"MeanShift Silhouette Score: {meanshift_score:.3f}")

# Save results with KMeans clusters (as example)
df['Cluster'] = kmeans_labels

print("\nResults saved to clustered_results.csv")
#finding best algorithm
best_algorithm = None
best_score = -1
best_labels = None

if kmeans_score > best_score:
    best_algorithm = "KMeans"
    best_score = kmeans_score
    best_labels = kmeans_labels

if agglo_score > best_score:
    best_algorithm = "Agglomerative"
    best_score = agglo_score
    best_labels = agglo_labels

if dbscan_score > best_score:
    best_algorithm = "DBSCAN"
    best_score = dbscan_score
    best_labels = dbscan_labels

if meanshift_score > best_score:
    best_algorithm = "MeanShift"
    best_score = meanshift_score
    best_labels = meanshift_labels

print(f"Best Algorithm: {best_algorithm} with Silhouette Score: {best_score}")



df['cluster'] = best_labels


cluster_means = df.groupby('cluster').mean(numeric_only=True)

#heatmap to visualize the mean features of each cluster
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of Cluster Features")
plt.savefig("cluster_heatmap.png")
plt.show()


df.to_csv("clustered_results.csv", index=False)
