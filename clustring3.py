import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load and prepare data
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

k = 6  # Number of clusters for parametric algorithms

# Define clustering algorithms
algorithms = {
    'KMeans': KMeans(n_clusters=k, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=k),
    'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
    'MeanShift': MeanShift(bandwidth=0.5)
}

# Run all clustering algorithms
results = []
for name, algorithm in algorithms.items():
    labels = algorithm.fit_predict(X_scaled)
    score = evaluate_clustering(X_scaled, labels)
    n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise for DBSCAN
    results.append({
        'name': name,
        'labels': labels,
        'score': score,
        'n_clusters': n_clusters
    })

# Find best algorithm
best_result = max(results, key=lambda x: x['score'])
df['cluster'] = best_result['labels']  # Save best clusters to dataframe

# Visualization - All results in one frame
plt.figure(figsize=(20, 15))

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for i, result in enumerate(results, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                   hue=result['labels'], palette="viridis", alpha=0.7)
    plt.title(f"{result['name']}\nSilhouette: {result['score']:.3f}, Clusters: {result['n_clusters']}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")

plt.tight_layout()
plt.savefig("all_clustering_results.png")
plt.show()

# Visualization - Best result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
               hue=best_result['labels'], palette="viridis", alpha=0.7)
plt.title(f"Best Algorithm: {best_result['name']}\nSilhouette: {best_result['score']:.3f}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("best_clustering_result.png")
plt.show()

# Cluster means heatmap
cluster_means = df.groupby('cluster').mean(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Cluster Characteristics Heatmap")
plt.savefig("cluster_characteristics.png")
plt.show()

# Save results
df.to_csv("clustered_results.csv", index=False)

# Print summary
print("\nClustering Algorithm Comparison:")
print(pd.DataFrame({
    'Algorithm': [r['name'] for r in results],
    'Silhouette Score': [r['score'] for r in results],
    'Number of Clusters': [r['n_clusters'] for r in results]
}).to_markdown(index=False))

print(f"\nBest Algorithm: {best_result['name']}")
print(f"Silhouette Score: {best_result['score']:.3f}")
print(f"Number of Clusters: {best_result['n_clusters']}")
print("\nResults saved to:")
print("- clustered_results.csv")
print("- all_clustering_results.png")
print("- best_clustering_result.png")
print("- cluster_characteristics.png")