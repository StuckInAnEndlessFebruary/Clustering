import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# load selected features
selected_features = []
with open("selected_features.txt", "r") as file:
    for line in file:
        selected_features.append(int(line.strip()))

#load feature data
df = pd.read_csv("features.csv")

#select only the numeric columns (selected features)
X = df.iloc[:, selected_features].values

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  evaluate clustering using Silhouette Score
def evaluate_clustering(X, labels):
    if len(np.unique(labels)) > 1:  #need at least 2 clusters to calculate
        return silhouette_score(X, labels)
    return -1  

# KMeans
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = evaluate_clustering(X_scaled, kmeans_labels)
print(f"KMeans Silhouette Score: {kmeans_score}")

# Agglomerative
agglo = AgglomerativeClustering(n_clusters=6)
agglo_labels = agglo.fit_predict(X_scaled)
agglo_score = evaluate_clustering(X_scaled, agglo_labels)
print(f"Agglomerative Silhouette Score: {agglo_score}")

#DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_score = evaluate_clustering(X_scaled, dbscan_labels)
print(f"DBSCAN Silhouette Score: {dbscan_score}")

#MeanShift
meanshift = MeanShift()
meanshift_labels = meanshift.fit_predict(X_scaled)
meanshift_score = evaluate_clustering(X_scaled, meanshift_labels)
print(f"MeanShift Silhouette Score: {meanshift_score}")

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
plt.show()


df.to_csv("clustered_data.csv", index=False)
