import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Load clustered data
# df = pd.read_csv("clustered_data.csv")
df = pd.read_csv("clustered_results.csv")

labels = df["cluster"].values

# Select only feature columns (excluding label and cluster columns)
feature_columns = [col for col in df.columns if col not in ["label", "cluster"]]
X = df[feature_columns].values

# Reduce dimensions using PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis", alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of Clusters")
plt.legend(title="Clusters")
plt.show()

# Reduce dimensions using PCA (3D)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

# Plot 3D PCA visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=labels, cmap="viridis", alpha=0.7)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("3D PCA Visualization of Clusters")
plt.colorbar(scatter, ax=ax, label="Cluster Labels")
plt.show()

# Reduce dimensions using t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="coolwarm", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Clusters")
plt.legend(title="Clusters")
plt.show()