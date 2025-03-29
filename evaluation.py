import numpy as np
import pandas as pd
from collections import defaultdict

def calculate_silhouette(X, labels):

    n = len(X)
    silhouette_scores = []
    
    for i in range(n):
        #calculate average distance to points in same cluster (a_i)
        cluster_i = labels[i]
        same_cluster = X[labels == cluster_i]
        a_i = np.mean(np.linalg.norm(X[i] - same_cluster, axis=1))
        
        #calculate average distance to nearest other cluster (b_i)
        other_clusters = [c for c in np.unique(labels) if c != cluster_i]
        b_i = np.min([
            np.mean(np.linalg.norm(X[i] - X[labels == c], axis=1))
            for c in other_clusters
        ])
        
        #calculate silhouette score for this point
        if a_i == 0 and b_i == 0:
            s_i = 0 
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)

def calculate_precision_recall_f1(true_labels, pred_labels):
    """
    Precision = how pure each cluster is (majority class)
    Recall = how well each class is captured in clusters
    F1 = harmonic mean of precision and recall
    """
    # Create mapping from clusters to true classes
    cluster_to_classes = defaultdict(list)
    for cluster, true_class in zip(pred_labels, true_labels):
        cluster_to_classes[cluster].append(true_class)
    
    # Create confusion matrix (actual vs predicted)
    confusion_matrix = pd.crosstab(pd.Series(true_labels, name='Actual'),
                                 pd.Series(pred_labels, name='Predicted'))
    
    #precision for each cluster
    precision = {}
    for cluster in confusion_matrix.columns:
        cluster_total = confusion_matrix[cluster].sum()
        if cluster_total > 0:
            precision[cluster] = confusion_matrix[cluster].max() / cluster_total
        else:
            precision[cluster] = 0
    
    #recall for each class
    recall = {}
    for true_class in confusion_matrix.index:
        class_total = confusion_matrix.loc[true_class].sum()
        if class_total > 0:
            recall[true_class] = confusion_matrix.loc[true_class].max() / class_total
        else:
            recall[true_class] = 0
    
    #f1-Score for each cluster
    f1 = {}
    for cluster in confusion_matrix.columns:
        p = precision[cluster]
        # Find main class in this cluster
        main_class = confusion_matrix[cluster].idxmax()
        r = recall[main_class]
        if (p + r) > 0:
            f1[cluster] = 2 * (p * r) / (p + r)
        else:
            f1[cluster] = 0
    
    #average of all clusters
    return {
        'precision': np.mean(list(precision.values())),
        'recall': np.mean(list(recall.values())),
        'f1_score': np.mean(list(f1.values())),
        'confusion_matrix': confusion_matrix
    }

#load clustered result
df = pd.read_csv("clustered_results.csv")
X = df.drop(columns=['label', 'cluster']).values  # Feature values
true_labels = df['label'].values  # True class labels
cluster_labels = df['cluster'].values  # Cluster assignments

#calculate evaluation 
metrics = calculate_precision_recall_f1(true_labels, cluster_labels)
silhouette = calculate_silhouette(X, cluster_labels)

# Print
print("\nClustering Evaluation Results:")
print(f"Average Precision: {metrics['precision']:.4f}")
print(f"Average Recall: {metrics['recall']:.4f}")
print(f"Average F1-Score: {metrics['f1_score']:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])

# Save
results = {
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1_score': metrics['f1_score'],
    'silhouette': silhouette,
    'confusion_matrix': metrics['confusion_matrix'].to_dict()
}

pd.DataFrame.from_dict(results, orient='index').to_csv('evaluation_results.csv')
print("\nResults saved to evaluation_results.csv")