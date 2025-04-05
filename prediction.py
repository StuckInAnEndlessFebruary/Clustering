import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from feature_extraction import extract_features

def main():
    # Load files
    try:
        #selected features
        with open("selected_features.txt", "r") as f:
            selected_features = [int(line.strip()) for line in f]
        print(f"Loaded {len(selected_features)} selected features")
        
        #original training features for scaler fitting
        train_df = pd.read_csv("features.csv")
        X_train = train_df.iloc[:, selected_features].values
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Run clustering3.py first.")
        return

    # Standardize using the same scaler as training
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # use KMeans
    model = KMeans(n_clusters=6, random_state=42)
    model.fit(X_train_scaled)

    # Process test data
    test_features = []
    test_image_paths = []
    test_classes = []

    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    print("Processing test images...")
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                #use existing feature extraction function
                features_dict = extract_features(image)
                
                # Convert to list in correct order (must match training data)
                feature_order = [
                    'mean_R', 'std_R', 'mean_G', 'std_G', 'mean_B', 'std_B',
                    'mean_gray', 'var_gray', 'edge_density', 'contrast', 'homogeneity'
                ]
                features = [features_dict[key] for key in feature_order]
                
                test_features.append(features)
                test_image_paths.append(img_path)
                test_classes.append(class_name)

    if not test_features:
        print("Error: No test images were processed")
        return

    #prepare test data
    X_test = np.array(test_features)
    X_test_selected = X_test[:, selected_features]
    X_test_scaled = scaler.transform(X_test_selected)

    # Predict clusters
    test_clusters = model.predict(X_test_scaled)
    
    print(f"Predicted clusters for {len(test_clusters)} test images")

    # Save results
    results_df = pd.DataFrame({
        'image_path': test_image_paths,
        'true_class': test_classes,
        'cluster': test_clusters
    })
    results_df.to_csv("test_predictions.csv", index=False)
    print("Saved predictions to test_predictions.csv")

    # Visualization
    plot_cluster_samples(results_df, test_image_paths, test_classes, test_clusters)

def plot_cluster_samples(results_df, image_paths, classes, clusters, n_samples=10):
    plt.figure(figsize=(20, 25))
    plt.suptitle("Test Samples with Cluster Members", fontsize=16, y=1.02)

    # Randomly select samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(image_paths), n_samples, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        # Display test image
        test_img = cv2.cvtColor(cv2.imread(image_paths[idx]), cv2.COLOR_BGR2RGB)
        plt.subplot(n_samples, 6, (i-1)*6 + 1)
        plt.imshow(test_img)
        plt.title(f"Test\n{classes[idx]}\nCluster {clusters[idx]}", fontsize=8)
        plt.axis('off')
        
        # Display 5 samples from same cluster
        cluster = clusters[idx]
        cluster_members = results_df[results_df['cluster'] == cluster].sample(5, replace=True)
        
        for j, (_, row) in enumerate(cluster_members.iterrows(), 1):
            member_img = cv2.cvtColor(cv2.imread(row['image_path']), cv2.COLOR_BGR2RGB)
            plt.subplot(n_samples, 6, (i-1)*6 + j + 1)
            plt.imshow(member_img)
            plt.title(f"Cluster {cluster}\n{row['true_class']}", fontsize=8)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("test_samples_with_cluster_members.png", bbox_inches='tight')
    print("Saved visualization to test_samples_with_cluster_members.png")

if __name__ == "__main__":
    main()