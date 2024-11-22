import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('adultCleaned.csv' )

# Separate features and labels
X = data.drop(columns=['income']).values  # Features
y = data['income'].values  # Ground truth labels

def kmeans_clustering(X_train, y_train, k=2, max_iterations=100):
    #Initialize centroids randomly
    np.random.seed(42)
    initial_indices = np.random.choice(X_train.shape[0], k, replace=False)
    centroids = X_train[initial_indices]

    #K-Means Algorithm
    for iteration in range(max_iterations):
        #Assign clusters based on nearest centroid
        distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X_train])
        clusters = np.argmin(distances, axis=1)  # Assign to the closest centroid

        #Update centroids
        new_centroids = np.array([X_train[clusters == i].mean(axis=0) for i in range(k)])

        #Check for convergence
        if np.all(centroids == new_centroids):
            print(f"Converged after {iteration} iterations.")
            break

        centroids = new_centroids  # Update centroids for the next iteration

    #Align clusters with labels using majority voting
    cluster_labels_map = {}
    for cluster in range(k):
        cluster_points = y_train[clusters == cluster]
        if len(cluster_points) > 0:
            cluster_labels_map[cluster] = 1 if np.sum(cluster_points) > len(cluster_points) / 2 else 0
        else:
            cluster_labels_map[cluster] = 0  # Default to class 0 if the cluster is empty

    # Map clusters to predicted labels
    predicted_labels = np.array([cluster_labels_map[label] for label in clusters])

    return accuracy_score(y_train, predicted_labels)

split_ratios = [0.3, 0.5, 0.7]

for split in split_ratios:
    print(f"\nRunning K-Means with {int(split * 100)}% Training and {int((1 - split) * 100)}% Testing:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

    accuracy = kmeans_clustering(X_train, y_train, k=2, max_iterations=100)
    print(f"K-Means Accuracy on {int(split * 100)}% Training: {accuracy:.2f}")
