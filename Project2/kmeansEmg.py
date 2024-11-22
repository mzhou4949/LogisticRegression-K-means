import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


data = pd.read_csv("CleanedEmg.csv")

data['label'] = data['label'].map({-1: 0, 1: 1})

def kmeans_clustering(X_train, y_train, k=2, max_iterations=100):
    #Initialize centroids randomly
    np.random.seed(42)
    initial_indices = np.random.choice(X_train.shape[0], k, replace=False)
    centroids = X_train[initial_indices]

    for iteration in range(max_iterations):
        # Assign clusters based on nearest centroid
        distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X_train])
        clusters = np.argmin(distances, axis=1)  # Assign to the closest centroid

        #Update centroids
        new_centroids = np.array([X_train[clusters == i].mean(axis=0) for i in range(k)])

        #Convergence Check
        if np.all(centroids == new_centroids):
            print(f"Converged after {iteration} iterations.")
            break

        centroids = new_centroids  # Update centroids for the next iteration

    # Align clusters with ground truth labels
    accuracy = align_and_compute_accuracy(y_train, clusters, k)
    return accuracy

# Function to align clusters with labels and compute accuracy
def align_and_compute_accuracy(y_true, cluster_labels, k):
    # Build contingency matrix
    contingency_matrix = np.zeros((k, k), dtype=int)
    for i in range(len(y_true)):
        contingency_matrix[cluster_labels[i], y_true[i]] += 1

    # Solve the assignment problem to align clusters with labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Create mapping from clusters to true labels
    cluster_to_label_mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Map cluster labels to true labels
    aligned_labels = np.array([cluster_to_label_mapping[label] for label in cluster_labels])

    # Compute accuracy
    accuracy = accuracy_score(y_true, aligned_labels)
    return accuracy

# Split ratios for training and testing
split_ratios = [0.3, 0.5, 0.7]

for split in split_ratios:
    print(f"\nRunning K-Means with {int(split * 100)}% Training and {int((1 - split) * 100)}% Testing:")

    # Split the data
    X = data[['1', '2', '3', '4', '5', '6', '7', '8']].values  # Features
    y = data['label'].values  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

    # Run K-Means on the training data
    accuracy = kmeans_clustering(X_train, y_train, k=2, max_iterations=100)
    print(f"K-Means Accuracy on {int(split * 100)}% Training: {accuracy:.2f}")
