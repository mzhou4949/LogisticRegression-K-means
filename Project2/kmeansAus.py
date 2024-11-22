import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("ausClean.csv")

data['label'] = data['label'].map({-1: 0, 1: 1})

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

        #Convergence check
        if np.all(centroids == new_centroids):
            print(f"Converged after {iteration} iterations.")
            break

        centroids = new_centroids  #Update centroids for the next iteration

    # Align clusters with ground truth labels 
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_points = y_train[clusters == i]
        cluster_labels[i] = 1 if np.sum(cluster_points) > len(cluster_points) / 2 else 0

    # Map predicted clusters to ground truth labels
    predicted_labels = np.array([cluster_labels[cluster] for cluster in clusters])

    # Compute accuracy
    accuracy = accuracy_score(y_train, predicted_labels)
    return accuracy

# Define train-test split ratios
split_ratios = [0.3, 0.5, 0.7]

# Loop through each train-test split ratio
for split in split_ratios:
    print(f"\nRunning K-Means with {int(split * 100)}% Training and {int((1 - split) * 100)}% Testing:")

    # Split the data
    X = data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']].values  # Features
    y = data['label'].values  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

    # Run K-Means on the training data
    accuracy = kmeans_clustering(X_train, y_train, k=2, max_iterations=100)
    print(f"K-Means Accuracy on {int(split * 100)}% Training: {accuracy:.2f}")
