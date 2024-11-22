import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("ausClean.csv")

# Ensure columns are correctly named (if the file does not already include column headers)
data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 'label']
data['label'] = data['label'].map({-1: 0, 1: 1})

# Function to split data, run logistic regression, and collect ROC and PR metrics
def logistic_regression_with_split(train_ratio):
    # Step 1: Separate features and target variable
    X = data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']].values  # Features
    y = data['label'].values  # Target

    # Step 2: Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    # Step 3: Normalize the features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Step 4: Initialize parameters
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)  # Initialize weights to zeros
    bias = 0  # Initialize bias to zero

    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000

    # Step 5: Train logistic regression using gradient descent
    m = len(y_train)
    for iteration in range(num_iterations):
        # Linear combination: z = Xw + b
        z = np.dot(X_train, weights) + bias

        # Sigmoid function
        predictions = 1 / (1 + np.exp(-z))

        # Binary cross-entropy loss
        loss = -(1 / m) * np.sum(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))

        # Gradients
        dw = (1 / m) * np.dot(X_train.T, (predictions - y_train))
        db = (1 / m) * np.sum(predictions - y_train)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # Step 6: Evaluate the model on the test set
    z_test = np.dot(X_test, weights) + bias
    predicted_probabilities = 1 / (1 + np.exp(-z_test))

    # Compute ROC and PR metrics
    fpr, tpr, _ = roc_curve(y_test, predicted_probabilities)
    precision, recall, _ = precision_recall_curve(y_test, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    return fpr, tpr, roc_auc, precision, recall, pr_auc, train_ratio

# Collect results for different splits
splits = [0.3, 0.5, 0.7]
results = [logistic_regression_with_split(split) for split in splits]

# Plot all ROC and PR curves in a single display
plt.figure(figsize=(14, 6))

# Subplot 1: Combined ROC Curves
plt.subplot(1, 2, 1)
for fpr, tpr, roc_auc, _, _, _, split in results:
    plt.plot(fpr, tpr, label=f'{int(split * 100)}%-{int((1 - split) * 100)}% (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # Random baseline
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves')
plt.legend()

# Subplot 2: Combined PR Curves
plt.subplot(1, 2, 2)
for _, _, _, precision, recall, pr_auc, split in results:
    plt.plot(recall, precision, label=f'{int(split * 100)}%-{int((1 - split) * 100)}% (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()

# Display combined plots
plt.tight_layout()
plt.show()
