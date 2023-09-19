from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
wine = pd.read_csv("./wine.csv")

# Print columns
print(wine.columns)

# Standardize features
scaler = StandardScaler()
scaler.fit(wine.drop('Wine', axis=1))
scaler_features = scaler.transform(wine.drop('Wine', axis=1))
wine_feat = pd.DataFrame(scaler_features, columns=wine.columns[1:])
print(wine_feat.head())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(scaler_features, wine['Wine'], test_size=0.3, random_state=103)


# Define KNN function
def knn(X_train, Y_train, X_test, k):
    n_test = X_test.shape[0]
    predictions = np.empty(n_test)

    for i in range(n_test):
        # Compute Euclidean distances between the test point and all training points
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))

        # Get indices of k nearest neighbors
        nearest_neighbors = np.argsort(distances)[:k]

        # Get the most common class among the k nearest neighbors
        predictions[i] = np.bincount(Y_train.iloc[nearest_neighbors.tolist()]).argmax()

    return predictions

# Test the model with different k values
error_rate = []
for i in range(1, 40):
    pred_i = knn(X_train, Y_train, X_test, k=i)
    error_rate.append(np.mean(pred_i != Y_test))

# Plot error rate vs. K values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs K values')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Choose the best K (let's say k=7)
best_k = 7
pred = knn(X_train, Y_train, X_test, best_k)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(Y_test, pred))
print("\nClassification Report:")
print(classification_report(Y_test, pred))
print("Accuracy Score:", accuracy_score(Y_test, pred)) # Accuracy= (True Positives+True Negatives) / Total Predictions

from matplotlib.colors import ListedColormap

# Choose two features for visualization, e.g., the first two features
feature1_index = 0  # Index of the first feature
feature2_index = 1  # Index of the second feature

# Define the meshgrid based on the selected features
x_min, x_max = X_train[:, feature1_index].min() - 1, X_train[:, feature1_index].max() + 1
y_min, y_max = X_train[:, feature2_index].min() - 1, X_train[:, feature2_index].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Train the KNN classifier on the selected features
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train[:, [feature1_index, feature2_index]], Y_train)

# Predict for each point in the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'green', 'blue')))
plt.scatter(X_train[:, feature1_index], X_train[:, feature2_index], c=Y_train, cmap=ListedColormap(('red', 'green', 'blue')), edgecolor='k', s=20)
plt.xlabel('alcohol {}'.format(feature1_index + 1))
plt.ylabel('Malic.acid {}'.format(feature2_index + 1))
plt.title('KNN Decision Boundaries (k=7)')
plt.show()
