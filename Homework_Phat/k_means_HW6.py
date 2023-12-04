import pandas as pd

# Apply K-means algorithm for Mall_Customer task

# Đọc dữ liệu
df = pd.read_csv(".\data\Mall_Customers.csv")
features = ["CustomerID", "Gender", "Age", "Annual Income", "Spending Score"]

# Infor
print(df.info())


# Label encoder non-numerical feature
from sklearn.preprocessing import  LabelEncoder

label_encoder = LabelEncoder()
df.Gender = label_encoder.fit_transform(df.Gender)
print("Data after label encoder: ", df)

# Data scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age','Annual Income','Spending Score']] = scaler.fit_transform(df[['Age','Annual Income','Spending Score']])

print("Data after scaling:\n ", df)

from sklearn.cluster import KMeans

# Apply K-means (k = 4)
model_k_means = KMeans(n_clusters= 4)
model_k_means.fit(df)
df['cluster'] = model_k_means.labels_

print("Data after apply K-means, at cluster\n", df)

# Plot result


# Plot 2-d clusters (Choose 2 feature)
import matplotlib.pyplot as plt

def plot_cluster(df, feature1, feature2):
    """
    Plot 2-d clusters from two random feature of dataset
    """
    for i in range(len(df["cluster"].unique())):
        cluster_i = df[df["cluster"] == i]
        plt.scatter(cluster_i[feature1], cluster_i[feature2], label = "group {}".format(i + 1))

    plt.title('K-means cluster result for {} - {}'.format(feature1, feature2))
    plt.ylabel(feature2)
    plt.xlabel(feature1)
    plt.legend()
    plt.axis()
    plt.show()

features = ["CustomerID", "Gender", "Age", "Annual Income", "Spending Score"]

# Plot cluster for feature CustomerID - Gender
plot_cluster(df, feature1=features[0], feature2=features[1])

# Plot cluster for feature "Annual Income" -"Spending Score"
plot_cluster(df, feature1= features[3], feature2=features[4])

# Plot cluster for feature  "Age" - "Annual Income"
plot_cluster(df, feature1= features[2], feature2=features[3])

# Plot cluster for feature "Age" - "Spending Score"
plot_cluster(df, feature1= features[2], feature2=features[4])