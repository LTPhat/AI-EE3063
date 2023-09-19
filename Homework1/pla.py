import numpy as np
 # Imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


def avtivate_func(x):
    return np.where(x > 0 , 1, 0)



class Perceptron():

    def __init__(self,  activate_func, learning_rate=0.01 , n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func =  activate_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted



def avtivate_func(x):
    return np.where(x > 0 , 1, 0)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class Dataset():
    def __init__(self, n_samples, cluster_std, n_features = 2, n_centers = 2, random_state = 42):
        self.n_samples = n_samples
        self.cluster_std = cluster_std
        self.X = None
        self.y = None
    def _create_dataset(self):
        self.X, self.y  = datasets.make_blobs(
        n_samples=self.n_samples,  n_features=2, centers=2, cluster_std=self.cluster_std, random_state=42)
        return self.X, self.y

    def _train_test_split(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test       
        


# Testing
if __name__ == "__main__":

    dataset_obj = Dataset(n_samples=200, cluster_std= 2)
    X, y = dataset_obj._create_dataset()
    X_train, X_test, y_train, y_test = dataset_obj._train_test_split(test_size=0.8)

    p = Perceptron(activate_func=avtivate_func, learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    plt.title("PLA in test set")
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()