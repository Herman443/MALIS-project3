import numpy as np


class Perceptron:
    """
    Perceptron algorithm for multi-class classification.

    Parameters:
    - alpha: Learning rate, a positive float that scales the weight updates.
    - n_classes: Number of unique classes in the dataset.
    """

    def __init__(self, alpha, n_classes):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        self.alpha = alpha
        self.n_classes = n_classes
        self.weights = None  # Weight matrix, initialized during training

    def train(self, X, y, sample_weights=None, epochs=10):
        """
        Train the perceptron on the given dataset.

        Parameters:
        - X: (numpy.ndarray) Input features of shape (n_samples, n_features).
        - y: (numpy.ndarray) Labels of shape (n_samples,).
        - sample_weights: (numpy.ndarray) Weights for each sample. If None, uniform weights are used.
        - epochs: Number of iterations over the training data.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(
            (self.n_classes, n_features)
        )  # One weight vector per class

        # Initialize sample weights if none are provided
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        # Iterate through the dataset for the given number of epochs
        for epoch in range(epochs):
            for i in range(n_samples):
                # Compute scores for each class
                scores = np.dot(self.weights, X[i])
                predicted = np.argmax(scores)  # Class with the highest score

                # Update weights if the prediction is incorrect
                if predicted != y[i]:
                    self.weights[y[i]] += (
                        self.alpha * sample_weights[i] * X[i]
                    )  # Correct class
                    self.weights[predicted] -= (
                        self.alpha * sample_weights[i] * X[i]
                    )  # Penalize incorrect class

    def predict(self, X_new):
        """
        Predict labels for new samples.

        Parameters:
        - X_new: (numpy.ndarray) Input features of shape (n_samples, n_features).

        Returns:
        - Predicted labels of shape (n_samples,).
        """
        scores = np.dot(X_new, self.weights.T)  # Compute scores for each class
        return np.argmax(scores, axis=1)  # Return the class with the highest score
