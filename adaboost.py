import numpy as np


class AdaBoostSAMME:
    """
    AdaBoost implementation for multi-class classification using SAMME algorithm.

    Parameters:
    - base_learner_class: The base classifier class to use (e.g., Perceptron).
    - n_classes: Number of unique classes in the dataset.
    - n_estimators: Number of boosting rounds (iterations).
    """

    def __init__(self, base_learner_class, n_classes, n_estimators=10):
        self.base_learner_class = base_learner_class
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learners = []  # List to store the trained weak learners
        self.alphas = (
            []
        )  # List to store the alpha (importance) values for each weak learner

    def train(self, X, y):
        """
        Train the AdaBoost model on the given dataset.

        Parameters:
        - X: (numpy.ndarray) Input features of shape (n_samples, n_features).
        - y: (numpy.ndarray) Labels of shape (n_samples,).
        """
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples  # Initialize sample weights uniformly

        for _ in range(self.n_estimators):
            # Train a new base learner with weighted samples
            learner = self.base_learner_class(alpha=0.0001, n_classes=self.n_classes)
            learner.train(X, y, sample_weights=weights)
            predictions = learner.predict(X)

            # Compute the weighted error rate
            incorrect = predictions != y
            err = np.dot(weights, incorrect) / np.sum(weights)

            # Compute alpha (importance of the weak learner)
            alpha = np.log((1 - err) / max(err, 1e-10)) + np.log(self.n_classes - 1)
            self.learners.append(learner)
            self.alphas.append(alpha)

            # Update sample weights: Increase weights for misclassified samples
            weights *= np.exp(alpha * incorrect)
            weights /= np.sum(weights)  # Normalize weights

    def predict(self, X):
        """
        Predict labels for new samples using the trained AdaBoost model.

        Parameters:
        - X: (numpy.ndarray) Input features of shape (n_samples, n_features).

        Returns:
        - Predicted labels of shape (n_samples,).
        """
        final_scores = np.zeros(
            (X.shape[0], self.n_classes)
        )  # Initialize scores for each class
        for alpha, learner in zip(self.alphas, self.learners):
            predictions = learner.predict(X)
            for i, pred in enumerate(predictions):
                final_scores[i, pred] += alpha  # Add alpha to the predicted class
        return np.argmax(
            final_scores, axis=1
        )  # Return the class with the highest aggregated score
