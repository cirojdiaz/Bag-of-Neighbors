"""
This class creates a regressor that combines local linear approximations
in a bagging process to create an ensemble of it.
The local neighborhoods were local approximation occur are defined after
subsampling the date into centers and finding their neighbors.
"""

# Import necessary libraries
import numpy as np
from sklearn.neighbors import KDTree

# Define the LinearBagNeighbours class
class LinearBagNeighbours:
    # Initialize the class with the number of rounds and number of centers
    def __init__(self, numOfRounds, numOfCenters):
        self.numOfRounds = numOfRounds  # Store the number of rounds
        self.numOfCenters = numOfCenters  # Store the number of centers
        self.predictors = []  # Initialize the list to store predictors

    # Define the fit method to train the model
    def fit(self, X, y, weight=False, numOfNeighbors=10):

        """
        Weight the probability of a sample being selected as a center
        depending on how populated its neighborhood is. If no weighting is chosen,
        then all weights will be evenly set to one
        """

        self.predictors = []  # Clear any existing predictors

        if not weight:
            weights = np.ones(X.shape[0])  # Set equal weights if no weighting is chosen
        else:
            if numOfNeighbors is None:
                numOfNeighbors = 10  # Set the default number of neighbors if not provided
            kdTree = KDTree(X)  # Create a KDTree for the data
            _, weights = kdTree.query(X, k=numOfNeighbors + 1)  # Query the KDTree for neighbors
            weights = np.max(weights[:, 1:], axis=1)  # Use maximum distance to the neighbors as weights
            weights = weights.astype(np.float64)  # Change the data type of the weights to float64
            weights /= np.sum(weights)  # Normalize the weights

        for k in range(self.numOfRounds):
            if not weight:
                random_indices = np.random.permutation(X.shape[0])[:self.numOfCenters]  # Randomly select centers
            else:
                random_indices = np.random.choice(X.shape[0], self.numOfCenters, replace=False, p=weights)  # Select centers based on weights

            centers = X[random_indices, :]  # Get the center points

            kdTree = KDTree(centers)  # Create a KDTree for the centers
            nearestNeighborIdx = kdTree.query(X, k=1)[1]  # Find the nearest center for each point in X

            predictors_round = []  # Initialize a list for predictors of this round
            for i in range(self.numOfCenters):
                indices = (nearestNeighborIdx == i).squeeze()  # Get indices of points nearest to center i
                M = np.concatenate((X[indices, :], np.ones((np.sum(indices), 1))), axis=1)  # Create design matrix
                b = y[indices]  # Get target values for these points

                if b.shape[0] ** 2 < X.shape[1] + 1:
                    """
                    Going constant if there are not enough elements in the neighbourhood
                    """
                    M = M[:, -1:]  # Use only the bias term if not enough points
                    params = np.concatenate((np.zeros(X.shape[1]), np.linalg.lstsq(M, b, rcond=None)[0]), axis=0)  # Fit constant model
                else:
                    """
                    Going linear if there are enough elements in the neighbourhood
                    """
                    params = np.linalg.lstsq(M, b, rcond=None)[0]  # Fit linear model

                predictors_round.append(np.concatenate((centers[i, :], params)))  # Store the center and parameters

            self.predictors.append(predictors_round)  # Add this round's predictors to the list

    # Define the predict method to make predictions
    def predict(self, X):
        predictions = np.zeros((self.numOfRounds, X.shape[0]))  # Initialize an array to store predictions

        for k in range(self.numOfRounds):
            predictors_round = np.array(self.predictors[k])  # Get predictors for this round
            centers = predictors_round[:, :X.shape[1]]  # Extract centers
            params = predictors_round[:, X.shape[1]:]  # Extract parameters

            kdTree = KDTree(centers)  # Create KDTree for centers
            nearestNeighborIdx = kdTree.query(X, k=1)[1]  # Find the nearest center for each point in X
            params = params[nearestNeighborIdx.flatten()]  # Get parameters for these centers

            means = np.sum(np.concatenate((params[:, :-1] * X, params[:, -1:]), axis=1), axis=1)  # Compute predictions
            predictions[k, :] = means  # Store predictions for this round

        final = np.sum(predictions, axis=0) / self.numOfRounds  # Average predictions across rounds
        return final, predictions  # Return final predictions and all round predictions
