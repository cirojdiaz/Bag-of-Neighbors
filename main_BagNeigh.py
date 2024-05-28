
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from example import example
from LinearBagNeighbours import LinearBagNeighbours
from PlotAndSurf import PlotAndSurf


# Define the main function for testing BagNeighbours
def main_BagNeigh():
    """
    This function is a test unit for the BagNeighbours class.
    It will compare BagNeighbours' performance against XGBoost, Random Forest and ANN and Plot them.
    @return:
    """

    NP = 2000  # Number of predictors
    blank = False  # If true, only show results around the data defined. Otherwise, go broad.

    X, Z, XX, YY, ZZ = example(3)  # Generate example data from function example(example_number=1,2 or 3)

    N = len(Z)  # Number of samples
    Nf = 2  # Number of Features (2 for the examples we are running)

    sn = int(np.sqrt(N) / (Nf + 1))  # Number of centers, typically the square root of N divided by (Nf+1)

    print(f'Using {sn} centers out of {N} samples')  # Print the number of centers being used

    # Define and Train the model
    BN = LinearBagNeighbours(NP, sn)  # Initialize LinearBagNeighbours with NP predictors and sn centers
    BN.fit(X, Z, True, 1)  # Fit the model with the data, using weights and 1 neighbor

    # Make predictions
    X_grid = np.column_stack((XX.ravel(), YY.ravel()))  # Prepare grid for predictions
    final, predictions = BN.predict(X_grid)  # Predict with the trained model
    PlotAndSurf(X, Z, XX, YY, final.reshape(XX.shape), False, ZZ, blank, 'BagNeigh')  # Plot the results

    # Create an ANN regressor
    net = MLPRegressor(hidden_layer_sizes=(20, 15), activation='relu', solver='lbfgs')  # Define ANN model
    net.fit(X, Z)  # Train the ANN model
    NET_pred = net.predict(X_grid)  # Make predictions with ANN
    PlotAndSurf(X, Z, XX, YY, NET_pred.reshape(XX.shape), False, ZZ, blank, 'ANN')  # Plot the results

    # Create XGBoost regressor
    xgbRegressor = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5)  # Define XGBoost model
    xgbRegressor.fit(X, Z)  # Train the XGBoost model
    XGB_pred = xgbRegressor.predict(X_grid)  # Make predictions with XGBoost
    PlotAndSurf(X, Z, XX, YY, XGB_pred.reshape(XX.shape), False, ZZ, blank, 'XGBoost')  # Plot the results

    # Compare to Random Forest
    numTrees = 5000  # Number of trees for Random Forest
    rfRegressor = RandomForestRegressor(n_estimators=numTrees, min_samples_leaf=5,
                                        n_jobs=-1)  # Define Random Forest model
    rfRegressor.fit(X, Z)  # Train the Random Forest model
    RF_pred = rfRegressor.predict(X_grid)  # Make predictions with Random Forest
    PlotAndSurf(X, Z, XX, YY, RF_pred.reshape(XX.shape), False, ZZ, blank, 'Random Forest')  # Plot the results


# Call the main_BagNeigh function
main_BagNeigh()
