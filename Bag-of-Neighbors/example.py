import numpy as np  # Import the numpy library for numerical operations

# Define the example function
def example(ex):
    """
    Generate and process features based on the example number.
    And creates a Gaussian example for the unit test to run
    @param ex: Example number (1, 2, or 3)
    @return: Processed features and corresponding outputs
    """

    feat = features(ex)  # Generate features based on the example number
    x, y = feat[:, 0], feat[:, 1]  # Extract x and y coordinates from features
    X = np.column_stack((x, y))  # Combine x and y into a single array

    mean = np.mean(X, axis=0)  # Calculate the mean of the features
    sigma_inv = np.linalg.inv(np.diag([1, 1]))  # Invert the diagonal covariance matrix

    # Define a Gaussian function
    def gaussian(X):
        diff = X - mean  # Calculate the difference from the mean
        exponents = -np.sum(np.dot(diff, sigma_inv) * diff, axis=1)  # Calculate the exponents
        return np.exp(exponents)  # Return the exponentiated values

    z = gaussian(feat) + 0.02 * np.random.randn(feat.shape[0])  # Generate Gaussian values with noise

    x_range = np.linspace(np.min(x), np.max(x), 100)  # Create a range of x values
    y_range = np.linspace(np.min(y), np.max(y), 100)  # Create a range of y values
    XX, YY = np.meshgrid(x_range, y_range)  # Create a meshgrid from x and y ranges
    Z = gaussian(np.column_stack((XX.ravel(), YY.ravel())))  # Generate Gaussian values for the grid
    ZZ = Z.reshape(XX.shape)  # Reshape the Gaussian values to match the grid

    return X, z, XX, YY, ZZ  # Return the processed features and corresponding outputs

# Define the features function
def features(ex):
    """
    Generate features based on the example number.
    @param ex: Example number (1, 2, or 3)
    @return: Generated features
    """

    if ex == 1:
        t = np.linspace(0, 1, 100)  # Create a linear space from 0 to 1
        X, Y = np.sin(np.pi * t), t  # Generate sinusoidal x values and linear y values
        covariance_matrix = [[0.002, 0], [0, 0.003]]  # Define the covariance matrix
        num_points_per_center = 10  # Define the number of points per center

    elif ex == 2:
        t = np.linspace(0, 1, 100)  # Create a linear space from 0 to 1
        X, Y = np.sin(2 * np.pi * t), t  # Generate sinusoidal x values and linear y values
        rotation_matrix = np.array([[np.sin(np.pi / 6), np.cos(np.pi / 6)],
                                    [np.cos(np.pi / 6), -np.sin(np.pi / 6)]])  # Define the rotation matrix
        X, Y = np.dot(rotation_matrix, np.vstack((X, Y)))  # Apply the rotation matrix to x and y values
        covariance_matrix = [[0.001, 0], [0, 0.001]]  # Define the covariance matrix
        num_points_per_center = 10  # Define the number of points per center

    elif ex == 3:
        t = np.linspace(0, 1, 10)  # Create a linear space from 0 to 1
        X, Y = np.sin(2 * np.pi * t), t  # Generate sinusoidal x values and linear y values
        rotation_matrix = np.array([[np.sin(np.pi / 6), np.cos(np.pi / 6)],
                                    [np.cos(np.pi / 6), -np.sin(np.pi / 6)]])  # Define the rotation matrix
        X, Y = np.dot(rotation_matrix, np.vstack((X, Y)))  # Apply the rotation matrix to x and y values
        covariance_matrix = [[0.001, 0], [0, 0.001]]  # Define the covariance matrix
        num_points_per_center = 100  # Define the number of points per center
    else:
        raise ValueError("Example number must be 1, 2, or 3")  # Raise an exception if the example number is invalid


    centers = np.column_stack((X, Y))  # Combine x and y values into center points
    generated_points = np.vstack([np.random.multivariate_normal(center, covariance_matrix, num_points_per_center)
                                  for center in centers])  # Generate points around each center
    generated_points = np.unique(generated_points, axis=0)  # Remove duplicate points

    return generated_points  # Return the generated points
