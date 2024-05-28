import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# Define the PlotAndSurf function
def PlotAndSurf(X, z, XX, YY, ZZ, animation=False, Z_true=None, blank=False, title=''):
    global neighborhood_radius  # Use a global variable for neighborhood radius
    fig = plt.figure(figsize=(8, 6))  # Create a new figure with specified size
    ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure
    ax.scatter(X[:, 0], X[:, 1], z, c='r', marker='.', label='Grid Points')  # Scatter plot of the data points
    ax.set_xlabel('X')  # Set the x-axis label
    ax.set_ylabel('Y')  # Set the y-axis label
    ax.set_zlabel('Z')  # Set the z-axis label
    ax.set_title('Grid Points and Surface Plot' if not title else title)  # Set the plot title
    ax.grid(True)  # Enable the grid

    if blank:  # If blank is True, remove points that are not neighbors
        neighborhood_radius = 0.05  # Set the neighborhood radius
        for i in range(len(XX.ravel())):  # Iterate through all points in the grid
            current_point = np.array([XX.ravel()[i], YY.ravel()[i]])  # Get the current point
            is_neighbor = np.any(np.linalg.norm(X - current_point, axis=1) <= neighborhood_radius)  # Check if it has neighbors
            if not is_neighbor:  # If it has no neighbors
                ZZ.ravel()[i] = np.nan  # Set the corresponding Z value to NaN

    norm = Normalize(vmin=ZZ.min(), vmax=ZZ.max())  # Normalize the ZZ values for color mapping
    colors = plt.cm.viridis(norm(ZZ))  # Get colors based on the normalized ZZ values
    ax.plot_surface(XX, YY, ZZ, cmap='viridis', alpha=0.2, facecolors=colors, rstride=1, cstride=1)  # Plot the surface

    if Z_true is not None:  # If Z_true is provided
        if blank:  # If blank is True, remove points that are not neighbors
            for i in range(len(XX.ravel())):  # Iterate through all points in the grid
                current_point = np.array([XX.ravel()[i], YY.ravel()[i]])  # Get the current point
                is_neighbor = np.any(np.linalg.norm(X - current_point, axis=1) <= neighborhood_radius)  # Check if it has neighbors
                if not is_neighbor:  # If it has no neighbors
                    Z_true.ravel()[i] = np.nan  # Set the corresponding Z_true value to NaN
        norm = Normalize(vmin=Z_true.min(), vmax=Z_true.max())  # Normalize the Z_true values for color mapping
        colors = plt.cm.viridis(norm(Z_true))  # Get colors based on the normalized Z_true values
        ax.plot_surface(XX, YY, Z_true, cmap='viridis', alpha=0.1, facecolors=colors, rstride=1, cstride=1)  # Plot the true surface

    ax.view_init(azim=30, elev=20)  # Set the initial view angle

    if animation:  # If animation is True, create an animation
        def update_angle(num, ax):  # Define the update function for the animation
            ax.view_init(azim=num, elev=20)  # Update the azimuth angle
            return ax  # Return the updated axes

        ani = FuncAnimation(fig, update_angle, frames=range(0, 91), fargs=(ax,), blit=False)  # Create the animation
        ani.save('Constant_Model.mp4', writer='ffmpeg', dpi=300)  # Save the animation as a video

    plt.show()  # Display the plot
