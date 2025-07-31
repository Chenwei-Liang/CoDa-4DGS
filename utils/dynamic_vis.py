import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def plot_colored_points_with_speed(means3D_final, dx, filename="colored_points_with_speed.png", sample_ratio=0.1):
    """
    Plots 3D points with colors representing the speed magnitude.

    Parameters:
        means3D_final (numpy.ndarray): 3D coordinates of the points.
        dx (numpy.ndarray): Displacement vectors corresponding to the 3D points (used to calculate speed).
        filename (str): The name of the file to save the plot.
        sample_ratio (float): Fraction of points to randomly sample for plotting (default is 10%).
    """
    # Randomly sample a subset of points if needed
    sample_size = int(len(means3D_final) * sample_ratio)
    sample_indices = np.random.choice(len(means3D_final), sample_size, replace=False)
    sampled_means3D_final = means3D_final[sample_indices]
    sampled_dx = dx[sample_indices]
    
    # Calculate the speed magnitude for each point based on its displacement vector
    speed_magnitude = np.linalg.norm(sampled_dx, axis=1)
    
    # Normalize the speed magnitudes for consistent color mapping
    norm = Normalize(vmin=speed_magnitude.min(), vmax=speed_magnitude.max())
    norm_speed = norm(speed_magnitude)

    # Map the normalized speeds to colors: Red for high speed, blue for low speed, with fixed transparency
    colors = plt.cm.RdBu(1 - norm_speed)
    colors[:, 3] = 0.5  # Set a constant alpha value for transparency

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points in the 3D space
    sc = ax.scatter(sampled_means3D_final[:, 0], sampled_means3D_final[:, 1], sampled_means3D_final[:, 2],
                    c=colors, s=1)  # s=1 sets the marker size to small
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Points Colored by Speed Magnitude')

    # Add a color bar to represent the speed magnitudes
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu'), ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Speed Magnitude')
    cbar.set_ticks([norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax])
    cbar.set_ticklabels(['Low (Blue)', 'Medium', 'High (Red)'])

    # Save the plot to a PNG file
    plt.savefig(filename, format="png", dpi=300)
    plt.show()
