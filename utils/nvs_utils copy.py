import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import torch

def generate_delta_R(angle_degrees, num_rotations=10, rotation_axis=(0, 0, 1)):
    """
    Generate a list of small rotation matrices (delta R) for given small angles
    around a specified rotation axis.
    
    Args:
        angle_degrees (float): Total rotation angle in degrees for a full circle.
        num_rotations (int): Number of rotation matrices to generate.
        rotation_axis (tuple): Axis around which to rotate (default is Z-axis).
    
    Returns:
        list: List of rotation matrices.
    """
    # Calculate the small angle increment for each rotation
    angle_radians = np.radians(angle_degrees) / num_rotations
    delta_rotations = []

    # Generate incremental rotations around the specified axis
    for i in range(num_rotations):
        r = R.from_rotvec((i + 1) * angle_radians * np.array(rotation_axis))
        delta_rotations.append(r.as_matrix())
        print((i + 1) * angle_radians)
    return delta_rotations

def generate_delta_T(translation_distance):
    """
    Generate a list of small translation vectors (delta T) along each principal axis.
    """
    delta_Ts = [
        np.array([translation_distance, 0, 0]),
        np.array([-translation_distance, 0, 0]),
        np.array([0, translation_distance, 0]),
        np.array([0, -translation_distance, 0]),
        np.array([0, 0, translation_distance]),
        np.array([0, 0, -translation_distance]),
    ]
    return delta_Ts

def generate_new_viewpoints(original_viewpoint, angle_degrees=5, translation_distance=0.1):
    """
    Generate a set of new viewpoints by applying delta R and delta T to the original
    camera rotation and translation.
    
    Args:
        original_viewpoint: The original viewpoint object with properties R and T
        angle_degrees (float): Small rotation angle in degrees.
        translation_distance (float): Small translation distance.
    
    Returns:
        list: List of new viewpoint objects with modified rotation and translation.
    """
    delta_Rs = generate_delta_R(angle_degrees)
    delta_Ts = generate_delta_T(translation_distance)

    new_viewpoints = []
    
    # Combine each delta R and delta T with the original R and T
    for delta_R in delta_Rs:
        #for delta_T in delta_Ts:
            new_viewpoint = copy.deepcopy(original_viewpoint)
            new_viewpoint.R = new_viewpoint.R @ delta_R  # Apply rotation
            #new_viewpoint.T = new_viewpoint.T + delta_T  # Apply translation
            new_viewpoint.full_proj_transform = torch.from_numpy(new_viewpoint.R[:3, :3]).float()
            new_viewpoints.append(new_viewpoint)
            
    
    return new_viewpoints