VIS = 0
import logging
import os
from typing import Callable, Dict, List, Optional
from plyfile import PlyData, PlyElement

import imageio
import numpy as np
import torch
from utils.loss_utils import l1_loss
from skimage.metrics import structural_similarity as ssim
from lpipsPyTorch import lpips
from torch import Tensor
from tqdm import tqdm, trange
from gaussian_renderer import render

from utils.image_utils import psnr
from utils.nvs_utils import generate_delta_R, generate_delta_T, generate_new_viewpoints

import numpy as np
from PIL import Image

from sklearn.decomposition import PCA


pca = PCA(n_components=3)
pca_initialized = False

def initialize_pca_if_needed(reference_feature_map):
    """
    Initializes the PCA (Principal Component Analysis) model if it has not already been initialized.

    Parameters:
        reference_feature_map (torch.Tensor): The reference feature map used to fit the PCA model.
            Expected to have a shape where the first dimension corresponds to feature channels.

    Globals:
        pca (sklearn.decomposition.PCA): The global PCA object used for dimensionality reduction.
        pca_initialized (bool): A flag indicating whether the PCA model has been initialized.

    Steps:
        1. Check if the PCA has already been initialized.
        2. If not initialized, reshape the reference feature map to prepare it for PCA fitting.
        3. Fit the PCA model using the reshaped feature map.
        4. Set the global flag to indicate the PCA has been initialized.
        5. Print a confirmation message.
    """
    global pca, pca_initialized
    if not pca_initialized:
        # Convert the feature map to a NumPy array and reshape it for PCA
        reference_feature_map_np = reference_feature_map.cpu().numpy().reshape(128, -1).T
        
        # Fit the PCA model using the reshaped feature map
        pca.fit(reference_feature_map_np)
        
        # Update the global flag to mark PCA as initialized
        pca_initialized = True
        
        # Confirmation message
        print("PCA initialized based on the reference frame.")

def apply_fixed_pca(semantic_feature_map):
    """
    Applies a pre-trained PCA to transform a semantic feature map into an RGB representation.

    Parameters:
        semantic_feature_map (torch.Tensor): The input semantic feature map. Expected shape:
            (channels, height, width), where channels must match the number used to train the PCA (e.g., 128).

    Returns:
        numpy.ndarray: The transformed feature map in RGB format with values normalized to the [0, 1] range.
            Shape: (height, width, 3).
    """
    # Convert the feature map to a NumPy array and reshape for PCA input
    # Reshape to (num_pixels, channels) where each row is a pixel's feature vector
    semantic_feature_map_np = semantic_feature_map.cpu().numpy().reshape(128, -1).T

    # Apply the PCA transformation to reduce dimensions to 3 (RGB channels)
    semantic_feature_map_rgb = pca.transform(semantic_feature_map_np)  # Shape: (num_pixels, 3)

    # Normalize the RGB values to the [0, 1] range
    epsilon = 1e-8  # Small value to prevent division by zero
    semantic_feature_map_rgb -= semantic_feature_map_rgb.min(axis=0)  # Shift to start from 0
    semantic_feature_map_rgb /= (semantic_feature_map_rgb.max(axis=0) + epsilon)  # Scale to maximum of 1

    # Reshape the RGB values back to the original spatial dimensions
    h, w = semantic_feature_map.shape[1], semantic_feature_map.shape[2]  # Get height and width from the input
    semantic_feature_map_rgb = semantic_feature_map_rgb.reshape(h, w, 3)  # Reshape to (height, width, 3)

    return semantic_feature_map_rgb


from utils.visualization_tools import (
    resize_five_views,
    scene_flow_to_rgb,
    to8b,
    visualize_depth,
)

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)
flow_visualizer = (
    lambda frame: scene_flow_to_rgb(
        frame,
        background="bright",
        flow_max_radius=1.0,
    )
    .cpu()
    .numpy()
)
get_numpy: Callable[[Tensor], np.ndarray] = lambda x: x.squeeze().cpu().numpy()
non_zero_mean: Callable[[Tensor], float] = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    """
    Performs robust PCA on a feature matrix and identifies RGB scaling bounds to reduce outlier influence.

    Parameters:
        features (torch.Tensor): Input feature matrix of shape (N, C), where N is the number of samples
                                 and C is the feature dimension.
        m (float): Hyperparameter controlling the threshold for outliers. Default is 2 (in terms of standard deviations).
        remove_first_component (bool): If True, removes the first PCA component before computing the RGB bounds.
    
    Returns:
        reduction_mat (torch.Tensor): The PCA reduction matrix of shape (C, 3), reducing features to 3 components.
        rgb_min (torch.Tensor): The minimum values for robust scaling in RGB space (shape: (3,)).
        rgb_max (torch.Tensor): The maximum values for robust scaling in RGB space (shape: (3,)).
    """
    # Ensure the input feature matrix has the correct shape (N, C)
    assert len(features.shape) == 2, "features should have shape (N, C)"

    # Perform low-rank PCA to compute the reduction matrix for the top 3 components
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat  # Project the features onto the PCA components (N, 3)

    if remove_first_component:
        # Normalize the colors to [0, 1] for filtering the first component
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        
        # Create a mask to filter out regions where the first component is dominant
        fg_mask = tmp_colors[..., 0] < 0.2
        
        # Recompute PCA using only the masked features
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat  # Re-project features using the updated reduction matrix
    else:
        # If the first component is not removed, use all features
        fg_mask = torch.ones_like(colors[:, 0]).bool()

    # Compute the deviation of each feature from the median
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    
    # Calculate the robust measure of spread (median absolute deviation)
    mdev = torch.median(d, dim=0).values
    
    # Scale deviations to identify outliers
    s = d / mdev

    # Filter inliers based on the hyperparameter `m` (number of standard deviations allowed)
    rins = colors[fg_mask][s[:, 0] < m, 0]  # Red channel inliers
    gins = colors[fg_mask][s[:, 1] < m, 1]  # Green channel inliers
    bins = colors[fg_mask][s[:, 2] < m, 2]  # Blue channel inliers

    # Calculate the robust RGB scaling bounds based on inliers
    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    # Return the PCA reduction matrix and the RGB scaling bounds
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def render_pixels(
    viewpoint_stack,
    gaussians,
    bg,
    pipe,
    compute_metrics: bool = True,
    return_decomposition: bool = True,
    debug: bool = False,
    step: int = 0,
    dcn=None,
):
    """
    Render pixel-related outputs using the specified model and compute optional metrics.

    Args:
        viewpoint_stack: List of viewpoints for rendering.
        gaussians: Gaussian parameters for the model.
        bg: Background model or parameters.
        pipe: Pipeline object used for rendering.
        compute_metrics (bool, optional): If True, computes evaluation metrics such as PSNR, SSIM, and LPIPS.
                                           Defaults to True.
        return_decomposition (bool, optional): If True, includes static-dynamic decomposition in the results.
                                               Defaults to True.
        debug (bool, optional): If True, enables debugging mode for the render function. Defaults to False.
        step (int, optional): Current training or evaluation step for logging or debugging. Defaults to 0.
        dcn: Optional secondary MLP model for additional computations. Defaults to None.

    Returns:
        dict: Rendered outputs, potentially including metrics and decomposition results.
    """
    # Perform rendering with the specified parameters
    render_results = render_func(
        viewpoint_stack,
        gaussians,
        pipe,
        bg,
        compute_metrics=compute_metrics,
        return_decomposition=return_decomposition,
        debug=debug,
        step=step,
        dcn=dcn,
    )

    # Compute and log metrics if required
    if compute_metrics:
        num_samples = len(viewpoint_stack)
        print(f"Evaluating over {num_samples} images:")
        print(f"\tPSNR: {render_results.get('psnr', 0.0):.4f}")
        print(f"\tSSIM: {render_results.get('ssim', 0.0):.4f}")
        print(f"\tLPIPS: {render_results.get('lpips', 0.0):.4f}")
        print(f"\tMasked PSNR: {render_results.get('masked_psnr', 0.0):.4f}")
        print(f"\tMasked SSIM: {render_results.get('masked_ssim', 0.0):.4f}")

    return render_results



def render_func(
    viewpoint_stack,
    gaussians,
    pipe,
    bg,
    compute_metrics: bool = False,
    return_decomposition:bool = False,
    num_cams: int = 3,
    debug: bool = False,
    save_seperate_pcd = False,
    step: int = 0,
    dcn = None,
):
    """
    Renders a dataset utilizing a specified render function.
    For efficiency and space-saving reasons, this function doesn't store the original features; instead, it keeps
    the colors reduced via PCA.
    TODO: clean up this function

    Parameters:
        dataset: Dataset to render.
        render_func: Callable function used for rendering the dataset.
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
    """
    # rgbs
    rgbs, gt_rgbs = [], []
    static_rgbs, dynamic_rgbs = [], []
    shadow_reduced_static_rgbs, shadow_only_static_rgbs = [], []

    # depths
    depths, median_depths = [], []
    static_depths, static_opacities = [], []
    dynamic_depths, dynamic_opacities = [], []
    opacities, sky_masks = [], []

    # features
    pred_dinos, gt_dinos = [], []
    pred_dinos_pe_free, pred_dino_pe = [], []
    static_dinos, dynamic_dinos = [], []  
    dynamic_dino_on_static_rgbs, dynamic_rgb_on_static_dinos = [], []
    context_awareness = []
    forward_flows, backward_flows = [], []
    dx_list = []

    if compute_metrics:
        psnrs, ssim_scores, feat_psnrs = [], [], []
        masked_psnrs, masked_ssims = [], []
        masked_feat_psnrs = [],
        lpipss = []

    with torch.no_grad():
        
        for i in tqdm(range(len(viewpoint_stack)), desc=f"rendering full data", dynamic_ncols=True):
            viewpoint_cam = viewpoint_stack[i]
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,return_decomposition = return_decomposition,return_dx=True,iteration=step, deformation_feature=dcn)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            rgb = image
            gt_rgb = viewpoint_cam.original_image.cuda()

            rgbs.append(get_numpy(rgb.permute(1, 2, 0)))
            gt_rgbs.append(get_numpy(gt_rgb.permute(1, 2, 0)))
            
            if "render_s" in render_pkg:
                static_rgbs.append(get_numpy(render_pkg["render_s"].permute(1, 2, 0)))
                visibility_filter_s = render_pkg['visibility_filter_s']
            if "render_d" in render_pkg:
                dy_rgb = render_pkg["render_d"].permute(1, 2, 0)
                dynamic_rgbs.append(get_numpy(dy_rgb))
                visibility_filter_d = render_pkg['visibility_filter_d']
            if "feature_map" in render_pkg:
                initialize_pca_if_needed(render_pkg["feature_map"])
                semantic_feature_map = render_pkg["feature_map"]
                semantic_feature_map_rgb = apply_fixed_pca(semantic_feature_map)
                context_awareness.append(semantic_feature_map_rgb)
            
            # ------------- depth ------------- #
            depth = render_pkg["depth"]
            depth_np = depth.permute(1, 2, 0).cpu().numpy()
            depth_np /= depth_np.max()
            depths.append(depth_np)

            # ------------- flow ------------- #
            if "dx" in render_pkg and render_pkg['dx'] is not None:
                dx = render_pkg['dx']
                dx = torch.tensor(dx)
                dx_max = torch.max(dx)
                dx_min = torch.min(dx)
                dx_list.append(dx)     
            if compute_metrics:
                psnrs.append(psnr(rgb, gt_rgb).mean().double().item())
                ssim_scores.append(
                    ssim(
                        get_numpy(rgb),
                        get_numpy(gt_rgb),
                        data_range=1.0,
                        channel_axis=0,
                    )
                )
                lpipss.append(torch.tensor(lpips(rgb, gt_rgb,net_type='alex')).mean().item())
                
                dynamic_mask = get_numpy(viewpoint_cam.dynamic_mask).astype(bool)
                if dynamic_mask.sum() > 0:
                    rgb_d = rgb.permute(1, 2, 0)[dynamic_mask]
                    rgb_d = rgb_d.permute(1, 0)
                    gt_rgb_d = gt_rgb.permute(1, 2, 0)[dynamic_mask]
                    gt_rgb_d = gt_rgb_d.permute(1, 0)

                    masked_psnrs.append(
                    psnr(rgb_d, gt_rgb_d).mean().double().item()
                    )
                    masked_ssims.append(
                        ssim(
                            get_numpy(rgb.permute(1, 2, 0)),
                            get_numpy(gt_rgb.permute(1, 2, 0)),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][dynamic_mask].mean()
                    )

        if save_seperate_pcd and len(dx_list)>1:    
                
            dynamic_pcd_path = os.path.join('test','dynamic.ply')
            static_pcd_path = os.path.join('test','static.ply')

            gaussians.save_ply_split(dynamic_pcd_path, static_pcd_path, dx_list, visibility_filter)

        if len(dx_list)>1:
            bf_color_first = []
            ff_color_last = []
            for t in range(len(dx_list)): 
                if t < len(dx_list)-num_cams:
                    forward_flow_t = dx_list[t + num_cams] - dx_list[t]
                    ff_color = flow_visualizer(forward_flow_t)
                    ff_color = torch.from_numpy(ff_color).to("cuda") 
                    if debug:
                        ff_color = (ff_color - torch.min(ff_color)) / (torch.max(ff_color) - torch.min(ff_color) + 1e-6) 

                    if t == len(dx_list)-num_cams-1 or t == len(dx_list)-num_cams-2 or t == len(dx_list)-num_cams-3: 
                        ff_color_last.append(ff_color)              
                    render_pkg2 = render(viewpoint_stack[t], gaussians, pipe, bg, override_color=ff_color, iteration=step, deformation_feature=dcn)
                    ff_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    forward_flows.append(ff_map)
                
                if t > num_cams-1:
                    backward_flow_t = dx_list[t] - dx_list[t - num_cams]
                    bf_color = flow_visualizer(backward_flow_t)
                    bf_color = torch.from_numpy(bf_color).to("cuda") 
                    if debug:
                        bf_color = (bf_color - torch.min(bf_color)) / (torch.max(bf_color) - torch.min(bf_color) + 1e-6) 
                    if t == num_cams or t == num_cams+1 or t == num_cams+2: 
                        bf_color_first.append(bf_color)                 
                    render_pkg2 = render(viewpoint_stack[t], gaussians, pipe, bg, override_color=bf_color, iteration=step, deformation_feature=dcn)
                    bf_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    backward_flows.append(bf_map)

            for i, bf_color in enumerate(bf_color_first):
                render_pkg3 = render(viewpoint_stack[i], gaussians, pipe, bg, override_color=bf_color,iteration=step, deformation_feature=dcn)            
                bf_map_first = render_pkg3['render'].permute(1, 2, 0).cpu().numpy()       
                backward_flows.insert(i, bf_map_first)

            for i, ff_color in enumerate(ff_color_last):
                render_pkg4 = render(viewpoint_stack[len(viewpoint_stack)-num_cams+i], gaussians, pipe, bg, override_color=ff_color,iteration=step, deformation_feature=dcn)            
                ff_map_last = render_pkg4['render'].permute(1, 2, 0).cpu().numpy()       
                forward_flows.append(ff_map_last)           

    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1

    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["opacities"] = opacities

    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(static_rgbs)>0:
        results_dict["static_rgbs"] = static_rgbs
    if len(dynamic_rgbs)>0:
        results_dict["dynamic_rgbs"] = dynamic_rgbs
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(pred_dinos) > 0:
        results_dict["dino_feats"] = pred_dinos
    if len(gt_dinos) > 0:
        results_dict["gt_dino_feats"] = gt_dinos
    if len(pred_dinos_pe_free) > 0:
        results_dict["dino_feats_pe_free"] = pred_dinos_pe_free
    if len(pred_dino_pe) > 0:
        results_dict["dino_pe"] = pred_dino_pe
    if len(static_dinos) > 0:
        results_dict["static_dino_feats"] = static_dinos
    if len(dynamic_dinos) > 0:
        results_dict["dynamic_dino_feats"] = dynamic_dinos
    if len(dynamic_dino_on_static_rgbs) > 0:
        results_dict["dynamic_dino_on_static_rgbs"] = dynamic_dino_on_static_rgbs
    if len(dynamic_rgb_on_static_dinos) > 0:
        results_dict["dynamic_rgb_on_static_dinos"] = dynamic_rgb_on_static_dinos
    if len(shadow_reduced_static_rgbs) > 0:
        results_dict["shadow_reduced_static_rgbs"] = shadow_reduced_static_rgbs
    if len(shadow_only_static_rgbs) > 0:
        results_dict["shadow_only_static_rgbs"] = shadow_only_static_rgbs
    if len(forward_flows) > 0:
        results_dict["forward_flows"] = forward_flows
    if len(backward_flows) > 0:
        results_dict["backward_flows"] = backward_flows
    if len(median_depths) > 0:
        results_dict["median_depths"] = median_depths
    if len(dx_list) > 0:
        results_dict['dx_list'] = dx_list
    if len(context_awareness)>0:
        results_dict["context_awareness"] = context_awareness
    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "context_awareness"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "context_awareness"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1: 
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        for key in keys:
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]

            frames = resize_five_views(frames)
            frames = np.concatenate(frames, axis=1)
            merged_list.append(frames)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        print(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "context_awareness"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        print(key)
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if key not in render_results or len(render_results[key]) == 0:
            continue
        for i in range(num_timestamps): 
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            frames = resize_five_views(frames)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"_{i*3 + j:03d}.png"),
                        to8b(frame),
                    )
            frames = to8b(np.concatenate(frames, axis=1))
            writer.append_data(frames) # [H,W,3]
            if i == return_frame_id:
                return_frame_dict[key] = frames
        writer.close()
        del writer
        if verbose:
            print(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
