#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
import matplotlib.pyplot as plt
from scene.utils import pytorch_knn
from utils.dynamic_vis import plot_colored_points_with_speed

from sklearn.metrics.pairwise import cosine_similarity
import random

import torch
import random
from sklearn.metrics.pairwise import cosine_similarity


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine",return_decomposition=False,return_dx=False,render_feat=False, iteration=0, deformation_feature=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    if "coarse" in stage:
        #print("test_coarse_1")
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        semantic_feature = pc.get_semantic_feature
        
    elif "fine" in stage:
        semantic_feature = pc.get_semantic_feature.squeeze(1)
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, feat, dshs, sf = pc._deformation(means3D, scales, 
                                                        rotations, opacity, shs,
                                                        time, semantic_feature)
        point_semantic_feature= sf
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, dshs = deformation_feature(means3D_final, scales_final, 
                                                        rotations_final, opacity_final, shs_final,
                                                            time, point_semantic_feature, dx)

        semantic_feature=sf.unsqueeze(1)
    else:
        raise NotImplementedError



    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = shs_final.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        else:
            pass
    else:
        colors_precomp = override_color

    if colors_precomp is not None:
        shs_final = None
    rendered_image, feature_map, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    result_dict = {}
    
    result_dict.update({
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'feature_map': feature_map,
        "depth":depth})

    features_precomp = None

    if render_feat and "fine" in stage:
        colors_precomp = feat
        shs_final = None
        rendered_image2,feature_map, _, _ = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp, # [N,3]
            semantic_feature = semantic_feature,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
        
        result_dict.update({"feat": rendered_image2})

    if return_decomposition and dx is not None:
        dx_abs = torch.abs(dx) # [N,3]
        max_values = torch.max(dx_abs, dim=1)[0] # [N]
        thre = torch.mean(max_values)
        
        dynamic_mask = max_values > thre
        
        rendered_image_d, feature_map_d,radii_d, depth_d = rasterizer(
            means3D = means3D_final[dynamic_mask],
            means2D = means2D[dynamic_mask],
            shs = shs_final[dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[dynamic_mask] if colors_precomp is not None else None, # [N,3]
            semantic_feature = semantic_feature[dynamic_mask] if colors_precomp is not None else None,
            opacities = opacity[dynamic_mask],
            scales = scales_final[dynamic_mask],
            rotations = rotations_final[dynamic_mask],
            cov3D_precomp = cov3D_precomp[dynamic_mask] if cov3D_precomp is not None else None)
        
        rendered_image_s,feature_map_s, radii_s, depth_s = rasterizer(
            means3D = means3D_final[~dynamic_mask],
            means2D = means2D[~dynamic_mask],
            shs = shs_final[~dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[~dynamic_mask] if colors_precomp is not None else None, # [N,3]
            semantic_feature = semantic_feature[~dynamic_mask] if colors_precomp is not None else None,
            opacities = opacity[~dynamic_mask],
            scales = scales_final[~dynamic_mask],
            rotations = rotations_final[~dynamic_mask],
            cov3D_precomp = cov3D_precomp[~dynamic_mask] if cov3D_precomp is not None else None
            )
        
        result_dict.update({
            "render_d": rendered_image_d,
            "depth_d":depth_d,
            "feature_map_d": feature_map_d,
            "visibility_filter_d" : radii_d > 0,
            "render_s": rendered_image_s,
            "depth_s":depth_s,
            "feature_map_s": feature_map_s,
            "visibility_filter_s" : radii_s > 0,
            })
        
    if return_dx and "fine" in stage:
        result_dict.update({"dx": dx})
        result_dict.update({'dshs' : dshs})
    
    return result_dict





