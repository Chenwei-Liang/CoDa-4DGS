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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deformation_feature import deform_network_feature
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.nn import functional as F
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0],
                 load_coarse=False,
                 #for waymo
                 bg_gaussians: GaussianModel=None, 
                 build_octree=False, replace_pcd_by_octree_center=False,
                 build_grid=False, build_featgrid=False,
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.bg_gaussians = bg_gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.full_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"frame_info.json")):
            print("Found frame_info.json file, assuming Waymo data set!")
            scene_info = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.white_background, args.eval,
                                    use_bg_gs = bg_gaussians is not None,
                                    load_sky_mask = args.load_sky_mask, 
                                    load_panoptic_mask = args.load_panoptic_mask,
                                    load_intrinsic = args.load_intrinsic, 
                                    load_c2w = args.load_c2w, 
                                    load_sam_mask = args.load_sam_mask, 
                                    load_dynamic_mask = args.load_dynamic_mask, 
                                    load_feat_map = args.load_feat_map, 
                                    start_time = args.start_time, 
                                    end_time = args.end_time, 
                                    num_pts = args.num_pts,
                                    save_occ_grid = args.save_occ_grid,
                                    occ_voxel_size = args.occ_voxel_size,
                                    recompute_occ_grid = args.recompute_occ_grid,
                                    stride = args.stride,
                                    original_start_time = args.original_start_time,
                                    semantic_feature_path = args.semantic_feature_path,
                                    )
            dataset_type="waymo"
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Full Cameras")
            self.full_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.full_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if bg_gaussians is not None:
                self.bg_gaussians.load_ply(os.path.join(self.model_path,
                                                              "point_cloud",
                                                              "iteration_" + str(self.loaded_iter),
                                                              "bg_point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, semantic_feature_size=128)
            # for waymo
            if bg_gaussians is not None:
                self.bg_gaussians.create_from_pcd(scene_info.bg_point_cloud, self.cameras_extent, semantic_feature_size=128)

        self.gaussians.aabb = scene_info.cam_frustum_aabb
        self.gaussians.aabb_tensor = torch.tensor(scene_info.cam_frustum_aabb, dtype=torch.float32).cuda()
        self.gaussians.nerf_normalization = scene_info.nerf_normalization
        self.gaussians.img_width = scene_info.train_cameras[0].width
        self.gaussians.img_height = scene_info.train_cameras[0].height
        if scene_info.occ_grid is not None:
            self.gaussians.occ_grid = torch.tensor(scene_info.occ_grid, dtype=torch.bool).cuda() 
        else:
            self.gaussians.occ_grid = scene_info.occ_grid
        self.gaussians.occ_voxel_size = args.occ_voxel_size

        if args.load_panoptic_mask:
            self.gaussians.num_panoptic_objects = scene_info.num_panoptic_objects
            self.gaussians.panoptic_object_ids = scene_info.panoptic_object_ids
            self.gaussians.panoptic_id_to_idx = scene_info.panoptic_id_to_idx
        # for deformation-field
        if hasattr(self.gaussians, '_deformation'):
            self.gaussians._deformation.deformation_net.set_aabb(scene_info.cam_frustum_aabb[1],
                                                scene_info.cam_frustum_aabb[0])
        if build_octree:
            # forward : point cloud -> octree
            self.gaussians.build_octree(aabb= scene_info.cam_frustum_aabb, # use camera-extent aabb
                                        resolution=5, threshold=10)
            if replace_pcd_by_octree_center:
                self.gaussians.replace_pcd_by_octree_node()
            # to cuda 
            self.gaussians.octree.to_cuda()
            # backward : octree -> point cloud : get point - octree.node correspondence
            self.node_list, self.rot_list, self.scale_list = self.gaussians.octree.get_point_node_list()
            # check if all nodes are leaf nodes
            assert all([node.is_leaf() for node in self.node_list]), 'all nodes should be leaf nodes'
        if build_grid:
            self.gaussians.build_grid(aabb= scene_info.cam_frustum_aabb, # use camera-extent aabb
                                    res=[128, 128, 128])
        if build_featgrid:
            self.gaussians.build_featgrid(aabb= scene_info.cam_frustum_aabb, # use camera-extent aabb
                                res=[128, 128, 128])

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def save_gridgs(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_grid".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # background
        if self.bg_gaussians is not None:
            self.bg_gaussians.save_ply(os.path.join(point_cloud_path, "bg_point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getFullCameras(self, scale=1.0):
        return self.full_cameras[scale]