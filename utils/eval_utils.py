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
import gc
import random
import torch
import sys

import time
import json

import numpy as np
from utils.video_utils import render_pixels, save_videos


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

TENSORBOARD_FOUND = False
   
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    "feature_map_rgbs",
]

@torch.no_grad()
def do_evaluation(
    viewpoint_stack_full,
    viewpoint_stack_test,
    viewpoint_stack_train,
    gaussians,
    bg,
    pipe,
    eval_dir,
    render_full,
    step: int = 0,
    args = None,
    dcn=None,
):
    if len(viewpoint_stack_test) != 0:
        print("Evaluating Test Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_test,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test,
            step=step,
            dcn=dcn
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                "masked_psnr",
                "masked_ssim",
            ]:
                eval_dict[f"pixel_metrics/test/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/test_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_test_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/test_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_test)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    if len(viewpoint_stack_train) != 0 and len(viewpoint_stack_test) != 0:
        print("Evaluating train Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_train,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=False,
            debug=args.debug_test,
            step=step,
            dcn=dcn,
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                "masked_psnr",
                "masked_ssim",
            ]:
                eval_dict[f"pixel_metrics/train/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/train_videos", exist_ok=True)
        
        train_metrics_file = f"{eval_dir}/metrics/{step}_images_train.json"
        with open(train_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {train_metrics_file}")

        video_output_pth = f"{eval_dir}/train_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_train)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results
        torch.cuda.empty_cache()

    if render_full:
        print("Evaluating Full Set...")
        render_results = render_pixels(
            viewpoint_stack_full,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test,
            step=step,
            dcn=dcn,
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                "masked_psnr",
                "masked_ssim",
            ]:
                eval_dict[f"pixel_metrics/full/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/full_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_full_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/full_videos/{step}.mp4"
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_full)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )
        
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
