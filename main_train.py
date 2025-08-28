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
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import random
import torch
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, deform_network_feature
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.scene_utils import render_training_image
import copy

import numpy as np

import json
import torch.nn.functional as F
from scene.cnn_decoder import CNN_decoder
from utils.eval_utils import do_evaluation
import time
import mmcv
from utils.params_utils import merge_hparams
import yaml

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

   
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    "context_awareness",
]

def scene_reconstruction(dataset, opt, hyper, pipe, checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args, cnn_decoder, cnn_decoder_optimizer, dcn, dcn_optimizer):
    first_iter = 0

    gaussians.training_setup(opt)
    print(stage)
    print(checkpoint)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            print(first_iter)
            gaussians.restore(model_params, opt)
            if stage == "fine":
                print("load dcn")
                dcn=deform_network_feature(hyper)
                iteration = re.search(r"(\d+)\.pth$", checkpoint).group(1)
                dir_path = os.path.dirname(checkpoint)
                dcn_path = os.path.join(dir_path, f"deformation_feature_fine_{iteration}.pth")
                dcn.load_state_dict(torch.load(dcn_path))
                dcn.cuda()
                cnn_decoder_path = os.path.join(dir_path, f"cnn_decoder_fine_{iteration}.pth")
                cnn_decoder.load_state_dict(torch.load(cnn_decoder_path))
                cnn_decoder.cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack:
        
        viewpoint_stack = list(train_cams)
        random.shuffle(viewpoint_stack)
    
    batch_size = opt.batch_size
    print("data loading done")    
        
    count = 0
    psnr_dict = {}
    for iteration in range(first_iter, final_iter+1):     

        iter_start.record()

        position_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # batch size
        idx = 0
        viewpoint_cams = []

        while idx < batch_size :    
            
            if not viewpoint_stack:

                viewpoint_stack = list(train_cams)
                random.shuffle(viewpoint_stack)

            viewpoint_cams.append(viewpoint_stack.pop())
            idx +=1
        if len(viewpoint_cams) == 0:
            continue

        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        depth_preds = []
        gt_depths = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        gt_feature_maps =[]
        feature_maps=[]
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,return_dx=True,render_feat = True if ('fine' in stage and args.feat_head) else False, iteration=iteration, deformation_feature=dcn)
            feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth_pred = render_pkg["depth"]
            depth_preds.append(depth_pred.unsqueeze(0))
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.depth_map.cuda()

            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)


            gt_feature_map = viewpoint_cam.semantic_feature.cuda()
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 

            feature_map = cnn_decoder(feature_map)
            gt_feature_maps.append(gt_feature_map.unsqueeze(0))
            feature_maps.append(feature_map.unsqueeze(0))

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_pred_tensor = torch.cat(depth_preds,0)
        feature_map_tensor=torch.cat(feature_maps,0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths,0).float()
        gt_feature_map_tensor=torch.cat(gt_feature_maps,0).float()
       

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
       
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1

        Ll1_feature = torch.nn.SmoothL1Loss()(feature_map_tensor, gt_feature_map_tensor) 
        loss+=Ll1_feature
        
        # dx loss
        if 'fine' in stage and not args.no_dx and opt.lambda_dx !=0:
            dx_abs = torch.abs(render_pkg['dx'])
            dx_loss = torch.mean(dx_abs) * opt.lambda_dx
            loss += dx_loss
        if 'fine' in stage and not args.no_dshs and opt.lambda_dshs != 0:
            dshs_abs = torch.abs(render_pkg['dshs'])
            dshs_loss = torch.mean(dshs_abs) * opt.lambda_dshs
            loss += dshs_loss
        if opt.lambda_depth != 0:
            depth_loss = compute_depth("l2", depth_pred_tensor, gt_depth_tensor) * opt.lambda_depth
            loss += depth_loss
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if stage == 'fine' and args.feat_head:
            feat = render_pkg['feat'].to('cuda') # [3,640,960]
            gt_feat = viewpoint_cam.feat_map.permute(2,0,1).to('cuda')
            loss_feat = l2_loss(feat, gt_feat) * opt.lambda_feat
            loss += loss_feat
        loss.backward()
        
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                dynamic_points = 0
                if 'fine' in stage and not args.no_dx:
                    dx_abs = torch.abs(render_pkg['dx']) # [N,3]
                    max_values = torch.max(dx_abs, dim=1)[0] # [N]
                    thre = torch.mean(max_values)                   
                    mask = (max_values > thre)
                    dynamic_points = torch.sum(mask).item()

                print_dict = {
                    "step": f"{iteration}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "loss_origin": f"{loss}",
                    "psnr": f"{psnr_:.{2}f}",
                    "dynamic point": f"{dynamic_points}",
                    "point":f"{total_point}",
                    }
                progress_bar.set_postfix(print_dict)
                metrics_file = f"{scene.model_path}/logger.json"
                with open(metrics_file, "a") as f:
                    json.dump(print_dict, f)
                    f.write('\n')

                progress_bar.update(100)
            if iteration == final_iter:
                progress_bar.close()

           
            timer.pause()

            if dataset.render_process:
                if (iteration == 1) \
                    or (iteration < 10000 and iteration % 1000 == 999) \
                        or (iteration < 30000 and iteration % 2000 == 1999) \
                            or (iteration >= 30000 and iteration %  3000 == 2999) :
                        if len(test_cams) != 0:
                            render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(), dcn)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),dcn)


            timer.start()
            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0]>200000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            if iteration < final_iter+1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                cnn_decoder_optimizer.step()
                cnn_decoder_optimizer.zero_grad(set_to_none = True)  
                dcn_optimizer.step()    
                dcn_optimizer.zero_grad(set_to_none=True)       

            if (iteration in checkpoint_iterations):

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
                torch.save(dcn.state_dict(), scene.model_path + "/deformation_feature" +f"_{stage}_" + str(iteration) + ".pth")
                torch.save(cnn_decoder.state_dict(), scene.model_path + "/cnn_decoder" +f"_{stage}_" + str(iteration) + ".pth")

            if (iteration %  10000== 0):
                eval_dir = os.path.join(args.model_path,"eval")
                os.makedirs(eval_dir,exist_ok=True)
                viewpoint_stack_full = scene.getFullCameras().copy()
                viewpoint_stack_test = scene.getTestCameras().copy()
                viewpoint_stack_train = scene.getTrainCameras().copy()

                do_evaluation(
                    viewpoint_stack_full,
                    viewpoint_stack_test,
                    viewpoint_stack_train,
                    gaussians,
                    background,
                    pipe,
                    eval_dir,
                    render_full=True,
                    step=iteration,
                    args=args,
                    dcn=dcn
                )


def training(dataset, hyper, opt, pipe, checkpoint_iterations, checkpoint, debug_from, expname, tb_writer, args):
    """
    Main training function for the scene reconstruction pipeline.

    Args:
        dataset: The dataset object containing data and configurations.
        hyper: Hyperparameter object for training.
        opt: Options/arguments for training.
        pipe: Processing pipeline for the model.
        checkpoint_iterations: Frequency of saving intermediate checkpoints.
        checkpoint: Checkpoint file to resume training from.
        debug_from: Iteration step to enable debugging.
        expname: Name of the experiment for logging.

    Returns:
        None
    """

    # Initialize models
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dcn = deform_network_feature(hyper).to("cuda")

    # Prepare scene and timer
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer = Timer()
    timer.start()

    dcn_optimizer = torch.optim.Adam(dcn.parameters(), lr=0.000016 * scene.cameras_extent)

    cnn_decoder = CNN_decoder(128, 512).to("cuda")
    cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


    # Configure background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Prepare evaluation directories
    eval_dir = os.path.join(args.model_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Prepare camera stacks for evaluation
    viewpoint_stack_full = scene.getFullCameras().copy()
    viewpoint_stack_test = scene.getTestCameras().copy()
    viewpoint_stack_train = scene.getTrainCameras().copy()

    # Stage 1: Coarse reconstruction
    scene_reconstruction(
        dataset, opt, hyper, pipe,
        checkpoint_iterations, checkpoint, debug_from, gaussians, scene, 
        "coarse", tb_writer, opt.coarse_iterations, timer, args, 
        cnn_decoder, cnn_decoder_optimizer, dcn, dcn_optimizer
    )

    # Load prior checkpoint for fine stage if provided
    if args.prior_checkpoint:
        assert 'fine' in args.prior_checkpoint, "Checkpoint must correspond to 'fine' stage."
        gaussians = load_prior_checkpoint(args.prior_checkpoint, dataset.sh_degree, hyper, gaussians)

    # Stage 2: Fine reconstruction
    scene_reconstruction(
        dataset, opt, hyper, pipe,
        checkpoint_iterations, checkpoint, debug_from, gaussians, scene, 
        "fine", tb_writer, opt.iterations, timer, args,
        cnn_decoder, cnn_decoder_optimizer, dcn, dcn_optimizer
    )

    # Perform evaluation
    do_evaluation(
        viewpoint_stack_full, viewpoint_stack_test, viewpoint_stack_train,
        gaussians, background, pipe, eval_dir, 
        render_full=True, step=opt.iterations, args=args, dcn=dcn
    )


def load_prior_checkpoint(prior_checkpoint, sh_degree, hyper, gaussians):
    """
    Loads the prior checkpoint and restores deformation network.

    Args:
        prior_checkpoint: Path to the checkpoint file.
        sh_degree: Spherical harmonics degree from dataset.
        hyper: Hyperparameter object.
        gaussians: Gaussian model instance to restore.

    Returns:
        Updated Gaussian model with restored deformation network.
    """
    gaussians_prev = GaussianModel(sh_degree, hyper)
    model_params, _ = torch.load(prior_checkpoint)
    gaussians_prev.restore(model_params)
    deformation_net = gaussians_prev._deformation

    # Clear unused objects to free memory
    del gaussians_prev
    gc.collect()
    torch.cuda.empty_cache()

    gaussians._deformation = deformation_net.to("cuda")
    return gaussians


def prepare_output_and_logger(expname, args):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
):
    """
    Logs training and testing metrics during the training process.

    Parameters:
        tb_writer: Tensorboard writer for logging.
        iteration: Current training iteration.
        Ll1: L1 loss component of the current training iteration.
        loss: Total loss value of the current iteration.
        l1_loss: Function to compute L1 loss.
        elapsed: Time elapsed for the current iteration.
        scene: Scene object containing training and testing data.
        renderFunc: Function for rendering images.
        renderArgs: Additional arguments for the render function.
        stage: Current training stage.
    """
    if tb_writer:
        # Log training loss and iteration time
        tb_writer.add_scalar(f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{stage}/train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    torch.cuda.empty_cache()
    validation_configs = _prepare_validation_configs(scene)

    for config in validation_configs:
        if config["cameras"]:
            evaluate_and_log(
                config,
                tb_writer,
                iteration,
                scene,
                renderFunc,
                renderArgs,
                l1_loss,
                stage,
            )

    # Log additional scene properties to TensorBoard
    if tb_writer:
        log_scene_properties(scene, tb_writer, iteration, stage)

    torch.cuda.empty_cache()


def _prepare_validation_configs(scene):
    """
    Prepares validation configurations for training and testing cameras.

    Parameters:
        scene: Scene object containing train and test cameras.

    Returns:
        List of dictionaries containing validation configurations.
    """
    train_cameras = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]
    test_cameras = (
        [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]
        if len(scene.getTestCameras()) > 0
        else []
    )

    validation_configs = [{"name": "train", "cameras": train_cameras}]
    if test_cameras:
        validation_configs.append({"name": "test", "cameras": test_cameras})
    return validation_configs


def evaluate_and_log(config, tb_writer, iteration, scene, renderFunc, renderArgs, l1_loss, stage):
    """
    Evaluates the given configuration and logs results.

    Parameters:
        config: Dictionary containing validation name and cameras.
        tb_writer: Tensorboard writer for logging.
        iteration: Current training iteration.
        scene: Scene object containing scene data.
        renderFunc: Function for rendering images.
        renderArgs: Additional arguments for the render function.
        l1_loss: Function to compute L1 loss.
        stage: Current training stage.
    """
    l1_test = 0.0
    psnr_test = 0.0

    for idx, viewpoint in enumerate(config["cameras"]):
        image = torch.clamp(
            renderFunc(viewpoint, scene.gaussians, stage=stage, *renderArgs)["render"],
            0.0,
            1.0,
        )
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        if tb_writer and idx < 5:
            _log_rendered_images(tb_writer, iteration, image, gt_image, stage, config["name"], viewpoint.image_name)

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()

    l1_test /= len(config["cameras"])
    psnr_test /= len(config["cameras"])

    print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f}, PSNR {psnr_test:.4f}")
    if tb_writer:
        tb_writer.add_scalar(f"{stage}/{config['name']}/loss_viewpoint_l1", l1_test, iteration)
        tb_writer.add_scalar(f"{stage}/{config['name']}/loss_viewpoint_psnr", psnr_test, iteration)


def _log_rendered_images(tb_writer, iteration, image, gt_image, stage, config_name, image_name):
    """
    Logs rendered and ground-truth images to TensorBoard.

    Parameters:
        tb_writer: Tensorboard writer for logging.
        iteration: Current training iteration.
        image: Rendered image tensor.
        gt_image: Ground-truth image tensor.
        stage: Current training stage.
        config_name: Name of the validation configuration.
        image_name: Identifier for the image.
    """
    try:
        tb_writer.add_images(f"{stage}/{config_name}_view_{image_name}/render", image[None], global_step=iteration)
        if iteration == 0:
            tb_writer.add_images(
                f"{stage}/{config_name}_view_{image_name}/ground_truth", gt_image[None], global_step=iteration
            )
    except Exception as e:
        print(f"Error logging images to TensorBoard: {e}")


def log_scene_properties(scene, tb_writer, iteration, stage):
    """
    Logs scene properties such as opacity histograms and deformation rates to TensorBoard.

    Parameters:
        scene: Scene object containing scene data.
        tb_writer: Tensorboard writer for logging.
        iteration: Current training iteration.
        stage: Current training stage.
    """
    tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
    tb_writer.add_scalar(f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration)
    tb_writer.add_scalar(
        f"{stage}/deformation_rate",
        scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0],
        iteration,
    )
    tb_writer.add_histogram(
        f"{stage}/scene/motion_histogram",
        scene.gaussians._deformation_accum.mean(dim=-1) / 100,
        iteration,
        max_bins=500,
    )

def setup_seed(seed):
    """
    Set up a consistent random seed across libraries to ensure reproducibility.

    Parameters:
        seed (int): The seed value to use for random number generation.
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's random module
    random.seed(seed)

    # Ensure reproducibility in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for deterministic behavior

    print(f"Random seed set to {seed}.")

def setup_args():
    """
    Sets up the argument parser with all necessary parameters and returns the parsed arguments.
    """
    parser = ArgumentParser(description="Training script parameters")
    
    # Add seed for reproducibility
    setup_seed(1244)
    
    # Model and optimization parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # General configuration arguments
    parser.add_argument('--ip', type=str, default="127.0.0.1", help="IP address for GUI server")
    parser.add_argument('--port', type=int, default=6009, help="Port for GUI server")
    parser.add_argument('--debug_from', type=int, default=-1, help="Iteration to start debugging")
    parser.add_argument('--detect_anomaly', action='store_true', default=False, help="Enable anomaly detection")
    parser.add_argument('--quiet', action='store_true', help="Suppress log outputs")
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[50000], help="Checkpoint iterations")
    parser.add_argument('--start_checkpoint', type=str, default="", help="Start checkpoint path")
    parser.add_argument('--expname', type=str, default="waymo", help="Experiment name")
    parser.add_argument('--configs', type=str, default="", help="Path to config file")
    parser.add_argument('--eval_only', action='store_true', help="Perform evaluation only")
    parser.add_argument('--prior_checkpoint', type=str, default=None, help="Path to prior checkpoint")
    parser.add_argument('--prior_checkpoint2', type=str, default=None, help="Path to second prior checkpoint")
    parser.add_argument('--config_yaml', type=str, default="", help="Path to yaml config file")

    # Parse arguments
    args = parser.parse_args(sys.argv[1:])
    if args.config_yaml:
        with open(args.config_yaml, "r") as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    return args, lp, op, pp, hp

def main():
    """
    Main function to set up and run the training process.
    """

    torch.cuda.empty_cache()  # Clear GPU memory cache
    args, lp, op, pp, hp = setup_args()

    print(f"Optimizing model at: {args.model_path}")
    print("Arguments:", args)


    tb_writer = prepare_output_and_logger(args.expname, args)      
    # Initialize system state (e.g., RNG, logging)
    safe_state(args.quiet)

    # Enable anomaly detection if specified
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Run the training process
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
        tb_writer,
        args,
    )

    print("\nTraining complete.")

if __name__ == "__main__":
    main()