import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = 255 
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args

        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])

        self.mask_net = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))

        def init_mask_net(mask_net, negative_bias=-10):
            
            for layer in mask_net:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0, std=1e-5)  
                    torch.nn.init.constant_(layer.bias, negative_bias)  
            return mask_net


        self.mask_net = init_mask_net(self.mask_net, negative_bias=-10)

        self.ratio=0
        self.create_net()

    def create_net(self):
        mlp_out_dim = 0
        def init_deform_layer(output_dim, std=1e-5):
            layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.W, self.W),
                nn.ReLU(),
                nn.Linear(self.W, output_dim)
            )

            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    torch.nn.init.normal_(sublayer.weight, mean=0, std=std)
                    torch.nn.init.constant_(sublayer.bias, 0)
            return layer


        self.pos_deform = init_deform_layer(3, std=1e-7)
        self.scales_deform = init_deform_layer(3, std=1e-7)
        self.rotations_deform = init_deform_layer(4, std=1e-7)
        self.opacity_deform = init_deform_layer(1, std=1e-7)
        self.shs_deform = init_deform_layer(16 * 3, std=1e-7)

        self.hidden_net = nn.Sequential(nn.ReLU(),nn.Linear(self.W, 2*self.W), nn.ReLU(),nn.Linear(2*self.W, self.W))


    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, opacity = None,shs_emb=None, time_feature=None, time_emb=None, semantic_feature=None, point=None, scales=None, rotations=None, dx=None):

        return self.forward_dynamic(opacity, shs_emb, time_emb,semantic_feature, point, scales, rotations, dx)

    def forward_dynamic(self, opacity_emb, shs_emb, time_emb, semantic_feature, point, scales_input,rotations_input,dx):
        shs_emb_reshaped = shs_emb.reshape(shs_emb.shape[0], 48)

        time_position_enc = torch.zeros(64, device=time_emb.device)
        for i in range(0, 64, 2):
            time_position_enc[i] = torch.sin(time_emb[0,0] / (10000 ** (2 * i / 64)))
            if i + 1 < 64:
                time_position_enc[i + 1] = torch.cos(time_emb[0,0] / (10000 ** (2 * i / 64)))
        time_position_enc=time_position_enc.unsqueeze(0)
        expanded_tim_position_enc = time_position_enc.expand(time_emb.size(0), 64)
        hidden = torch.cat((point, rotations_input, scales_input, opacity_emb, shs_emb_reshaped,semantic_feature, dx, time_emb, expanded_tim_position_enc), dim=1)

        mask = self.mask_net(hidden)
        mask=torch.sigmoid(mask)

        dx = self.pos_deform(hidden) # [N, 3]
        pts = torch.zeros_like(point[:,:3])
        pts=point+dx*mask
        if self.args.no_ds :
            
            scales = scales_input[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_input[:,:3])
            scales=scales_input+ds*mask
        dr = self.rotations_deform(hidden)
        rotations=rotations_input+dr*mask
         
        opacity = opacity_emb[:,:1]

        dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
        shs=shs_emb+dshs*mask.unsqueeze(-1)

        return pts, scales, rotations, opacity, shs, dx, dshs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network_feature(nn.Module):
    def __init__(self, args) :
        super(deform_network_feature, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, semantic_feature=None, dx=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel,  semantic_feature, dx)
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, semantic_feature=None, dx=None):

        means3D, scales, rotations, opacity, shs, dx, dshs = self.deformation_net(
                                                opacity,
                                                shs,
                                                None,
                                                times_sel,
                                                semantic_feature,
                                                point,
                                                scales,
                                                rotations,
                                                dx) # [N, 1]
        return means3D, scales, rotations, opacity, shs, dx , dshs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb