"""
Backbone modules.
"""

import torch.nn as nn 

from BrainID.models.unet3d.model import UNet3D, ResidualUNet3D, ResidualUNetSE3D, UNet2D


backbone_options = {
    'unet2d': UNet2D,
    'unet3d': UNet3D, 
    'res_unet3d': ResidualUNet3D,
    'res_unet3d_se': ResidualUNetSE3D,
}



####################################


def build_backbone(args):

    backbone = backbone_options[args.backbone](args, args.in_channels, args.f_maps)
    
    return backbone

