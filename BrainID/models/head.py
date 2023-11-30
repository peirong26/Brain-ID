"""
Model heads
"""


import torch
import torch.nn as nn 

        
        
class IndepHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-independent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(IndepHead, self).__init__()
        self.out_feat_level = out_feat_level

        layers = [] # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(in_feature_num, f_maps_list[i+1], stride = 1, is_3d = is_3d) 
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        self.out_names = out_channels.keys()
        for out_name, out_channels_num in out_channels.items():
            self.add_module("final_conv_%s" % out_name, conv(f_maps_list[-1], out_channels_num, 1)) 

    def forward(self, x, *kwargs):
        x = x[self.out_feat_level]
        for layer in self.layers:
            x = layer(x)
        out = {}
        for name in self.out_names: 
            out[name] = getattr(self, f"final_conv_{name}")(x)
        return out
    

class DepHead(nn.Module):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(DepHead, self).__init__()
        self.out_feat_level = out_feat_level

        f_maps_list[0] += 1 # add one input image/contrast channel

        layers = [] # additional layers (same-size-output 3x3 conv) before final_conv, if len( f_maps_list ) > 1
        for i, in_feature_num in enumerate(f_maps_list[:-1]):
            layer = ConvBlock(in_feature_num, f_maps_list[i+1], stride = 1, is_3d = is_3d) 
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        self.out_names = out_channels.keys()
        for out_name, out_channels_num in out_channels.items():
            self.add_module("final_conv_%s" % out_name, conv(f_maps_list[-1], out_channels_num, 2 if args.losses.uncertainty is not None else 1)) 

    def forward(self, x, image):
        x = x[self.out_feat_level]
        x = torch.cat([x, image],  dim = 1)
        for layer in self.layers:
            x = layer(x)
        out = {}
        for name in self.out_names: 
            out[name] = getattr(self, f"final_conv_{name}")(x)
        return out
    

    
class MultiInputDepHead(DepHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-dependent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(MultiInputDepHead, self).__init__(args, f_maps_list, out_channels, is_3d, out_feat_level)

    def forward(self, feat_list, image_list):   
        outs = []
        for i, x in enumerate(feat_list):
            x = x[self.out_feat_level] 
            x = torch.cat([x, image_list[i]],  dim = 1) 
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names: 
                out[name] = getattr(self, f"final_conv_{name}")(x) 
            outs.append(out)
        return outs
    


class MultiInputIndepHead(IndepHead):
    """
    Task-specific head that takes a list of sample features as inputs
    For contrast-independent tasks
    """

    def __init__(self, args, f_maps_list, out_channels, is_3d, out_feat_level = -1, *kwargs):
        super(MultiInputIndepHead, self).__init__(args, f_maps_list, out_channels, is_3d, out_feat_level)

    def forward(self, feat_list, *kwargs):   
        outs = []
        for x in feat_list:
            x = x[self.out_feat_level]
            for layer in self.layers:
                x = layer(x)
            out = {}
            for name in self.out_names: 
                out[name] = getattr(self, f"final_conv_{name}")(x) 
            outs.append(out)
        return outs




class ConvBlock(nn.Module):
    """
    Specific same-size-output 3x3 convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1, is_3d=True):
        super().__init__()

        conv = nn.Conv3d if is_3d else nn.Conv2d 
        self.main = conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out



################################



def get_head(args, f_maps_list, out_channels, is_3d, out_feat_level):
    task = args.task
    if 'feat' in task:
        return IndepHead(args, f_maps_list, out_channels, is_3d, out_feat_level)
    else:
        if 'sr' in task or 'bf' in task:
            return MultiInputDepHead(args, f_maps_list, out_channels, is_3d, out_feat_level)
        else:
            return MultiInputIndepHead(args, f_maps_list, out_channels, is_3d, out_feat_level)