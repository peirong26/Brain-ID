import torch.nn as nn
import torch.nn.functional as F

from .buildingblocks import DoubleConv, create_decoders, create_encoders
from .utils import get_class, number_of_features_per_level


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is_3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is_unit_vector = False, is_3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            self.f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        else:
            assert isinstance(self.f_maps, list) or isinstance(self.f_maps, tuple)
            self.f_maps = f_maps

        assert len(self.f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, self.f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is_3d)

        # create decoder path
        self.decoders = create_decoders(self.f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is_3d)

        self.is_unit_vector = is_unit_vector

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        
        if self.is_unit_vector:
            x = F.normalize(x, dim=1)

        return x 
    

    def get_feature(self, x): 
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x) 
            encoders_features.insert(0, x) 
        encoders_features = encoders_features[1:]

        decoders_features = [x]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
            decoders_features.append(x) 
        if self.is_unit_vector:
            decoders_features[-1] = F.normalize(decoders_features[-1], dim=1)
        return decoders_features



class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, f_maps, layer_order='gcl', num_groups=8, num_levels=5, is_unit_vector=False, conv_padding=1, **kwargs):

        super(UNet3D, self).__init__(in_channels=in_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_unit_vector=is_unit_vector,
                                     conv_padding=conv_padding,
                                     is_3d=True) 

    

class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, args, in_channels, f_maps, conv_padding=1, **kwargs):

        super(UNet2D, self).__init__(in_channels=in_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=args.layer_order,
                                     num_groups=args.num_groups,
                                     num_levels=args.num_levels,
                                     conv_padding=conv_padding,
                                     is_3d=True)
        


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)

