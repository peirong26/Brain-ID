
"""
Losses
"""

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

import utils.misc as utils

import math 
import numpy as np

from pytorch3dunet.unet3d.utils import expand_as_one_hot

 


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None):
        super(NCC, self).__init__()  
        self.win = win

    def forward(self, outputs, targets):

        Ii = outputs
        Ji = targets

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class GradientSmoothness(nn.Module): 
    def __init__(self, mode = 'l2', loss_mult = 2):
        super(GradientSmoothness, self).__init__() 
        self.mode = mode
        self.loss_mult = loss_mult
        
    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def forward(self, x):
        if self.mode == 'l1':
            dif = [torch.abs(f) for f in self._diffs(x)]
        else:
            assert self.mode == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.mode
            dif = [f * f for f in self._diffs(x)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().cuda()
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


class SlicedWassLoss(nn.Module):

    def __init__(self, scale_factor = 1):
        super(SlicedWassLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, feats_x, feats_y): # n_levels * (b, d, s, r, c)

        # iterate over layers
        loss = 0
        for l in range(len(feats_x)): 
            # get dimensions
            b = feats_y[l].shape[0] # b
            dim = feats_y[l].shape[1] # d
            n = feats_y[l].shape[2] * feats_y[l].shape[3] * feats_y[l].shape[4] # s * r * c
            # linearize layer activations and duplicate example activations according to scaling factor
            feat_y = feats_y[l].view(b, dim, n).repeat(1, 1, self.scale_factor * self.scale_factor * self.scale_factor)
            feat_x = feats_x[l].view(b, dim, n * self.scale_factor * self.scale_factor * self.scale_factor)
            # sample random directions
            Ndirection = dim
            directions = torch.randn(Ndirection, dim).to(torch.device(feats_x[0].device))
            directions = directions / torch.sqrt(torch.sum(directions ** 2, dim=1, keepdim=True))
            # project activations over random directions
            projected_feat_y = torch.einsum('bdn,md->bmn', feat_y, directions)  
            projected_feat_x = torch.einsum('bdn,md->bmn', feat_x, directions)  
            # sort the projections
            sorted_feat_y = torch.sort(projected_feat_y, dim=2)[0]
            sorted_feat_x = torch.sort(projected_feat_x, dim=2)[0]
            # L2 over sorted lists
            loss += torch.mean( (sorted_feat_y - sorted_feat_x)**2 ) 
        return loss
    

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss (altered),  
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self, weight, level_weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.criterion = torch.nn.L1Loss() 
        self.level_weights = level_weights

    def forward(self, teacher_feat, student_feat):  
        percep_loss = 0.0 
        for l in range(len(teacher_feat)):
            teacher_feat[l] = teacher_feat[l].detach()
            percep_loss += self.level_weights[l] * self.criterion(teacher_feat[l], student_feat[l])  
        return percep_loss


class GradientLoss(nn.Module):
    def __init__(self, mode = 'l1', mask = False):
        super(GradientLoss, self).__init__() 
        self.mask = mask
        if mode == 'l1':
            self.loss_func = nn.L1Loss()
        elif mode == 'l2':
            self.loss_func = nn.MSEloss()
        else:
            raise ValueError('Not supported loss_func for GradientLoss:', mode)

    def gradient(self, x): 
        # x: (b, c, s, r, c) -->  dx, dy, dz: (b, c, s, r, c) 
        back = F.pad(x, [0, 1, 0, 0, 0, 0])[:, :, :, :, 1:] 
        right = F.pad(x, [0, 0, 0, 1, 0, 0])[:, :, :, 1:, :] 
        bottom = F.pad(x, [0, 0, 0, 0, 0, 1])[:, :, 1:, :, :]

        dx, dy, dz = back - x, right - x, bottom - x
        dx[:, :, :, :, -1] = 0
        dy[:, :, :, -1] = 0
        dz[:, :, -1] = 0
        return dx, dy, dz
    
    def forward_archive(self, input, target):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            dx_i[dx_t == 0.] = 0.
            dy_i[dy_t == 0.] = 0.
            dz_i[dz_t == 0.] = 0.
        return (self.loss_func(dx_i, dx_t) + self.loss_func(dy_i, dy_t) + self.loss_func(dz_i, dz_t)).mean()

    def forward(self, input, target):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            diff_dx = abs(dx_i - dx_t)
            diff_dy = abs(dy_i - dy_t)
            diff_dz = abs(dz_i - dz_t)
            diff_dx[target == 0.] = 0.
            diff_dy[target == 0.] = 0.
            diff_dz[target == 0.] = 0.
            return (diff_dx + diff_dy + diff_dz).mean()
        return (self.loss_func(dx_i, dx_t) + self.loss_func(dy_i, dy_t) + self.loss_func(dz_i, dz_t)).mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        weight = torch.tensor(weight)

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


#######################################################################################################################

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'],
                                    initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
