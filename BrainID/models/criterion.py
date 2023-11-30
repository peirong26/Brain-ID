"""
Criterion modules.
"""

import numpy as np
import torch
import torch.nn as nn

from BrainID.models.losses import GradientLoss, gaussian_loss, laplace_loss


uncertainty_loss = {'gaussian': gaussian_loss, 'laplace': laplace_loss}


class SetCriterion(nn.Module):
    """ 
    This class computes the loss for BrainID.
    """
    def __init__(self, args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetCriterion, self).__init__()
        self.args = args
        self.weight_dict = weight_dict
        self.loss_names = loss_names 
 
        self.mse = nn.MSELoss()

        self.loss_regression_type = args.losses.uncertainty if args.losses.uncertainty is not None else 'l1' 
        self.loss_regression = uncertainty_loss[args.losses.uncertainty] if args.losses.uncertainty is not None else nn.L1Loss()

        self.grad = GradientLoss('l1')

        self.bflog_loss = nn.L1Loss() if args.losses.bias_field_log_type == 'l1' else self.mse

        if 'contrastive' in self.loss_names:
            self.temp_alpha = args.contrastive_temperatures.alpha
            self.temp_beta = args.contrastive_temperatures.beta
            self.temp_gamma = args.contrastive_temperatures.gamma
        
        # initialize weights

        weights_with_csf = torch.ones(args.n_labels_with_csf).to(device)
        weights_with_csf[args.base_generator.label_list_segmentation_with_csf==77] = args.relative_weight_lesions # give (more) importance to lesions
        weights_with_csf = weights_with_csf / torch.sum(weights_with_csf)

        weights_without_csf = torch.ones(args.n_labels_without_csf).to(device)
        weights_without_csf[args.base_generator.label_list_segmentation_without_csf==77] = args.relative_weight_lesions # give (more) importance to lesions
        weights_without_csf = weights_without_csf / torch.sum(weights_without_csf)

        self.weights_ce = weights_with_csf[None, :, None, None, None]
        self.weights_dice = weights_with_csf[None, :]
        self.weights_dice_sup = weights_without_csf[None, :] 

        self.csf_ind = torch.tensor(np.where(np.array(args.base_generator.label_list_segmentation_with_csf)==24)[0][0])
        self.csf_v = torch.tensor(np.concatenate([np.arange(0, self.csf_ind), np.arange(self.csf_ind+1, args.n_labels_with_csf)]))  

        self.loss_map = {
            'seg_ce': self.loss_seg_ce,
            'seg_dice': self.loss_seg_dice,
            'dist': self.loss_dist,
            'sr': self.loss_sr,
            'sr_grad': self.loss_sr_grad,
            'image': self.loss_image,
            'image_grad': self.loss_image_grad,
            "bias_field_log": self.loss_bias_field_log,
            'supervised_seg': self.loss_supervised_seg, 
            'contrastive': self.loss_feat_contrastive, 
        }

    def loss_feat_contrastive(self, outputs, *kwargs):
        """
        outputs: [feat1, feat2]
        feat shape: (b, feat_dim, s, r, c)
        """
        feat1, feat2 = outputs[0]['feat'][-1], outputs[1]['feat'][-1]
        num = torch.sum(torch.exp(feat1 * feat2 / self.temp_alpha), dim = 1) 
        den = torch.zeros_like(feat1[:, 0]) 
        for i in range(feat1.shape[1]): 
            den1 = torch.exp(feat1[:, i] ** 2 / self.temp_beta)
            den2 = torch.exp((torch.sum(feat1[:, i][:, None] * feat1, dim = 1) - feat1[:, i] ** 2) / self.temp_gamma) 
            den += den1 + den2 
        loss_contrastive = torch.mean(- torch.log(num / den)) 
        return {'loss_contrastive': loss_contrastive}

    def loss_seg_ce(self, outputs, targets, *kwargs):
        """
        Cross entropy of segmentation
        """
        loss_seg_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['seg'], min=1e-5)) * self.weights_ce * targets['seg'], dim=1)) 
        return {'loss_seg_ce': loss_seg_ce}

    def loss_seg_dice(self, outputs, targets, *kwargs):
        """
        Dice of segmentation
        """
        loss_seg_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['seg'] * targets['seg']).sum(dim=[2, 3, 4])) 
                                                       / torch.clamp((outputs['seg'] + targets['seg']).sum(dim=[2, 3, 4]), min=1e-5)))
        return {'loss_seg_dice': loss_seg_dice}
    
    def loss_dist(self, outputs, targets, *kwargs): 
        loss_dist = self.mse(outputs['dist'], targets['dist'])
        return {'loss_image': loss_dist}
    
    def loss_sr(self, outputs, targets, samples):
        if self.loss_regression_type != 'l1':
            loss_sr = self.loss_regression(outputs['image'], outputs['image_sigma'], samples['orig'])
        else:
            loss_sr = self.loss_regression(outputs['image'], samples['orig'])  
        return {'loss_sr': loss_sr}
    
    def loss_sr_grad(self, outputs, targets, samples):
        loss_sr_grad = self.grad(outputs['image'], samples['orig'])
        return {'loss_sr_grad': loss_sr_grad}

    def loss_image(self, outputs, targets, *kwargs):
        if self.loss_regression_type != 'l1':
            loss_image = self.loss_regression(outputs['image'], outputs['image_sigma'], targets['image'])
        else:
            loss_image = self.loss_regression(outputs['image'], targets['image'])
        return {'loss_image': loss_image}
    
    def loss_image_grad(self, outputs, targets, *kwargs):
        loss_image_grad = self.grad(outputs['image'], targets['image'])
        return {'loss_image_grad': loss_image_grad}
    
    def loss_bias_field_log(self, outputs, targets, samples):
        bf_soft_mask = 1. - targets['seg'][:, 0]
        loss_bias_field_log = self.bflog_loss(outputs['bias_field_log'] * bf_soft_mask, samples['bias_field_log'] * bf_soft_mask)
        return {'loss_bias_field_log': loss_bias_field_log}
    
    def loss_supervised_seg(self, outputs, targets, *kwargs):
        """
        Supervised segmentation differences (for dataset_name == synth)
        """
        onehot_withoutcsf = targets['seg'].clone()
        onehot_withoutcsf = onehot_withoutcsf[:, self.csf_v, ...]
        onehot_withoutcsf[:, 0, :, :, :] = onehot_withoutcsf[:, 0, :, :, :] + targets['seg'][:, self.csf_ind, :, :, :]

        loss_supervised_seg = torch.sum(self.weights_dice_sup * (1.0 - 2.0 * ((outputs['supervised_seg'] * onehot_withoutcsf).sum(dim=[2, 3, 4])) 
                                                                 / torch.clamp((outputs['supervised_seg'] + onehot_withoutcsf).sum(dim=[2, 3, 4]), min=1e-5)))

        return {'loss_supervised_seg': loss_supervised_seg} 

    def get_loss(self, loss_name, outputs, targets, *kwargs):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        return self.loss_map[loss_name](outputs, targets, *kwargs)

    def forward(self, outputs, targets, *kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs, targets, *kwargs))
        return losses
    


class SetMultiCriterion(SetCriterion):
    """ 
    This class computes the loss for BrainID with a list of results as inputs.
    """
    def __init__(self, args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetMultiCriterion, self).__init__(args, weight_dict, loss_names, device)
        self.all_samples = args.all_samples

    def get_loss(self, loss_name, outputs_list, targets, samples_list):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        total_loss = 0.
        for i_sample, outputs in enumerate(outputs_list): 
            total_loss += self.loss_map[loss_name](outputs, targets, samples_list[i_sample])['loss_' + loss_name]
        return {'loss_' + loss_name: total_loss / self.all_samples}
    
    def forward(self, outputs_list, targets, samples_list):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs_list, targets, samples_list))
        return losses

