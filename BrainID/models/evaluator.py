
"""
Evaluator modules
"""

import os

import math
import numpy as np
import torch
import torch.nn as nn 
from pytorch_msssim import ssim, ms_ssim 


from utils.misc import MRIread, MRIwrite


#########################################

# some constants
label_list_segmentation = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41,
                                    42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60] # 33
n_neutral_labels = 7
n_labels = len(label_list_segmentation)
nlat = int((n_labels - n_neutral_labels) / 2.0)
vflip = np.concatenate([np.array(range(n_neutral_labels)),
                        np.array(range(n_neutral_labels + nlat, n_labels)),
                        np.array(range(n_neutral_labels, n_neutral_labels + nlat))]) 

def get_onehot(label, device):
    # Matrix for one-hot encoding (includes a lookup-table)
    lut = torch.zeros(10000, dtype=torch.long, device=device)
    for l in range(n_labels):
        lut[label_list_segmentation[l]] = l
    onehotmatrix = torch.eye(n_labels, dtype=torch.float, device=device)

    label = torch.from_numpy(np.squeeze(label))
    onehot = onehotmatrix[lut[label.long()]]

    return onehot.permute([3, 0, 1, 2])

def align_shape(nda1, nda2):
    if nda1.shape != nda2.shape:
        print('pre-align', nda1.shape, nda2.shape)
        s = min(nda1.shape[0], nda2.shape[0])
        r = min(nda1.shape[1], nda2.shape[1])
        c = min(nda1.shape[2], nda2.shape[2])
        nda1 = nda1[:s, :r, :c]
        nda2 = nda2[:s, :r, :c]
        print('post-align', nda1.shape, nda2.shape)
    return nda1, nda2



class Evaluator:
    """ 
    This class computes the evaluation scores for BrainID.
    """
    def __init__(self, args, metric_names, device):

        self.args = args
        self.metric_names = metric_names 
        self.device = device
 
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.win_sigma = args.ssim_win_sigma

        self.metric_map = { 
            'seg_dice': self.get_dice,

            'feat_l1': self.get_l1,
            'recon_l1': self.get_l1,
            'sr_l1': self.get_l1,

            'bf_normalized_l2': self.get_normalized_l2,
            'bf_corrected_l1': self.get_l1,

            'recon_psnr': self.get_psnr,
            'sr_psnr': self.get_psnr,

            'feat_ssim': self.get_ssim,
            'recon_ssim': self.get_ssim,
            'sr_ssim': self.get_ssim,

            'feat_ms_ssim': self.get_ms_ssim,
            'recon_ms_ssim': self.get_ms_ssim,
            'sr_ms_ssim': self.get_ms_ssim,
        }

    def get_dice(self, metric_name, output, target, *kwargs):
        """
        Dice of segmentation
        """
        dice = torch.mean((2.0 * ((output * target).sum(dim=[2, 3, 4]))
                          / torch.clamp((output + target).sum(dim=[2, 3, 4]), min=1e-5)))
        return {metric_name: dice.cpu().numpy()}

    def get_normalized_l2(self, metric_name, output, target, *kwargs):
        w = torch.sum(output * target) / (torch.sum(output ** 2) + 1e-7)
        l2 = 0. + torch.sqrt( torch.sum( (w * output - target) ** 2 ) / (torch.sum(target ** 2) + 1e-7) )
        return {metric_name: l2.cpu().numpy()}

    def get_l1(self, metric_name, output, target, nonzero_only=False, *kwargs):
        if nonzero_only: # compute only within face_aware_region #
            nonzero_mask = target!=0
            l1 = (abs(target - output) * nonzero_mask).sum(dim=0) / nonzero_mask.sum(dim=0) 
        else:
            l1 = self.l1(output, target)
        return {metric_name: l1.cpu().numpy()}
        
    def get_psnr(self, metric_name, output, target, *kwargs): 
        mse = self.mse(output, target).cpu().numpy()
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * math.log10(np.max(target.cpu().numpy()) / math.sqrt(mse))
        return {metric_name: psnr}
    
    def get_ssim(self, metric_name, output, target, *kwargs):
        '''
        Ref: https://github.com/jorge-pessoa/pytorch-msssim
        '''
        output = (output - output.min()) / (output.max() - output.min())
        target = (target - target.min()) / (target.max() - target.min())
        ss = ssim(output, target, data_range = 1.0, size_average = False, win_sigma = self.win_sigma)
        return {metric_name: ss.mean().cpu().numpy()}
        
    def get_ms_ssim(self, metric_name, output, target, *kwargs):
        '''
        Ref: https://github.com/jorge-pessoa/pytorch-msssim
        '''
        output = (output - output.min()) / (output.max() - output.min())
        target = (target - target.min()) / (target.max() - target.min())
        try:
            ms_ss = ms_ssim(output, target, data_range = 1.0, size_average = False, win_sigma = self.win_sigma)
            return {metric_name: ms_ss.mean().cpu().numpy()}
        except:
            print('Error in MS-SSIM: Image too small for Multi-scale SSIM computation. Skipping...')
            return {metric_name: float('nan')}

    def get_score(self, metric_name, output, target, **kwargs):
        assert metric_name in self.metric_map, f'do you really want to compute {metric_name} metric?'
        return self.metric_map[metric_name](metric_name, output, target, **kwargs)

    def eval(self, pred_path, target_path, clamp = False, is_seg = False, normalize = False, add_mask = False, flip = False, kill_target_labels = [], **kwargs): 

        pred = MRIread(pred_path, im_only=True, dtype='int' if 'label' in os.path.basename(pred_path) else 'float')
        target, aff = MRIread(target_path, im_only=False, dtype='int' if 'label' in os.path.basename(target_path) else 'float')

        #print(pred.shape, target.shape)
        pred, target = align_shape(pred, target)
        
        if flip:
            pred = np.flip(pred, 0)
        
        for label in kill_target_labels:
            target[target == label] = 0
            pred[pred == label] = 0

        if add_mask and '_masked' not in pred_path:
            pred[target == 0] = 0
            pred[pred < 0] = 0
            MRIwrite(pred, aff, pred_path.split('.')[0] + '_masked.nii.gz')

        if normalize:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        if is_seg:
            pred = get_onehot(pred.copy(), self.device)
            target = get_onehot(target, self.device)
        else:
            pred = torch.tensor(np.squeeze(pred), dtype=torch.float32, device=self.device)
            target = torch.tensor(np.squeeze(target), dtype=torch.float32, device=self.device) 

        if clamp:
            pred = torch.clamp(pred, min = 0., max = 1.)
            target = torch.clamp(target, min = 0., max = 1.)

        if len(pred.shape) == 3:
            pred = pred[None, None]
            target = target[None, None]
        elif len(pred.shape) == 4: # seg
            pred = pred[None] 
            target = target[None]
        assert len(pred.shape) == len(target.shape) == 5

        score = {}
        for metric_name in self.metric_names:
            score.update(self.get_score(metric_name, pred, target, **kwargs))
        
        return score


