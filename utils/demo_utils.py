
import numpy as np
import torch

from BrainID.models import build_feat_model
from utils.checkpoint import load_checkpoint 
import utils.misc as utils 



# default & gpu cfg #
default_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/BrainID/cfgs/default_train.yaml'
default_data_file = '/autofs/space/yogurt_003/users/pl629/code/BrainID/cfgs/default_dataset.yaml'
submit_cfg_file = '/autofs/space/yogurt_003/users/pl629/code/BrainID/cfgs/submit.yaml'


def center_crop(img, win_size = [220, 220, 220]):
    # center crop
    if len(img.shape) == 4: 
        img = img[None]
    if len(img.shape) == 3: 
        img = img[None, None]
    assert len(img.shape) == 5

    orig_shp = img.shape[2:] # (1, d, s, r, c)
    if orig_shp[0] > win_size[0] or orig_shp[1] > win_size[1] or orig_shp[2] > win_size[2]:
        crop_start = [ max((orig_shp[i] - win_size[i]), 0) // 2 for i in range(3) ]
        return img[ :, :, crop_start[0] : crop_start[0] + win_size[0], 
                   crop_start[1] : crop_start[1] + win_size[1], 
                   crop_start[2] : crop_start[2] + win_size[2]], crop_start, orig_shp
    else:
        return img, [0, 0, 0], orig_shp


def prepare_image(img_path, win_size = [220, 220, 220], device = 'cpu'):
    im, aff = utils.MRIread(img_path, im_only=False, dtype='float')
    im = torch.tensor(np.squeeze(im), dtype=torch.float32, device=device)
    im = torch.nan_to_num(im)
    im -= torch.min(im)
    im /= torch.max(im)
    im, aff = utils.torch_resize(im, aff, 1.0)
    im, aff = utils.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3) 
    im, crop_start, orig_shp = center_crop(im, win_size)
    return im



@torch.no_grad()
def get_feature(inputs, ckp_path, device = 'cpu'):
    # inputs: (batch_size, 1, s, r, c)
    args = utils.preprocess_cfg([default_cfg_file, default_data_file, submit_cfg_file])

    # ============ testing ... ============
    _, model, _, _, _ = build_feat_model(args, device = device) 
    load_checkpoint(ckp_path, [model], None, ['model'], to_print = False)

    samples = [ { 'input': inputs } ]
    outputs, _ = model(samples) # dict with features

    return outputs[0]['feat'][-1] # (batch_size, 64, s, r, c)
