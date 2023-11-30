###############################
#######  Brani-ID Demo  #######
###############################
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import utils.demo_utils as utils 
 

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'



img_path = '/autofs/space/yogurt_001/users/pl629/data/synth/images/414569.nii' 
ckp_path = '/autofs/space/yogurt_002/users/pl629/results/finished/BrainID/feat-anat/grad/ckp/checkpoint_epoch_115.pth'

im = utils.prepare_image(img_path, device)
feats = utils.get_feature(im, ckp_path, device)

print(feats.size())