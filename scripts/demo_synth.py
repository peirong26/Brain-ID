###############################
####  Synthetic Data Demo  ####
###############################


import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import torch

import utils.misc as utils 
 

from BrainID.visualizer import TaskVisualizer
from BrainID.datasets import build_dataset_single 



# default & gpu cfg #
default_cfg_file = 'cfgs/default_train.yaml'
default_data_file = 'cfgs/default_dataset.yaml'
default_val_file = 'cfgs/default_val.yaml'
submit_cfg_file = 'cfgs/submit.yaml'
exp_cfg_file = 'cfgs/test/demo_synth.yaml'




def map_back_orig(img, idx, shp):
    if idx is None or shp is None:
        return img
    if len(img.shape) == 3:
        img = img[None, None]
    elif len(img.shape) == 4:
        img = img[None]
    return img[:, :, idx[0]:idx[0] + shp[0], idx[1]:idx[1] + shp[1], idx[2]:idx[2] + shp[2]]


def generate(args):

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'  
    print('device: %s' % device)

    print('out_dir:', args.out_dir)

    # ============ preparing data ... ============
    dataset = build_dataset_single(vars(args.dataset_name)[args.split], split = args.split, args = args, device = device)
    
    vis = TaskVisualizer(args)
       
    print("Start generating")
    start_time = time.time()


    dataset.mild_samples = 2
    dataset.all_samples = 4
    for itr in range(args.test_itr_limit):
        
        subj_name = os.path.basename(dataset.names[itr]).split('.nii')[0]
        save_dir = utils.make_dir(os.path.join(args.out_dir, subj_name))

        print('Processing image (%d/%d): %s' % (itr, len(dataset), dataset.names[itr]))

        for i_deform in range(args.num_deformations):
            def_save_dir = utils.make_dir(os.path.join(save_dir, 'deform-#%s' % i_deform))

            (subjects, samples) = dataset._getitem_from_id(itr)
                
            if 'aff' in subjects:
                aff = subjects['aff']
                shp = subjects['shp']
                loc_idx = subjects['loc_idx']
            else:
                aff = torch.eye((4))
                shp = loc_idx = None
            
            print('num samples:', len(samples))
            
            print('     deform:', i_deform)

            utils.viewVolume(subjects['image'], aff, names = ['image_orig'], save_dir = def_save_dir)
            if 'seg' in args.task:
                utils.viewVolume(subjects['label'], aff, names = ['label_orig'], save_dir = def_save_dir)

            for i_sample, sample in enumerate(samples):
                print('         sample:', i_sample)
                sample_save_dir = utils.make_dir(os.path.join(def_save_dir, 'sample-#%s' % i_sample))
                
                if 'input' in sample:
                    utils.viewVolume(map_back_orig(sample['input'], loc_idx, shp), aff, names = ['input'], save_dir = sample_save_dir)
                if 'sr' in args.task:
                    utils.viewVolume(map_back_orig(sample['orig'], loc_idx, shp), aff, names = ['high_field'], save_dir = sample_save_dir)

            if 'input' in sample:
                inputs = [x['input'].data.cpu().numpy() for x in samples] # n_samples * (b, d, s, r, c)
                curr_inputs = [vis.prepare_for_itk(inputs[i_sample].transpose((3, 2, 1, 0))) for i_sample in range(len(samples))] # n_samples * (d, x, y, z) -> n_samples (z, y, x, d)
                vis.visualize_sample(subj_name, curr_inputs, def_save_dir, postfix = '_inputs')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Generation time {}'.format(total_time_str))


#####################################################################################


if __name__ == '__main__':
    
    args = utils.preprocess_cfg([default_cfg_file, default_data_file, default_val_file, submit_cfg_file, exp_cfg_file])
    utils.launch_job(args, generate)

