
"""
Visualization modules
"""
import os
import numpy as np  
from math import ceil
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict

from utils.misc import make_dir


def match_shape(array, shape):
    # array: (channel_dim, *orig_shape)
    array = array[None]
    if list(array.shape[2:]) != list(shape):
        array = F.interpolate(array, size=shape) 
    return array[0]

def pad_shape(array_list):
    max_shape = [0] * len(array_list[0].shape)

    for array in array_list:
        max_shape = [max(max_shape[dim], array.shape[dim]) for dim in range(len(max_shape))]  
    pad_array_list = []
    for array in array_list: 
        start = [(max_shape[dim] - array.shape[dim]) // 2 for dim in range(len(max_shape))] 
        if len(start) == 2:
            pad_array = np.zeros((max_shape[0], max_shape[1]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1]] = array
        elif len(start) == 3:
            pad_array = np.zeros((max_shape[0], max_shape[1], max_shape[2]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1], start[2] : start[2] + array.shape[2]] = array
        elif len(start) == 4:
            pad_array = np.zeros((max_shape[0], max_shape[1], max_shape[2], max_shape[3]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1], start[2] : start[2] + array.shape[2], start[3] : start[3] + array.shape[3]] = array
        
        pad_array_list.append(pad_array) 
    return pad_array_list


def even_sample(orig_len, num):
     idx = []
     length = float(orig_len)
     for i in range(num):
             idx.append(int(ceil(i * length / num)))
     return idx


def normalize(nda, channel = None):
    if channel is not None:
        nda_max = np.max(nda, axis = channel, keepdims = True)
        nda_min = np.min(nda, axis = channel, keepdims = True)
    else:
        nda_max = np.max(nda)
        nda_min = np.min(nda)
    return (nda - nda_min) / (nda_max - nda_min + 1e-7)


##############################################


class BaseVisualizer(object):
    
    def __init__(self, args, draw_border=False):
        self.args = args
        self.task = args.task
        self.draw_border = draw_border

        self.vis_spacing = args.visualizer.spacing

        if 'sr' in self.task or 'reg' in self.task or 'bf' in self.task:
            self.subject_robust = False
        else:
            self.subject_robust = True
        

    def create_image_row(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=1)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(normalize(self.create_image_row(arg))) 
        return np.concatenate(out, axis=0) 

    def prepare_for_itk(self, array): # (s, r, c, *)
        return array[:, ::-1, :]

    def prepare_for_png(self, array, normalize = False): # (s, r, c, *)
        slc = array[::self.vis_spacing[0]] # (s', r, c *)
        row = array[:, ::self.vis_spacing[1]].transpose((1, 0, 2, 3))[:, ::-1] # (s, r', c, *) -> (r', s, c, *)
        col = array[:, :, ::self.vis_spacing[2]].transpose((2, 0, 1, 3))[:, ::-1] # (s, r, c', *) -> (c', s, r, *)

        if normalize:
            slc = (slc - np.min(slc)) / (np.max(slc) - np.min(slc))
            row = (slc - np.min(slc)) / (np.max(slc) - np.min(row))
            col = (slc - np.min(slc)) / (np.max(slc) - np.min(col))
        return slc, row, col



class FeatVisualizer(BaseVisualizer):
    
    def __init__(self, args, draw_border=False):
        BaseVisualizer.__init__(self, args, draw_border)
        self.feat_vis_num = args.visualizer.feat_vis_num

    def visualize_all_multi(self, subjects, multi_inputs, multi_outputs, out_dir):
        """
        For med-id student input samples: n_samples * [ (batch_size, channel_dim, *img_shp) ]
        For med-id student output features: n_samples * [ n_levels * (batch_size, channel_dim, *img_shp) ]
        """

        if 'reg' in self.task:
            out_dir = make_dir(os.path.join(out_dir, subjects['name'][0]))
            src_inputs = [x['source'] for x in multi_inputs] # n_samples * (b, d, s, r, c)
            src_features = [x['feat']['src_feat'] for x in multi_outputs]
            self.visualize_all_multi_features(subjects['name'], src_features, src_inputs, out_dir)

            tgt_inputs = [x['target'] for x in multi_inputs] # n_samples * (b, d, s, r, c)
            tgt_features = [x['feat']['tgt_feat'] for x in multi_outputs]
            self.visualize_all_multi_features(subjects['target_name'], tgt_features, tgt_inputs, out_dir)
        else:
            multi_inputs = [x['input'] for x in multi_inputs] # n_samples * (b, d, s, r, c)
            multi_features = [x['feat'] for x in multi_outputs] 
            self.visualize_all_multi_features(subjects['name'] , multi_features, multi_inputs, out_dir)
    
    def visualize_all_multi_features(self, names, multi_features, multi_inputs, out_dir):

        n_samples = len(multi_inputs)
        n_levels = len(multi_features[0])
        
        multi_inputs_reorg = [] # batch_size * [ n_samples * (channel_dim, *img_shp) ]
        multi_features_reorg = [] # batch_size * [ n_samples * [ n_levels * (channel_dim, *img_shp) ] ]
        for i_name, _ in enumerate(names):
            multi_features_reorg.append([[multi_features[i_sample][i_level][i_name] for i_level in range(n_levels)] for i_sample in range(n_samples)]) 
            multi_inputs_reorg.append([multi_inputs[i_sample][i_name] for i_sample in range(n_samples)])

        for i_name, name in enumerate(names): 

            inputs = multi_inputs_reorg[i_name]
            features = multi_features_reorg[i_name]

            all_sample_results = defaultdict(list)
            for i_sample in range(n_samples):

                curr_input = inputs[i_sample].data.cpu().numpy() # ( d=1, s, r, c)  
                curr_input = self.prepare_for_itk(curr_input.transpose(3, 2, 1, 0)) # (d, x, y, z) -> (z, y, x, d)

                curr_feat = features[i_sample] # n_levels * (channel_dim, s, r, c)
                curr_level_feats = []

                for l in range(n_levels):
                    curr_level_feat = curr_feat[l] # (channel_dim, s, r, c)

                    sub_idx = even_sample(curr_level_feat.shape[0], self.feat_vis_num)
                    curr_level_feat = torch.stack([curr_level_feat[idx] for idx in sub_idx], dim = 0) # (sub_channel_dim, s, r, c)
    
                    curr_level_feat = match_shape(curr_level_feat, list(curr_input.shape[:-1]))
                    curr_level_feats.append(self.prepare_for_itk((curr_level_feat.data.cpu().numpy().transpose((3, 2, 1, 0))))) 
                
                all_results = self.gather(curr_input, curr_level_feats) 
                
                for l, result in enumerate(all_results): # n_level * (r, c)
                    gap = np.zeros_like(result[:, :int( result.shape[1] / (curr_input.shape[0] / self.vis_spacing[0]) )]) 
                    all_sample_results[l] += [result] + [gap] 

            for l in all_sample_results.keys():
                curr_level_all_sample_feats = np.concatenate(list(all_sample_results[l][:-1]), axis=1) # (s, n_samples * c)
                Image.fromarray(curr_level_all_sample_feats).save(os.path.join(make_dir(os.path.join(out_dir, name)), name + '_feat_l%s.png' % str(l)))


    def visualize_all(self, names, inputs, features): 
        """
        For general (single-sample) inputs: (batch_size, channel_dim, *img_shp)
        For general (single-sample) output features: n_levels * (batch_size, channel_dim, *img_shp)
        """
 
        inputs = inputs.data.cpu().numpy() # (b, d=1, s, r, c)
        n_levels = len(features) # n_levels * (b, channel_dim, s, r, c)
        
        for i_name, name in enumerate(names): 
            curr_input = self.prepare_for_itk(inputs[i_name].transpose((3, 2, 1, 0))) # (d, x, y, z) -> (z, y, x, d) 
            curr_level_feats = []
            for l in range(n_levels):
                curr_feat = features[l][i_name] # (channel_dim, s, r, c)

                sub_idx = even_sample(curr_feat.shape[0], self.feat_vis_num)
                curr_feat = torch.stack([curr_feat[idx] for idx in sub_idx], dim = 0) # (sub_channel_dim, s, r, c)
 
                curr_feat = match_shape(curr_feat, list(curr_input.shape[:-1]))
                curr_level_feats.append(self.prepare_for_itk((curr_feat.data.cpu().numpy().transpose((3, 2, 1, 0))))) 
            
            self.gather(curr_input, curr_level_feats) 
        

    def gather(self, input, feats):

        input_slc = self.prepare_for_png(input, normalize = False)[0][..., 0] # (sub_s, r, c)
        all_images = []
        for l, feat in enumerate(feats):
            slc_images = [input_slc] # only plot along axial  
            slc_feat = normalize(feat[::self.vis_spacing[0]].transpose(3, 0, 1, 2), channel = 1) # (sub_s, r, c, sub_channel_dim) -> (sub_channel_dim, sub_s, r, c)
            slc_images = [input_slc, np.zeros_like(input_slc)] + list(slc_feat) # (1 + 1 + s', r, c *)
            slc_images = pad_shape(slc_images)

            slc_image = self.create_image_grid(*slc_images)
            slc_image = (255 * slc_image).astype(np.uint8)  
            all_images.append(slc_image)
            
        return all_images



class TaskVisualizer(BaseVisualizer):

    def __init__(self, args, draw_border=False):
        BaseVisualizer.__init__(self, args, draw_border)

    def visualize_all(self, subjects, samples, outputs, out_dir, output_names = ['image'], target_names = ['image']):

        if len(output_names) == 0:
            return
        
        n_samples = len(samples)

        names = subjects['name']
        input_name = 'source' if 'reg' in self.task else 'input'
        inputs = [x[input_name].data.cpu().numpy() for x in samples] # n_samples * (b, d, s, r, c)

        Idefs = None
        if 'image_def' in samples[0].keys():
            Idefs = [x['image_def'].data.cpu().numpy() for x in samples]

        out_images = {}
        for output_name in output_names:
            if output_name in outputs[0].keys(): 
                out_images[output_name] = [x[output_name].data.cpu().numpy() for x in outputs] # n_samples * (b, d, s, r, c)  
        
        for i, name in enumerate(names): 
            case_out_dir = make_dir(os.path.join(out_dir, name)) 
            curr_inputs = [self.prepare_for_itk(inputs[i_sample][i].transpose((3, 2, 1, 0))) for i_sample in range(n_samples)] # n_samples * (d, x, y, z) -> n_samples (z, y, x, d)
            self.visualize_sample(name, curr_inputs, case_out_dir, postfix = '_%s' % input_name)

            if Idefs is not None:
                curr_Idef = [self.prepare_for_itk(Idefs[i_sample][i].transpose((3, 2, 1, 0))) for i_sample in range(n_samples)] # n_samples * (d, x, y, z) -> n_samples (z, y, x, d)

            if len(out_images) > 0:
                curr_target = {} 
                if not self.subject_robust:
                    for target_name in target_names:  
                        if target_name in samples[0]:
                            curr_target[target_name] = [self.prepare_for_itk(samples[i_sample][target_name][i].data.cpu().numpy().transpose((3, 2, 1, 0))) for i_sample in range(n_samples)]
                else:
                    for target_name in target_names:
                        if target_name in subjects:
                            if 'bias_field' in target_name:
                                curr_target[target_name] = [self.prepare_for_itk(samples[i_sample][target_name][i].data.cpu().numpy().transpose((3, 2, 1, 0))) for i_sample in range(n_samples)]
                            else:
                                curr_target[target_name] = self.prepare_for_itk(subjects[target_name][i].data.cpu().numpy().transpose((3, 2, 1, 0))) # (d=1, s, r, c) -> (z, y, x, d)  

                curr_outputs = {}
                for output_name in output_names:
                    curr_outputs[output_name] = [self.prepare_for_itk(out_images[output_name][i_sample][i].transpose((3, 2, 1, 0))) for i_sample in range(n_samples)] # n_samples * (d, x, y, z) -> n_samples (z, y, x, d) 
                
                all_images = []

                for i_sample, curr_input in enumerate(curr_inputs):
                    target_list = [curr_input]
                    for target_name in target_names:
                        print('target_name', target_name)
                        if target_name in curr_target:
                            print('add target')
                            if not self.subject_robust or 'bias_field' in target_name:
                                target_list.append(curr_target[target_name][i_sample]) 
                            else:
                                target_list.append(curr_target[target_name])
                    if Idefs is not None:
                        target_list.append(curr_Idef[i_sample])

                    output_list = []
                    for ouput_name in output_names:
                        output_list.append(curr_outputs[ouput_name][i_sample])

                    all_image = self.gather(target_list, output_list) # (row, col)
                    all_images.append(all_image) # n_sample * (row, col)
                all_images = np.concatenate(all_images, axis=1).astype(np.uint8) # (row, n_sample * col)
                Image.fromarray(all_images).save(os.path.join(case_out_dir, name + '_all_outputs.png'))

        
    def visualize_sample(self, name, input, out_dir, postfix = '_input'):
        
        n_samples = len(input)

        slc_images, row_images, col_images = [], [], []
        for i_sample in range(n_samples):
            input_slc, input_row, input_col = self.prepare_for_png(input[i_sample], normalize = False)

            slc_images.append(input_slc)
            row_images.append(input_row)
            col_images.append(input_col)

        # add row gap 
        gap = [np.zeros_like(slc_images[0])]
        all_images = slc_images + gap + row_images + gap + col_images
        all_images = pad_shape(all_images)
        all_image = self.create_image_grid(*all_images)
        all_image = (255 * all_image).astype(np.uint8)  
        Image.fromarray(all_image[:, :, 0]).save(os.path.join(out_dir, name + '_all' + postfix + '.png')) # grey scale image last channel == 1
        return 

    def gather(self, target_list = [], output_list = []):

        slc_images, row_images, col_images = [], [], []

        for add_target in target_list:
            add_target_slc, add_target_row, add_target_col = self.prepare_for_png(add_target, normalize = False)
            slc_images += [add_target_slc]
            row_images += [add_target_row]
            col_images += [add_target_col]

        for add_output in output_list:
            add_output_slc, add_output_row, add_output_col = self.prepare_for_png(add_output, normalize = False)
            slc_images += [add_output_slc]
            row_images += [add_output_row]
            col_images += [add_output_col]

        # add row gap 
        gap = [np.zeros_like(add_target_slc)]
        all_images = slc_images + gap + row_images + gap + col_images
        all_images = pad_shape(all_images)
        all_image = self.create_image_grid(*all_images)

        all_image = (255 * all_image).astype(np.uint8) 
        return all_image[:, :, 0] # shrink last channel dimension (d=1)
