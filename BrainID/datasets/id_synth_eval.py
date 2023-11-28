import os
import random

import glob
import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset

from BrainID.datasets.utils import *
import utils.misc as utils 
import utils.interpol as interpol 



###############################
#  Teacher/Student GENERATOR  #
###############################

class IDSynthEval(Dataset): 
    """
    IDSynthEval dataset:
        for each __getitem__(), output *same* deformation, ~resolution, ~contrast within one subject [for evaluation]
    """

    def __init__(self, args, data_dir, device='cpu'): 
        super(IDSynthEval, self).__init__()
        self.args = args

        self.task = args.task
        
        self.label_list_segmentation = args.base_test_generator.label_list_segmentation_with_csf
        self.n_neutral_labels = args.base_test_generator.n_neutral_labels_with_csf
        self.n_steps_svf_integration = args.base_test_generator.n_steps_svf_integration

        self.deform_one_hots = args.base_test_generator.deform_one_hots
        self.produce_surfaces = args.base_test_generator.produce_surfaces
        self.bspline_zooming = args.base_test_generator.bspline_zooming

        self.max_rotation = args.base_test_generator.max_rotation
        self.max_shear = args.base_test_generator.max_shear
        self.max_scaling = args.base_test_generator.max_scaling 
        self.nonlin_scale_min = args.base_test_generator.nonlin_scale_min
        self.nonlin_scale_max = args.base_test_generator.nonlin_scale_max
        self.nonlin_std_max = args.base_test_generator.nonlin_std_max 
        self.bf_scale_min = args.base_test_generator.bf_scale_min
        self.bf_scale_max = args.base_test_generator.bf_scale_max
        self.bf_std_min = args.base_test_generator.bf_std_min
        self.bf_std_max = args.base_test_generator.bf_std_max
        self.bag_scale_min = args.base_test_generator.bag_scale_min
        self.bag_scale_max = args.base_test_generator.bag_scale_max 
        self.gamma_std = args.base_test_generator.gamma_std
        self.noise_std_min = args.base_test_generator.noise_std_min
        self.noise_std_max = args.base_test_generator.noise_std_max

        self.exvixo_prob = args.base_test_generator.exvixo_prob
        self.photo_prob = args.base_test_generator.photo_prob
        self.hyperfine_prob = args.base_test_generator.hyperfine_prob
        self.bag_prob = args.base_test_generator.bag_prob 
        self.pv = args.base_test_generator.pv

        self.save_pathology = args.base_test_generator.save_pathology 
        self.pathology_prob = args.base_test_generator.pathology_prob 
        self.pathology_thres_max = args.base_test_generator.pathology_thres_max 
        self.pathology_mu_multi = args.base_test_generator.pathology_mu_multi 
        self.pathology_sig_multi = args.base_test_generator.pathology_sig_multi 
        
        self.device = device

        self.mild_samples = args.test_mild_samples
        self.all_samples = args.test_all_samples 

        self.data_augmentation = args.base_test_generator.data_augmentation # if False, input original image 
        self.apply_deformation = args.base_test_generator.apply_deformation and self.data_augmentation 
        self.nonlinear_transform = args.base_test_generator.nonlinear_transform and self.data_augmentation and self.apply_deformation 
        self.integrate_deformation_fields = args.base_test_generator.integrate_deformation_fields and self.nonlinear_transform
        
        self.apply_gamma_transform = args.base_test_generator.apply_gamma_transform and self.data_augmentation
        self.apply_bias_field = args.base_test_generator.apply_bias_field and self.data_augmentation
        self.apply_resampling = args.base_test_generator.apply_resampling and self.data_augmentation
        self.hyperfine_prob = args.base_test_generator.hyperfine_prob if self.apply_resampling else 0.
        self.apply_noises = args.base_test_generator.apply_noises and self.data_augmentation

        self.bias_field_prediction = 'bf' in args.task

        self.res_testing_data = [1.0, 1.0, 1.0]

        names = glob.glob(os.path.join(data_dir, '*.nii.gz')) + glob.glob(os.path.join(data_dir, '*.nii'))
        if args.test_subset is not None:
            test_len = int(len(names) * args.test_subset)
            self.names = names[-test_len:]
        else:
            self.names = names
        

        print('IDSynthEval Generator is ready!')
        print('Number of testing cases:', len(self.names))

    def __len__(self):
        return len(self.names)
    

    def random_affine_transform(self, shp, max_rotation, max_shear, max_scaling):
        rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
        shears = (2 * max_shear * np.random.rand(3) - max_shear)
        scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
        A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)

        # sample center
        #max_shift = (torch.tensor(np.array(shp[0:3]) - img_size, dtype=torch.float, device=self.device)) / 2 # no shift in testing augmentation
        #max_shift[max_shift < 0] = 0
        
        c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device) 

        return A, c2
    
    def random_nonlinear_transform(self, shp, photo_mode, spac, nonlin_scale_min, nonlin_scale_max, nonlin_std_max):
        nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(shp)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(shp[1]/spac).astype(int)
        nonlin_std = nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = utils.myzoom_torch(Fsmall, np.array(shp) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if self.integrate_deformation_fields: # NOTE: slow
            steplength = 1.0 / (2.0 ** self.n_steps_svf_integration)
            Fsvf = F * steplength
            print('-- compute inv deform --')
            for _ in range(self.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0], self.yy + Fsvf[:, :, :, 1], self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0], self.yy + Fsvf_neg[:, :, :, 1], self.zz + Fsvf_neg[:, :, :, 2], 'linear')
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg

    def deform_image(self, shp, A, c2, F):
        if F is not None:
            # deform the images (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
        else:
            xx1 = self.xc
            yy1 = self.yc
            zz1 = self.zc

        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]  
        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2
    
    def read_ground_truth(self, img, loc_list):
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = loc_list
        img = img[x1:x2, y1:y2, z1:z2]
        if xx2 is None:
            return img
        else:
            return fast_3D_interp_torch(img, xx2, yy2, zz2, 'linear') 
    
    def generate_sample(self, I_def, I_shp, photo_mode, hyperfine_mode, spac, flip,
                        gamma_std=0.1, bf_scale_min=0.02, bf_scale_max=0.04, bf_std_min=0.1, bf_std_max=0.5, 
                        noise_std_min=5, noise_std_max=15, **kwargs): 

        I_sr = I_def.clone()

        if self.data_augmentation:
            # Gamma transform
            if self.apply_gamma_transform:
                gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=self.device)
                I_def = 300.0 * (I_def / 300.0) ** gamma

            # Bias field
            if self.apply_bias_field: 
                bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
                size_BF_small = np.round(bf_scale * np.array(I_shp)).astype(int).tolist()
                if photo_mode:
                    size_BF_small[1] = np.round(I_shp[1]/spac).astype(int)
                BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=torch.float, device=self.device) * torch.randn(size_BF_small, dtype=torch.float, device=self.device)
                BFlog = utils.myzoom_torch(BFsmall, np.array(I_shp) / size_BF_small)
                BF = torch.exp(BFlog)
                I_def = I_def * BF

            # Model Resolution
            if self.apply_resampling:

                # Sample resolution
                resolution, thickness = self.random_sampler(photo_mode, hyperfine_mode, spac)  

                stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) /np.pi * thickness / self.res_testing_data
                stds[thickness<=self.res_testing_data] = 0.0 # no blur if thickness is equal to the resolution of the training data
                I_def = gaussian_blur_3d(I_def, stds, self.device)
                new_size = (np.array(I_shp) * self.res_testing_data / resolution).astype(int)

                factors = np.array(new_size) / np.array(I_shp)
                delta = (1.0 - factors) / (2.0 * factors)
                vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
                vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
                vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
                II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
                II = torch.tensor(II, dtype=torch.float, device=self.device)
                JJ = torch.tensor(JJ, dtype=torch.float, device=self.device)
                KK = torch.tensor(KK, dtype=torch.float, device=self.device)

                I_def = fast_3D_interp_torch(I_def, II, JJ, KK, 'linear') 

            if self.apply_noises:
                noise_std = torch.tensor(noise_std_min + (noise_std_max - noise_std_min) * np.random.rand(1), dtype=torch.float, device=self.device)
                I_def = I_def + noise_std * torch.randn(I_def.shape, dtype=torch.float, device=self.device)
                I_def[I_def < 0] = 0

            # Back to original resolution if resampled
            if self.apply_resampling:
                if self.bspline_zooming:
                    I_def = interpol.resize(I_def, shape=I_shp, anchor='edge', interpolation=3, bound='dct2', prefilter=True) 
                else:
                    I_def = utils.myzoom_torch(I_def, 1 / factors) 

            maxi = torch.max(I_def)
            I_def = I_def / maxi

        if flip:
            I_def = torch.flip(I_def, [0])  
            I_sr = torch.flip(I_sr, [0]) 
            if self.apply_bias_field or self.bias_field_prediction:
                BFlog = torch.flip(BFlog, [0]) 

        sample = {'image_def': I_sr[None], 'input': I_def[None]} # add one channel dimension  
        if (self.apply_bias_field or self.bias_field_prediction) and self.data_augmentation: 
            sample.update({'bias_field_log': BFlog[None]})
        return sample
        
    def generate_deformation(self, Gshp):

        if self.apply_deformation:
            #print('add deform')
            if np.random.rand() < self.photo_prob:
                photo_mode = True
                hyperfine_mode = False
            else:
                photo_mode = False
                hyperfine_mode = np.random.rand() < self.hyperfine_prob

            spac = 2.0 + 10 * np.random.rand() if photo_mode else None

            if self.data_augmentation:
                flip = np.random.randn() < 0.5
            else:
                flip = False

            # sample affine deformation
            A, c2 = self.random_affine_transform(Gshp, self.max_rotation, self.max_shear, self.max_scaling)
            
            # sample nonlinear deformation
            F = Fneg = None 
            if self.nonlinear_transform:
                F, Fneg = self.random_nonlinear_transform(Gshp, photo_mode, spac, self.nonlin_scale_min, self.nonlin_scale_max, self.nonlin_std_max) 
                
            # deform the images 
            xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(Gshp, A, c2, F)
        
            return A, c2, F, Fneg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], photo_mode, hyperfine_mode, spac, flip
        
        return None, None, None, None, [None] * 9, False, False, None, False
        
    def random_sampler(self, photo_mode, hyperfine_mode, spac): 
        if photo_mode: 
            resolution = np.array([self.res_testing_data[0], spac, self.res_testing_data[2]])
            thickness = np.array([self.res_testing_data[0], 0.0001, self.res_testing_data[2]])
        elif hyperfine_mode:
            resolution = np.array([1.6, 1.6, 5.])
            thickness = np.array([1.6, 1.6, 5.])
        else:
            resolution, thickness = resolution_sampler()
        return resolution, thickness

    def prepare_data(self, image_path):
        im, aff = utils.MRIread(image_path, im_only=False, dtype='float')
        im = torch.tensor(np.squeeze(im), dtype=torch.float32, device=self.device)
        im, aff = utils.torch_resize(im, aff, 1.0)
        im, aff = utils.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        im_orig = im.clone()
        while len(im.shape) > 3:  # in case it's rgb
            im = torch.mean(im, axis=-1)
        im = im - torch.min(im)
        im = im / torch.max(im)
        W = (np.ceil(np.array(im.shape) / 32.0) * 32).astype('int')
        idx = np.floor((W - im.shape) / 2).astype('int')
        I = torch.zeros(*W, dtype=torch.float32, device=self.device)
        I[idx[0]:idx[0] + im.shape[0], idx[1]:idx[1] + im.shape[1], idx[2]:idx[2] + im.shape[2]] = im

        # update grid
        xx, yy, zz = np.meshgrid(range(im.shape[0]), range(im.shape[1]), range(im.shape[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor((np.array(im.shape) - 1) / 2, dtype=torch.float, device=self.device)
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]

        return im_orig, im.shape, aff, I, idx


    def _getitem_from_id(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        I_orig, shp, aff, I, loc_idx = self.prepare_data(self.names[idx])

        A, c2, F, Fneg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], photo_mode, hyperfine_mode, spac, flip = self.generate_deformation(shp)
        
        # Read in data 
        Idef = self.read_ground_truth(I, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2])

        samples = []
        for sample_i in range(self.all_samples):
            if sample_i < self.mild_samples:
                samples.append(self.generate_sample(Idef, shp, photo_mode, hyperfine_mode, spac, flip, **vars(self.args.mild_test_generator))) 
            else:
                samples.append(self.generate_sample(Idef, shp, photo_mode, hyperfine_mode, spac, flip, **vars(self.args.severe_test_generator)))  
        
        subjects = {'name': os.path.basename(self.names[idx]).split(".nii")[0], 
                    'image': I_orig[None], 'aff': aff, 'loc_idx': loc_idx, 'shp': torch.tensor(shp).to(self.device),
                    'A': A, 'c': c2, 'F': F, 'Fneg': Fneg}
        return subjects, samples 
    
    def __getitem__(self, idx):
        return self._getitem_from_id(idx)





class DeformIDSynthEval(IDSynthEval):
    """
    DeformIDSynthEval Augmentation dataset:
        for each __getitem__(), output *~*deformation, ~resolution, ~contrast within one subject [for evaluation]
    """

    def __init__(self, args, data_dir, device='cpu'): 
        super(DeformIDSynthEval, self).__init__(args, data_dir, device)
        print('DeformIDSynthEval Generator is ready!')

    def _getitem_from_id(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        I_orig, shp, aff, I, loc_idx = self.prepare_data(self.names[idx])

        samples = []
        for i_sample in range(self.all_samples):  
            A, c2, F, Fneg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], photo_mode, hyperfine_mode, spac, flip = self.generate_deformation(shp) 
            Idef = self.read_ground_truth(I, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2])
            if i_sample < self.mild_samples:
                sample = self.generate_sample(Idef, shp, photo_mode, hyperfine_mode, spac, flip, **vars(self.args.mild_test_generator))
            else:
                sample = self.generate_sample(Idef, shp, photo_mode, hyperfine_mode, spac, flip, **vars(self.args.severe_test_generator))
            sample.update({'A': A, 'c': c2, 'F': F, 'Fneg': Fneg})
            samples.append(sample)   

        subjects = {'name': os.path.basename(self.names[idx]).split(".nii")[0], 'image_orig': I_orig[None], 'aff': aff, 'loc_idx': loc_idx, 'shp': torch.tensor(shp)}
        return subjects, samples 
    

