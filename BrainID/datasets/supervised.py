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

from BrainID.datasets.id_synth_eval import IDSynthEval


class ContrastSpecificDataset(IDSynthEval): 
    """
    ContrastSpecific dataset
    For fine-tuning on real data
    """

    def __init__(self, args, data_dir, gt_dir, device='cpu'): 
        super(ContrastSpecificDataset, self).__init__(args, data_dir, device)

        self.args = args
        self.task = args.task
        self.device = device
        
        self.label_list_segmentation = args.base_generator.label_list_segmentation_with_csf
        self.n_neutral_labels = args.base_generator.n_neutral_labels_with_csf
        self.n_steps_svf_integration = args.base_generator.n_steps_svf_integration

        self.deform_one_hots = args.base_generator.deform_one_hots
        self.produce_surfaces = args.base_generator.produce_surfaces
        self.bspline_zooming = args.base_generator.bspline_zooming

        self.size = args.base_generator.size
        self.max_rotation = args.base_generator.max_rotation
        self.max_shear = args.base_generator.max_shear
        self.max_scaling = args.base_generator.max_scaling 
        self.nonlin_scale_min = args.base_generator.nonlin_scale_min
        self.nonlin_scale_max = args.base_generator.nonlin_scale_max
        self.nonlin_std_max = args.base_generator.nonlin_std_max 
        self.bf_scale_min = args.base_generator.bf_scale_min
        self.bf_scale_max = args.base_generator.bf_scale_max
        self.bf_std_min = args.base_generator.bf_std_min
        self.bf_std_max = args.base_generator.bf_std_max
        self.bag_scale_min = args.base_generator.bag_scale_min
        self.bag_scale_max = args.base_generator.bag_scale_max 
        self.gamma_std = args.base_generator.gamma_std
        self.noise_std_min = args.base_generator.noise_std_min
        self.noise_std_max = args.base_generator.noise_std_max

        self.exvixo_prob = 0.
        self.photo_prob = 0.
        self.hyperfine_prob = 0.
        self.bag_prob = 0.
        self.pv = False

        self.save_pathology = args.base_generator.save_pathology 
        self.pathology_prob = 0.
        self.pathology_thres_max = args.base_generator.pathology_thres_max 
        self.pathology_mu_multi = args.base_generator.pathology_mu_multi 
        self.pathology_sig_multi = args.base_generator.pathology_sig_multi 
        

        self.bias_field_prediction = 'bf' in args.task

        self.data_augmentation = args.base_generator.data_augmentation # if False, input original image 
        self.apply_deformation = args.base_generator.apply_deformation and self.data_augmentation 
        self.nonlinear_transform = args.base_generator.nonlinear_transform and self.data_augmentation and self.apply_deformation 
        self.integrate_deformation_fields = args.base_generator.integrate_deformation_fields and self.nonlinear_transform
        
        self.apply_gamma_transform = args.base_generator.apply_gamma_transform and self.data_augmentation
        self.apply_bias_field = 'bf' in self.task or (args.base_generator.apply_bias_field and self.data_augmentation)
        self.apply_resampling = args.base_generator.apply_resampling and self.data_augmentation
        self.apply_noises = args.base_generator.apply_noises and self.data_augmentation

        self.res_testing_data = [1.0, 1.0, 1.0]


        names = glob.glob(os.path.join(data_dir, '*.nii.gz')) + glob.glob(os.path.join(data_dir, '*.nii'))
        try:
            if args.test_subset is not None:
                test_len = int(len(names) * args.test_subset)
                self.names = names[-test_len:]
            elif args.train_subset is not None:
                train_len = int(len(names) * args.train_subset)
                self.names = names[:train_len]
            else:
                self.names = names
        except:
            self.names = glob.glob(os.path.join(data_dir, '*.nii.gz')) + glob.glob(os.path.join(data_dir, '*.nii'))
        
        # sanity check
        self.gt_dir = gt_dir
        gt_names = glob.glob(os.path.join(gt_dir, '*.nii.gz')) + glob.glob(os.path.join(gt_dir, '*.nii'))
        assert len(gt_names) >= len(self.names), '%s vs %s' % (len(gt_names), len(self.names))
        #assert len(gt_names) >= len(names), '%s vs %s' % (len(gt_names), len(names))


        # prepare grid
        print('Preparing grid...')
        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float, device=self.device)
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]

        # Matrix for one-hot encoding (includes a lookup-table)
        n_labels = len(self.label_list_segmentation)
        self.lut = torch.zeros(10000, dtype=torch.long, device=self.device)
        for l in range(n_labels):
            self.lut[self.label_list_segmentation[l]] = l
        self.onehotmatrix = torch.eye(n_labels, dtype=torch.float, device=self.device)

        nlat = int((n_labels - self.n_neutral_labels) / 2.0)
        self.vflip = np.concatenate([np.array(range(self.n_neutral_labels)),
                                np.array(range(self.n_neutral_labels + nlat, n_labels)),
                                np.array(range(self.n_neutral_labels, self.n_neutral_labels + nlat))])

        if 'anat' in self.task:
            self.target_name = 'image'
        elif 'sr' in self.task:
            self.target_name = 'orig'
        elif 'bf' in self.task:
            self.target_name = 'bias_field_log'
        elif 'seg' in self.task:
            self.target_name = 'label'
        else:
            raise ValueError('Not supported task type: %s' % self.task)

        print('ContrastSpecificDataset is ready!')

    def __len__(self):
        return len(self.names)


    def random_nonlinear_transform(self, img_shp, photo_mode, spac, nonlin_scale_min, nonlin_scale_max, nonlin_std_max):
        nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = utils.myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if self.integrate_deformation_fields: # NOTE: slow
            steplength = 1.0 / (2.0 ** self.n_steps_svf_integration)
            Fsvf = F * steplength
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
    
    def read_target(self, img, loc_list, flip):
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = loc_list
        if 'seg' in self.task or 'bf' in self.task:
            if x1 is None:
                S = torch.squeeze(torch.tensor(img.get_fdata().astype(int), dtype=torch.int, device=self.device))
            else:
                S = torch.squeeze(torch.tensor(img.get_fdata()[x1:x2, y1:y2, z1:z2].astype(int), dtype=torch.int, device=self.device))
            Sdef = fast_3D_interp_torch(S, xx2, yy2, zz2, 'nearest')
            if self.deform_one_hots:
                Sdef_OneHot = self.onehotmatrix[self.lut[S.long()]]
                Sdef_OneHot = fast_3D_interp_torch(Sdef_OneHot, xx2, yy2, zz2, 'linear')
            else:
                Sdef_OneHot = self.onehotmatrix[self.lut[Sdef.long()]]

            if flip:
                Sdef = torch.flip(Sdef, [0])   
                Sdef_OneHot = torch.flip(Sdef_OneHot, [0])[:, :, :, self.vflip]
            Sdef_OneHot = Sdef_OneHot.permute([3, 0, 1, 2])
            return {'label': Sdef[None], 'seg': Sdef_OneHot}, {}
        else:
            if x1 is None:
                GT = torch.squeeze(torch.tensor(img.get_fdata().astype(float), dtype=torch.float, device=self.device))
            else:
                GT = torch.squeeze(torch.tensor(img.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
            GT[GT < 0.] = 0.
            GTdef = fast_3D_interp_torch(GT, xx2, yy2, zz2, 'linear')
            maxi = torch.max(GTdef)
            GTdef = GTdef / maxi
            if flip:
                GTdef = torch.flip(GTdef, [0]) 
            #utils.viewVolume(GT, names = [self.target_name + '_orig'], save_dir = '/autofs/space/yogurt_002/users/pl629/')
            #utils.viewVolume(GTdef, names = [self.target_name], save_dir = '/autofs/space/yogurt_002/users/pl629/')
            if 'anat' in self.task: # subject-robust
                return {self.target_name: GTdef[None]}, {}
            else: # sr
                return {}, {self.target_name: GTdef[None]}

    def prepare_data(self, idx):
        I = nib.load(self.names[idx]).get_fdata()
        I = torch.tensor(np.squeeze(I), dtype=torch.float32, device=self.device)
        I -= torch.min(I)
        I /= torch.max(I)
        shp = I.shape
        return I, shp

        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read in data 
        I, shp = self.prepare_data(idx)

        A, c2, F, Fneg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], photo_mode, hyperfine_mode, spac, flip = self.generate_deformation(shp)
        
        Idef = self.read_ground_truth(I, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2])
        sample = self.generate_sample(Idef, Idef.shape, photo_mode, hyperfine_mode, spac, flip, **vars(self.args.mild_generator)) # mild corrup for real data
        subjects = {'name': os.path.basename(self.names[idx]).split(".nii")[0]}

        # Read in GT
        GT = nib.load(os.path.join(self.gt_dir, os.path.basename(self.names[idx])))
        GT_subject, GT_sample = self.read_target(GT, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], flip)

        sample.update(GT_sample) 
        subjects.update(GT_subject)
        
        return subjects, [sample] 
    

