import os
import random
import torch
import numpy as np
import nibabel as nib


from BrainID.datasets.synth import BaseSynth
from BrainID.datasets.utils import *
from utils.misc import myzoom_torch, viewVolume
import utils.interpol as interpol 

 

class IDSynth(BaseSynth): 
    """
    IDSynth Augmentation dataset
    For intra-subject augmentation, each sample will have the same deformation field
    """

    def __init__(self, args, data_dir, device='cpu'):

        super(IDSynth, self).__init__(args, data_dir, device)

        self.mild_samples = args.mild_samples
        self.all_samples = args.all_samples 
        self.all_contrasts = args.all_contrasts
        self.num_deformations = args.num_deformations
        self.sample_size = args.base_generator.sample_size # actual input sample size (downsampled if necessary)

        self.bias_field_prediction = 'bf' in args.task

        print('IDSynth Generator is ready!')
    

    def read_data(self, G, S, D, B, photo_mode, exvixo_prob, bag_prob, bag_scale_min, bag_scale_max):
        # Decide if we're simulating ex vivo (and possibly a bag) or photos
        if photo_mode or (np.random.rand() < exvixo_prob):
            G[G>255] = 0 # kill extracerebral
            if photo_mode:
                G[G == 7] = 0
                G[G == 8] = 0
                G[G == 16] = 0 
                S[S == 24] = 0
                S[S == 7] = 0
                S[S == 8] = 0
                S[S == 46] = 0
                S[S == 47] = 0
                S[S == 15] = 0
                S[S == 16] = 0
                if D is None: # without distance maps, killing 4 is the best we can do
                    G[G == 4] = 0
                else:
                    Dpial = torch.minimum(D[...,1], D[..., 3])
                    th = 1.5 * np.random.rand() # band of random width...
                    G[G==4] = 0
                    G[(G == 0) & (Dpial < th)] = 4
            elif ((B is not None) and (np.random.rand(1) < bag_prob)):
                bag_scale = bag_scale_min + np.random.rand(1) * (bag_scale_max - bag_scale_min)
                size_TH_small = np.round(bag_scale * np.array(G.shape)).astype(int).tolist()
                bag_tness = torch.tensor(np.sort(1.0 + 20 * np.random.rand(2)), dtype=torch.float, device=self.device)
                THsmall = bag_tness[0] + (bag_tness[1] - bag_tness[0]) * torch.rand(size_TH_small, dtype=torch.float, device=self.device)
                TH = myzoom_torch(THsmall, np.array(G.shape) / size_TH_small)
                G[(B>0) & (B<TH)] = 4
        return G, S
    
    def read_ground_truth(self, idx, img, loc_list, scaling_factor_distances, flip):
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = loc_list

        G = torch.squeeze(torch.tensor(img.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
        
        S = D = I = B = None
        if self.seg_dir is not None:
            Simg = nib.load(os.path.join(self.seg_dir, os.path.basename(self.names[idx])))
            S = torch.squeeze(torch.tensor(Simg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(int), dtype=torch.int, device=self.device))
        if self.dist_dir is not None: 
            Dimg = nib.load(os.path.join(self.dist_dir, os.path.basename(self.names[idx])))
            D = torch.squeeze(torch.tensor(Dimg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device)) 
            D /= scaling_factor_distances 
        if self.im_dir is not None: 
            Iimg = nib.load(os.path.join(self.im_dir, os.path.basename(self.names[idx])))
            I = torch.squeeze(torch.tensor(Iimg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
            I[I < 0] = 0
            I /= torch.median(I[G==2])
        if self.bag_dir is not None: 
            Bimg = nib.load(os.path.join(self.bag_dir, os.path.basename(self.names[idx])))
            B = torch.squeeze(torch.tensor(Bimg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device)) 
            B /= scaling_factor_distances

        Sdef_OneHot = Idef = 0.
        if S is not None:
            if self.deform_one_hots:
                Sonehot = self.onehotmatrix[self.lut[S.long()]]
                Sdef_OneHot = fast_3D_interp_torch(Sonehot, xx2, yy2, zz2, 'linear')
            else:
                Sdef = fast_3D_interp_torch(S, xx2, yy2, zz2, 'nearest', self.device)
                Sdef_OneHot = self.onehotmatrix[self.lut[Sdef.long()]]
        if I is not None:
            Idef = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear') 

        # Flip 50% of times
        if flip: # only flip subject-robust for output, other params to be flipped in process_sample()
            Sdef_OneHot = torch.flip(Sdef_OneHot, [0])[:, :, :, self.vflip]
            Idef = torch.flip(Idef, [0])  
            # mask real image if needed
            Idef *= (1.0 - Sdef_OneHot[:, :, :, 0])
            Sdef = torch.flip(Sdef, [0])

        # prepare for input
        Idef = Idef[None, ...] # add one channel dimension 
        Sdef_OneHot = Sdef_OneHot.permute([3, 0, 1, 2])
        Sdef = Sdef[None]

        return G, S, Sdef, Sdef_OneHot, D, Idef, B
    
    def process_sample(self, mus, sigmas, photo_mode, spac, thickness, resolution, flip, G, loc_list, 
                       gamma_std, bf_scale_min, bf_scale_max, bf_std_min, bf_std_max, noise_std_min, noise_std_max):
        xx2, yy2, zz2 = loc_list
        
        Gr = torch.round(G).long()
        
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape, dtype=torch.float, device=self.device)
        if self.pv:
            mask = (G!=Gr)
            SYN[mask] = 0
            Gv = G[mask]
            isv = torch.zeros(Gv.shape, dtype=torch.float, device=self.device )
            pw = (Gv<=3) * (3-Gv)
            isv += pw * mus[2] + pw * sigmas[2] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pg = (Gv<=3) * (Gv-2) + (Gv>3) * (4-Gv)
            isv += pg * mus[3] + pg * sigmas[3] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pcsf = (Gv>=3) * (Gv-3)
            isv += pcsf * mus[4] + pcsf * sigmas[4] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            SYN[mask] = isv 

        SYN[SYN < 0] = 0


        if 'sr' in self.task:
            SYN_cerebral = SYN
            SYN_cerebral[Gr == 0] = 0
            SYN_cerebral[Gr > 255] = 0
            SYN_cerebral = fast_3D_interp_torch(SYN_cerebral, xx2, yy2, zz2, 'linear')

        SYN_def = fast_3D_interp_torch(SYN, xx2, yy2, zz2, 'linear')

        # Gamma transform
        gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=self.device)
        SYN_gamma = 300.0 * (SYN_def / 300.0) ** gamma

        # Bias field
        bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
        size_BF_small = np.round(bf_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_BF_small[1] = np.round(self.size[1]/spac).astype(int)
        BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=torch.float, device=self.device) * torch.randn(size_BF_small, dtype=torch.float, device=self.device)
        BFlog = myzoom_torch(BFsmall, np.array(self.size) / size_BF_small)
        BF = torch.exp(BFlog)
        SYN_bf = SYN_gamma * BF

        # Model Resolution
        stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) /np.pi * thickness / self.res_training_data
        stds[thickness<=self.res_training_data] = 0.0 # no blur if thickness is equal to the resolution of the training data
        SYN_blur = gaussian_blur_3d(SYN_bf, stds, self.device)
        new_size = (np.array(self.size) * self.res_training_data / resolution).astype(int)

        factors = np.array(new_size) / np.array(self.size)
        delta = (1.0 - factors) / (2.0 * factors)
        vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
        vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
        vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
        II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
        II = torch.tensor(II, dtype=torch.float, device=self.device)
        JJ = torch.tensor(JJ, dtype=torch.float, device=self.device)
        KK = torch.tensor(KK, dtype=torch.float, device=self.device)

        SYN_small = fast_3D_interp_torch(SYN_blur, II, JJ, KK, 'linear') 
        noise_std = torch.tensor(noise_std_min + (noise_std_max - noise_std_min) * np.random.rand(1), dtype=torch.float, device=self.device)
        SYN_noisy = SYN_small + noise_std * torch.randn(SYN_small.shape, dtype=torch.float, device=self.device)
        SYN_noisy[SYN_noisy < 0] = 0

        # Back to original resolution
        if self.bspline_zooming:
            SYN_resized = interpol.resize(SYN_noisy, shape=self.size, anchor='edge', interpolation=3, bound='dct2', prefilter=True) 
        else:
            SYN_resized = myzoom_torch(SYN_noisy, 1 / factors) 
        maxi = torch.max(SYN_resized)
        SYN_final = SYN_resized / maxi


        # Flip 50% of times
        if flip:
            SYN_final = torch.flip(SYN_final, [0])  
            BFlog = torch.flip(BFlog, [0])  
            if 'sr' in self.task: 
                SYN_cerebral = torch.flip(SYN_cerebral, [0])   


        # prepare for input
        SYN_final = SYN_final[None, ...] # add one channel dimension 
        BFlog = BFlog[None, ...]


        sample = {'input': SYN_final}
        if self.bias_field_prediction:
            sample.update({'bias_field_log': BFlog})
        if 'sr' in self.task: 
            maxi = torch.max(SYN_cerebral)
            SYN_cerebral = SYN_cerebral / maxi
            sample.update({'orig': SYN_cerebral[None, ...]})
        return sample
    
    def generate_sample(self, mus, sigmas, G, S, D, B, photo_mode, hyperfine_mode, spac, flip,
                        xx2, yy2, zz2,
                        exvixo_prob=0.25, bag_prob=0.5, bag_scale_min=0.02, bag_scale_max=0.08,
                        gamma_std=0.1, bf_scale_min=0.02, bf_scale_max=0.04, bf_std_min=0.1, bf_std_max=0.5, 
                        noise_std_min=5, noise_std_max=15, **kwargs): 
        
        # Sample resolution
        resolution, thickness = self.random_sampler(photo_mode, hyperfine_mode, spac)   

        # Read in data
        G, S = self.read_data(G, S, D, B, photo_mode, exvixo_prob, bag_prob, bag_scale_min, bag_scale_max)

        return self.process_sample(mus, sigmas, photo_mode, spac, thickness, resolution, flip, G, [xx2, yy2, zz2],
                                     gamma_std, bf_scale_min, bf_scale_max, bf_std_min, bf_std_max, noise_std_min, noise_std_max)

    def get_deformation(self, Gshp):
        # pre-setup: case-wise global random values
        if np.random.rand() < self.photo_prob:
            photo_mode = True
            hyperfine_mode = False
        else:
            photo_mode = False
            hyperfine_mode = np.random.rand() < self.hyperfine_prob

        spac = 2.0 + 10 * np.random.rand() if photo_mode else None 
        flip = np.random.randn() < 0.5

        scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 \
            = self.generate_deformation(photo_mode, spac, Gshp)
        
        return photo_mode, hyperfine_mode, spac, flip, scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2
    
    def _getitem_from_id(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        Gimg = nib.load(self.names[idx])
        Gshp = Gimg.shape

        photo_mode, hyperfine_mode, spac, flip, \
            scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.get_deformation(Gshp)
        
        # Read in data 
        G, S, Sdef, Sdef_OneHot, D, Idef, B = self.read_ground_truth(idx, Gimg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], scaling_factor_distances, flip)

        samples = []
        for sample_i in range(self.all_samples):
            mus, sigmas = self.get_contrast(photo_mode)
            if sample_i < self.mild_samples:
                samples.append(self.generate_sample(mus, sigmas, G, S, D, B, photo_mode, hyperfine_mode, spac, flip, 
                                xx2, yy2, zz2, **vars(self.args.mild_generator))) 
            else:
                samples.append(self.generate_sample(mus, sigmas, G, S, D, B, photo_mode, hyperfine_mode, spac, flip, 
                                xx2, yy2, zz2, **vars(self.args.severe_generator)))

        subjects = {'name': os.path.basename(self.names[idx]).split(".nii")[0], 'image': Idef}
        if 'seg' in self.task or self.bias_field_prediction:
            subjects.update({'seg': Sdef_OneHot, 'label': Sdef})
        return subjects, samples 
    
    def __getitem__(self, idx):
        return self._getitem_from_id(idx)




class DeformIDSynth(IDSynth):
    """
    DeformIDSynth Augmentation dataset
    For intra-subject augmentation, each sample will have different deformation field
    """

    def __init__(self, args, data_dir, device='cpu'): 
        super(DeformIDSynth, self).__init__(args, data_dir, device)
        print('DeformIDSynth Generator is ready!')
    
    
    def _getitem_from_id(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        Gimg = nib.load(self.names[idx])
        Gshp = Gimg.shape

        samples = []

        for i_sample in range(self.all_samples): 
            
            if i_sample < self.num_deformations:
                print('generate new deformation')
                photo_mode, hyperfine_mode, spac, flip, \
                    scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.get_deformation(Gshp)
            
            G, S, Sdef, Sdef_OneHot, D, Idef, B = self.read_ground_truth(idx, Gimg, [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], scaling_factor_distances, flip)

            if i_sample < self.all_contrasts:
                mus, sigmas = self.get_contrast(photo_mode)

            if i_sample < self.mild_samples:
                sample = self.generate_sample(mus, sigmas, G, S, D, B, photo_mode, hyperfine_mode, spac, flip, 
                                xx2, yy2, zz2, **vars(self.args.mild_generator))
            else:
                sample = self.generate_sample(mus, sigmas, G, S, D, B, photo_mode, hyperfine_mode, spac, flip, 
                                xx2, yy2, zz2, **vars(self.args.severe_generator))
                
            sample.update({'image': Idef})
            if 'seg' in self.task:
                sample.update({'seg': Sdef_OneHot, 'label': Sdef})

            samples.append(sample) 
        
        subjects = {'name': os.path.basename(self.names[idx]).split(".nii")[0]}
        return subjects, samples
    

