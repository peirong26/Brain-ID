import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.io.matlab import loadmat


from BrainID.datasets.utils import *
from utils.misc import myzoom_torch
import utils.interpol as interpol 



class BaseSynth(Dataset):
    """BaseSynth dataset"""

    def __init__(self, args, data_dir, device='cpu'):

        self.args = args

        self.task = args.task
        
        self.label_list_segmentation = args.base_generator.label_list_segmentation_with_csf
        self.n_neutral_labels = args.base_generator.n_neutral_labels_with_csf
        self.n_steps_svf_integration = args.base_generator.n_steps_svf_integration

        self.nonlinear_transform = args.base_generator.nonlinear_transform
        self.deform_one_hots = args.base_generator.deform_one_hots
        self.integrate_deformation_fields = args.base_generator.integrate_deformation_fields
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

        self.exvixo_prob = args.base_generator.exvixo_prob
        self.photo_prob = args.base_generator.photo_prob
        self.hyperfine_prob = args.base_generator.hyperfine_prob
        self.bag_prob = args.base_generator.bag_prob 
        self.pv = args.base_generator.pv
        
        self.device = device

        # Paths to the different subdirectories
        self.gen_dir = os.path.join(data_dir, 'label_maps_generation')
        self.seg_dir = os.path.join(data_dir, 'label_maps_segmentation')
        self.dist_dir = os.path.join(data_dir, 'Dmaps')
        self.im_dir = os.path.join(data_dir, 'images')
        self.bag_dir = os.path.join(data_dir, 'DmapsBag')
        self.surface_dir = os.path.join(data_dir, 'surfaces')

        if os.path.exists(self.seg_dir) is False:
            print('Directory with target segmentations not found; target segmentations will not be generated')
            self.seg_dir = None
        if os.path.exists(self.dist_dir) is False:
            print('Directory with distance maps not found; distance maps will not be generated')
            self.dist_dir = None
        if os.path.exists(self.im_dir) is False:
            print('Directory with real images not found; real images will not be generated')
            self.im_dir = None
        if os.path.exists(self.bag_dir) is False:
            print('Directory with distance maps for bag simulation not found; fake bags will not be generated')
            self.bag_dir = None
            
        if self.produce_surfaces is False:
            print('Surface generation switched off by user')
            self.surface_dir = None
        else:
            if os.path.exists(self.surface_dir) is False:
                raise Exception('User is asking for surfaces but directory with surface files was not found!')
            if self.integrate_deformation_fields is False:
                raise Exception('Using surfaces requires integrating deformation fields; you need to switch on integrate_deformation_fields option')
            
        names = glob.glob(os.path.join(self.gen_dir, '*.nii.gz')) + glob.glob(os.path.join(self.gen_dir, '*.nii'))
        if args.train_subset is not None:
            train_len = int(len(self.names) * args.train_subset)
            self.names = names[:train_len]
        else:
            self.names = names
        
        # Get resolution of training data
        aff = nib.load(self.names[0]).affine
        self.res_training_data = np.sqrt(np.sum(aff[:-1, :-1], axis=0))

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

        print('BaseSynth Generator is ready!')

    def __len__(self):
        return len(self.names)
    
    def random_affine_transform(self, shp, max_rotation, max_shear, max_scaling, random_shift = True):
        rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
        shears = (2 * max_shear * np.random.rand(3) - max_shear)
        scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
        scaling_factor_distances = np.prod(scalings) ** .33333333333 # we divide distance maps by this, not perfect, but better than nothing
        A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)

        # sample center
        if random_shift:
            max_shift = (torch.tensor(np.array(shp[0:3]) - self.size, dtype=torch.float, device=self.device)) / 2
            max_shift[max_shift < 0] = 0
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device) + (2 * (max_shift * torch.rand(3, dtype=float, device=self.device)) - max_shift)
        else:
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device)
        return scaling_factor_distances, A, c2

    def random_nonlinear_transform(self, photo_mode, spac, nonlin_scale_min, nonlin_scale_max, nonlin_std_max):
        nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
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

    def generate_deformation(self, photo_mode, spac, Gshp, random_shift = True):

        # sample affine deformation
        scaling_factor_distances, A, c2 = self.random_affine_transform(Gshp, self.max_rotation, self.max_shear, self.max_scaling, random_shift)
        
        # sample nonlinear deformation 
        if self.nonlinear_transform:
            F, _ = self.random_nonlinear_transform(photo_mode, spac, self.nonlin_scale_min, self.nonlin_scale_max, self.nonlin_std_max) 
        else:
            F = None

        # deform the images 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(Gshp, A, c2, F)
        
        return scaling_factor_distances, xx2, yy2, zz2, x1, y1, z1, x2, y2, z2
    
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

    def generate_surface(self, idx, Fneg, A, c2):
        filename = os.path.basename(self.names[idx])
        if filename.endswith('.nii.gz'):
            filename = filename[:-7] + '.mat'
        else:
            filename = filename[:-4] + '.mat'
        mat = loadmat(os.path.join(self.surface_dir, filename ))
        Vlw = torch.tensor(mat['Vlw'], dtype=torch.float, device=self.device)
        Flw = torch.tensor(mat['Flw'], dtype=torch.int, device=self.device)
        Vrw = torch.tensor(mat['Vrw'], dtype=torch.float, device=self.device)
        Frw = torch.tensor(mat['Frw'], dtype=torch.int, device=self.device)
        Vlp = torch.tensor(mat['Vlp'], dtype=torch.float, device=self.device)
        Flp = torch.tensor(mat['Flp'], dtype=torch.int, device=self.device)
        Vrp = torch.tensor(mat['Vrp'], dtype=torch.float, device=self.device)
        Frp = torch.tensor(mat['Frp'], dtype=torch.int, device=self.device)

        Ainv = torch.inverse(A)
        Vlw -= c2[None, :]
        Vlw = Vlw @ torch.transpose(Ainv, 0, 1)
        Vlw += fast_3D_interp_torch(Fneg, Vlw[:, 0] + self.c[0], Vlw[:, 1]+self.c[1], Vlw[:, 2] + self.c[2], 'linear')
        Vlw += self.c[None, :]
        Vrw -= c2[None, :]
        Vrw = Vrw @ torch.transpose(Ainv, 0, 1)
        Vrw += fast_3D_interp_torch(Fneg, Vrw[:, 0] + self.c[0], Vrw[:, 1]+self.c[1], Vrw[:, 2] + self.c[2], 'linear')
        Vrw += self.c[None, :]
        Vlp -= c2[None, :]
        Vlp = Vlp @ torch.transpose(Ainv, 0, 1)
        Vlp += fast_3D_interp_torch(Fneg, Vlp[:, 0] + self.c[0], Vlp[:, 1] + self.c[1], Vlp[:, 2] + self.c[2], 'linear')
        Vlp += self.c[None, :]
        Vrp -= c2[None, :]
        Vrp = Vrp @ torch.transpose(Ainv, 0, 1)
        Vrp += fast_3D_interp_torch(Fneg, Vrp[:, 0] + self.c[0], Vrp[:, 1] + self.c[1], Vrp[:, 2] + self.c[2], 'linear')
        Vrp += self.c[None, :]
        return Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp
    
    def read_data(self, img, idx, scaling_factor_distances, photo_mode, loc_list, exvixo_prob, bag_prob, bag_scale_min, bag_scale_max):
        x1, x2, y1, y2, z1, z2 = loc_list
        
        G = torch.squeeze(torch.tensor(img.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
        
        S = D = I = B = PTH = None
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

        return G, S, D, I, PTH
    
    def process_sample(self, photo_mode, spac, thickness, resolution, flip, mus, sigmas, G, S, D, I, PTH, loc_list,
                       gamma_std, bf_scale_min, bf_scale_max, bf_std_min, bf_std_max, noise_std_min, noise_std_max,
                       Vlw=None, Flw=None, Vrw=None, Frw=None, Vlp=None, Flp=None, Vrp=None, Frp=None):
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

        SYN_def = fast_3D_interp_torch(SYN, xx2, yy2, zz2, 'linear')

        Sdef_OneHot = Ddef = Idef = Pdef = 0.
        Sdef = fast_3D_interp_torch(S, xx2, yy2, zz2, 'nearest')
        if S is not None:
            if self.deform_one_hots:
                Sonehot = self.onehotmatrix[self.lut[S.long()]]
                Sdef_OneHot = fast_3D_interp_torch(Sonehot, xx2, yy2, zz2, 'linear')
            else:
                Sdef_OneHot = self.onehotmatrix[self.lut[Sdef.long()]]
        if D is not None:
            Ddef = fast_3D_interp_torch(D, xx2, yy2, zz2, 'linear', default_value_linear=torch.max(D))
        if I is not None:
            Idef = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear')

        # Gamma transform
        gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=self.device)
        SYN_gamma = 300.0 * (SYN_def / 300.0) ** gamma

        # Bias field
        bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
        size_BF_small = np.round(bf_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_BF_small[1] = np.round(self.size[1]/spac).astype(int)
        BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=torch.float, device=self.device) * \
            torch.randn(size_BF_small, dtype=torch.float, device=self.device)
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
            if 'sr' in self.task: 
                SYN_def = torch.flip(SYN_def, [0])   
            if 'seg' in self.task:
                Sdef = torch.flip(Sdef, [0])   
            Sdef_OneHot = torch.flip(Sdef_OneHot, [0])[:, :, :, self.vflip]
            Ddef = torch.flip(Ddef, [0])[:, :, :, [2,3,0,1]]
            Idef = torch.flip(Idef, [0])
            BFlog = torch.flip(BFlog, [0])
            if self.produce_surfaces:
                Vlw[:, 0] = Idef.shape[0] - 1 - Vlw[:, 0]
                Vrw[:, 0] = Idef.shape[0] - 1 - Vrw[:, 0]
                Vlp[:, 0] = Idef.shape[0] - 1 - Vlp[:, 0]
                Vrp[:, 0] = Idef.shape[0] - 1 - Vrp[:, 0]
                Vlw, Vrw = Vrw, Vlw
                Vlp, Vrp = Vrp, Vlp
                Flw, Frw = Frw, Flw
                Flp, Frp = Frp, Flp

        # mask real image if needed
        if Idef is not None:
            Idef *= (1.0 - Sdef_OneHot[:, :, :, 0])

        # prepare for input
        SYN_final = SYN_final[None, ...] # add one channel dimension
        Sdef_OneHot = Sdef_OneHot.permute([3, 0, 1, 2])
        Ddef = Ddef.permute([3, 0, 1, 2])
        Idef = Idef[None, ...] 
        BFlog = BFlog[None, ...] 

        sample = { 
                'input': SYN_final, 
                'seg': Sdef_OneHot, 'dist': Ddef, 'image': Idef, 'bias_field_log': BFlog,
            }
        if self.produce_surfaces:
            sample.update({'Vlw': Vlw, 'Flw': Flw, 'Vrw': Vrw, 'Frw': Frw, 'Vlp': Vlp, 'Flp': Flp, 'Vrp': Vrp, 'Frp': Frp}) 
        if 'seg' in self.task:
            sample.update({'label': Sdef[None]})
        if 'sr' in self.task:
            maxi = torch.max(SYN_def)
            SYN_def = SYN_def / maxi
            sample.update({'orig': SYN_def[None]})
        return sample

    def get_contrast(self, photo_mode):
        # Sample Gaussian image
        mus = 25 + 200 * torch.rand(10000, dtype=torch.float, device=self.device)
        sigmas = 5 + 20 * torch.rand(10000, dtype=torch.float, device=self.device)
        if photo_mode or np.random.rand(1)<0.5: # set the background to zero every once in a while (or always in photo mode)
            mus[0] = 0
        return mus, sigmas
    
    def random_sampler(self, photo_mode, hyperfine_mode, spac): 
        if photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.0001, self.res_training_data[2]])
        elif hyperfine_mode:
            resolution = np.array([1.6, 1.6, 5.])
            thickness = np.array([1.6, 1.6, 5.])
        else:
            resolution, thickness = resolution_sampler()
        return resolution, thickness
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        Gimg = nib.load(self.names[idx])
        Gshp = Gimg.shape

        # The first thing we do is sampling the resolution and deformation, as this will give us a bounding box
        # of the image region we need, so we don't have to read the whole thing from disk (only works for uncompressed niftis!

        # pre-setup: case-wise global random values
        if np.random.rand() < self.photo_prob:
            photo_mode = True
            hyperfine_mode = False
        else:
            photo_mode = False
            hyperfine_mode = np.random.rand() < self.hyperfine_prob

        spac = 2.0 + 10 * np.random.rand() if photo_mode else None 
        flip = np.random.randn() < 0.5

        # sample affine deformation
        scaling_factor_distances, A, c2 = self.random_affine_transform(Gshp, self.max_rotation, self.max_shear, self.max_scaling)

        # sample nonlinear deformation
        F, Fneg = self.random_nonlinear_transform(photo_mode, spac, self.nonlin_scale_min, self.nonlin_scale_max, self.nonlin_std_max)


        # Start by deforming surfaces if needed (we need the inverse transform!)
        if self.produce_surfaces:
            Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp = self.generate_surface(idx, Fneg, A, c2)
        else:
            Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp = [None] * 8

        # deform the images 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_image(Gshp, A, c2, F)


        # Read in data
        G, S, D, I, PTH = self.read_data(Gimg, idx, scaling_factor_distances, photo_mode, [x1, x2, y1, y2, z1, z2],
                                    self.exvixo_prob, self.bag_prob, self.bag_scale_min, self.bag_scale_max)

        # Sampler
        resolution, thickness = self.random_sampler(photo_mode, hyperfine_mode, spac) 

        mus, sigmas = self.get_contrast(photo_mode)
        
        if self.produce_surfaces:
            return {'name': os.path.basename(self.names[idx]).split('.nii')[0]}, \
                self.process_sample(photo_mode, spac, thickness, resolution, flip, mus, sigmas, G, S, D, I, PTH, [xx2, yy2, zz2],
                                            self.gamma_std, self.bf_scale_min, self.bf_scale_max, self.bf_std_min, self.bf_std_max, self.noise_std_min, self.noise_std_max,
                                            Vlw, Flw, Vrw, Frw, Vlp, Flp, Vrp, Frp)
        else:
            return {'name': os.path.basename(self.names[idx]).split('.nii')[0]}, self.process_sample(photo_mode, spac, thickness, resolution, flip, mus, sigmas, G, S, D, I, PTH, [xx2, yy2, zz2],
                                            self.gamma_std, self.bf_scale_min, self.bf_scale_max, self.bf_std_min, self.bf_std_max, self.noise_std_min, self.noise_std_max)


