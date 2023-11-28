
import numpy as np

import torch
from torch.nn.functional import conv3d

import time


# Prepare generator
def resolution_sampler():
    r = np.random.rand()
    if r < 0.25: # 1mm isotropic
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
    elif r < 0.5: # clinical (low-res in one dimension)
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
        idx = np.random.randint(3)
        resolution[idx] = 2.5 + 6 * np.random.rand()
        thickness[idx] = np.min([resolution[idx], 4.0 + 2.0 * np.random.rand()])
    elif r < 0.75:  # low-field: stock sequences (always axial)
        resolution = np.array([1.3, 1.3, 5.0]) + 0.4 * np.random.rand(3)
        thickness = resolution.copy()
    else: # low-field: isotropic-ish (also good for scouts)
        resolution = 2.0 + 3.0 * np.random.rand(3)
        thickness = resolution.copy()
    return resolution, thickness


#######################
# Auxiliary functions #
#######################

def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A

def make_gaussian_kernel(sigma, device):

    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2*sl+1, dtype=torch.float, device=device)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel

def gaussian_blur_3d(input, stds, device):
    blurred = input[None, None, :, :, :]
    if stds[0]>0:
        kx = make_gaussian_kernel(stds[0], device=device)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1]>0:
        ky = make_gaussian_kernel(stds[1], device=device)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2]>0:
        kz = make_gaussian_kernel(stds[2], device=device)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)



def fast_3D_interp_torch_slow(X, II, JJ, KK, mode, device, default_value_linear=0.0):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], dtype=torch.float, device=device)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:,:,:,channel] = aux[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], dtype=torch.float, device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, dtype=torch.float, device=device)
            Yc[ok] = c.float()
            Yc[~ok] = default_value_linear
            Y[...,channel] = Yc

        if Y.shape[-1]==1:
            Y = Y[...,0]

    else:
        raise Exception('mode must be linear or nearest')

    return Y



def fast_3D_interp_torch(X, II, JJ, KK, mode, default_value_linear=0.0):

    if II is None:
        #print('no deform')
        return X

    #print('has deform')
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None] 
        Y = X[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = (IIv - fx)[..., None]
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = (JJv - fy)[..., None]
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = (KKv - fz)[..., None]
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None] 
        
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Y = torch.zeros([*II.shape, X.shape[3]], device=X.device) 
        Y[ok] = c.float()
        Y[~ok] = default_value_linear   

        if Y.shape[-1]==1:
            Y = Y[...,0] 
    else:
        raise Exception('mode must be linear or nearest')

    return Y



def fast_3D_interp_torch_test(X, II, JJ, KK, mode, default_value_linear=0.):
    time.sleep(3)

    start_time = time.time()
    Y2 = fast_3D_interp_torch_slow(X, II, JJ, KK, mode, X.device, default_value_linear)
    print('slow', X.shape[-1], time.time() - start_time)

    time.sleep(3)

    start_time = time.time()
    Y1 = fast_3D_interp_torch(X, II, JJ, KK, mode, default_value_linear)
    print('fast', X.shape[-1], time.time() - start_time)

    time.sleep(3)

    print('diff', (Y2 - Y1).mean(), (Y2 - Y1).max())
    return Y1