import torch
import numpy as np
import os
EPS = 1e-6

def sub2ind(height, width, y, x):
    return y*width + x

def ind2sub(height, width, ind):
    y = ind // width
    x = ind % width
    return y, x
    
def get_lr_str(lr):
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    return lrn
    
def strnum(x):
    s = '%g' % x
    if '.' in s:
        if x < 1.0:
            s = s[s.index('.'):]
        s = s[:min(len(s),4)]
    return s

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def print_stats(name, tensor):
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)
    
def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size(), dtype=d.dtype, device=d.device)
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out

def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda', on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def gridcloud2d(B, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy

def reduce_masked_mean(x, mask, dim=None, keepdim=False, broadcast=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    if not broadcast:
        for (a,b) in zip(x.size(), mask.size()):
            if not a==b:
                print('some shape mismatch:', x.shape, mask.shape)
            assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
    mean = numer/denom
    return mean
        
def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert(x.size() == mask.size())
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()
