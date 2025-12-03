import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import cv2
from PIL import Image
import imageio
import h5py
import re

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.3

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        # photometric augmentation params
        from torchvision.transforms import ColorJitter, GaussianBlur
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, valid):

        target_H, target_W = self.crop_size
        H, W = img1.shape[:2]

        # start by upsampling the data if we need to
        if target_H > H or target_W > W:
            scale = max(target_H / H, target_W / W)
            new_H, new_W = int(np.ceil(H * scale)), int(np.ceil(W * scale))
            img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)
            img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)
            flow = cv2.resize(flow, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT) * [scale, scale]
            valid = cv2.resize(valid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > img1.shape[0]:
            pad_b = self.crop_size[0] - img1.shape[0]
        if self.crop_size[1] > img1.shape[1]:
            pad_r = self.crop_size[1] - img1.shape[1]
        if pad_b != 0 or pad_r != 0:
            img1 = np.pad(img1, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            img2 = np.pad(img2, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            flow = np.pad(flow, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            valid = np.pad(valid, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=((0, 0), (0, 0)))
            
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * [scale_x, scale_y]
            valid = cv2.resize(valid, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]

        if img1.shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        if img1.shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.3

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        # photometric augmentation params
        from torchvision.transforms import ColorJitter, GaussianBlur
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > img1.shape[0]:
            pad_b = self.crop_size[0] - img1.shape[0]
        if self.crop_size[1] > img1.shape[1]:
            pad_r = self.crop_size[1] - img1.shape[1]
        if pad_b != 0 or pad_r != 0:
            img1 = np.pad(img1, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            img2 = np.pad(img2, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            flow = np.pad(flow, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            valid = np.pad(valid, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=((0, 0), (0, 0)))
            
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

def just_crop_flow(rgb0, rgb1, flow, valid, crop_size):
    H, W = rgb0.shape[:2]

    if H == crop_size[0]:
        y0 = 0
    else:
        y0 = np.random.randint(0, max(H - crop_size[0], 1))

    if W == crop_size[1]:
        x0 = 0
    else:
        x0 = np.random.randint(0, max(W - crop_size[1], 1))

    rgb0 = rgb0[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    rgb1 = rgb1[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    flow = flow[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    valid = valid[y0:y0+crop_size[0], x0:x0+crop_size[1]]
    return rgb0, rgb1, flow, valid

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file:', fn)
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]

def read_gen(file_name, pil=False):
    ext = os.path.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext == '.flo5':
        return readFlo5Flow(file_name)
    return imageio.imread(file_name)[:,:,0:3]

class MetaflowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 use_augs=False,
                 crop_size=(368, 496),
                 is_training=True,
                 shuffle=True,
                 dataset_name=None
    ):
        print('loading metaflow dataset...')

        self.data_root = data_root
        self.use_augs = use_augs
        self.crop_size = crop_size

        if use_augs:
            aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 1.0, 'do_flip': True}
            sparse_aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 1.0, 'do_flip': True}
            self.augmentor = FlowAugmentor(**aug_params)
            self.sparse_augmentor = SparseFlowAugmentor(**sparse_aug_params)

        self.rgb0_paths = []
        self.rgb1_paths = []
        self.flow_paths = []
        self.valid_paths = []
        self.dnames = []

        import socket
        host = socket.gethostname()
        print('host', host)
        
        if dataset_name is None:
            # use all the training datasets
            self.dataset_names = ['autoflow', 'chairs', 'driving', 'hd1k', 'kitti', 'monkaa', 'sintel', 'spring', 'tartanair', 'things', 'viper']
        else:
            self.dataset_names = [dataset_name] if isinstance(dataset_name, str) else dataset_name

        if is_training:
            dset = 'train'
        else:
            dset = 'val'

        if 'autoflow' in self.dataset_names:
            # AUTOFLOW (448x576)
            count = 0
            dataset_location = os.path.join(data_root, 'autoflow')

            metafolders = glob.glob(os.path.join(dataset_location, "static*"))
            for meta in metafolders:
                if is_training:
                    subfolders = glob.glob(os.path.join(meta, "table*[0-8]"))
                else:
                    subfolders = glob.glob(os.path.join(meta, "table*9"))
                    
                for sub in subfolders:
                    # print('sub', sub)
                    rgb0_path = os.path.join(sub, 'im0.png')
                    rgb1_path = os.path.join(sub, 'im1.png')
                    flow_path = os.path.join(sub, 'forward.flo')
                    self.rgb0_paths.append(rgb0_path)
                    self.rgb1_paths.append(rgb1_path)
                    self.flow_paths.append(flow_path)
                    self.valid_paths.append(None)
                    self.dnames.append('autoflow')
                    count += 1
            print('added %d samples from autoflow (%s)' % (count, dset))

        if 'chairs' in self.dataset_names:
            # CHAIRS (384x512)
            count = 0
            dataset_location = os.path.join(data_root, 'flyingchairs')
                
            if is_training:
                myrange = np.arange(0,22872-2000)
            else:
                myrange = np.arange(22872-2000,22872)
            for i in myrange:
                self.rgb0_paths.append('%s/FlyingChairs_release/data/%05d_img1.ppm' % (dataset_location, i+1))
                self.rgb1_paths.append('%s/FlyingChairs_release/data/%05d_img2.ppm' % (dataset_location, i+1))
                self.flow_paths.append('%s/FlyingChairs_release/data/%05d_flow.flo' % (dataset_location, i+1))
                self.valid_paths.append(None)
                self.dnames.append('chairs')
                count += 1
            print('added %d samples from chairs (%s)' % (count, dset))
        
        if 'driving' in self.dataset_names:
            # DRIVINGDATASET (540x960)
            count = 0
            dataset_location = os.path.join(data_root, 'drivingdataset')
                
            for dstype in ['frames_cleanpass_webp', 'frames_finalpass']:
                for fotype in ['35mm_focallength', '15mm_focallength']:
                    for sce in ['scene_backwards', 'scene_forwards']:
                        for spd in ['fast', 'slow']:
                            for cam in ['left', 'right']:
                                for direction in ['into_future', 'into_past']:
                                    idir = os.path.join(dataset_location, dstype, fotype, sce, spd, cam)
                                    fdir = os.path.join(dataset_location, 'optical_flow', fotype, sce, spd, direction, cam)
                                    if 'webp' in dstype:
                                        images = sorted(glob.glob(os.path.join(idir, '*.webp')) )
                                    else:
                                        images = sorted(glob.glob(os.path.join(idir, '*.png')) )
                                    flows = sorted(glob.glob(os.path.join(fdir, '*.pfm')) )
                                    for i in range(len(flows)-1):
                                        if direction == 'into_future':
                                            self.rgb0_paths.append(images[i])
                                            self.rgb1_paths.append(images[i+1])
                                            self.flow_paths.append(flows[i])
                                            self.valid_paths.append(None)
                                        elif direction == 'into_past':
                                            self.rgb0_paths.append(images[i+1])
                                            self.rgb1_paths.append(images[i])
                                            self.flow_paths.append(flows[i+1])
                                            self.valid_paths.append(None)
                                        self.dnames.append('driving')
                                        count += 1
            print('added %d samples from driving (%s)' % (count, dset))
            
        if 'hd1k' in self.dataset_names:
            # HD1K (1080x2560)
            dataset_location = os.path.join(data_root, 'hd1k')
            seq_ix = 0
            count = 0
            while 1:
                flows = sorted(glob.glob(os.path.join(dataset_location, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
                images = sorted(glob.glob(os.path.join(dataset_location, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))
                if len(flows) == 0:
                    break
                for i in range(len(flows)-1):
                    if (i==0 and (not is_training)) or (i>0 and is_training): # reserve zeroth for val
                        for mult in range(30): # copy sea-raft
                            self.rgb0_paths.append(images[i])
                            self.rgb1_paths.append(images[i+1])
                            self.flow_paths.append(flows[i])
                            self.valid_paths.append(None)
                            self.dnames.append('hd1k')
                            count += 1
                seq_ix += 1
            print('added %d samples from hd1k (%s)' % (count, dset))
        
        if 'kitti' in self.dataset_names:
            # KITTI (375x1242)
            dataset_location = os.path.join(data_root, 'kittiflow2015/training')
                
            images1 = sorted(glob.glob(os.path.join(dataset_location, 'image_2/*_10.png')))
            images2 = sorted(glob.glob(os.path.join(dataset_location, 'image_2/*_11.png')))
            flow_list = sorted(glob.glob(os.path.join(dataset_location, 'flow_occ/*_10.png')))
            # make train/val split
            t_len = len(images1)*9//10
            if is_training:
                images1 = images1[:t_len]
                images2 = images2[:t_len]
                flow_list = flow_list[:t_len]
            else:
                images1 = images1[t_len:]
                images2 = images2[t_len:]
                flow_list = flow_list[t_len:]
            images1 = images1 * 80 # copy sea-raft
            images2 = images2 * 80
            flow_list = flow_list * 80
            self.rgb0_paths += images1
            self.rgb1_paths += images2
            self.flow_paths += flow_list
            count = len(flow_list)
            self.dnames += ['kitti']*count
            self.valid_paths += [None]*count
            print('added %d samples from kitti (%s)' % (count, dset))

        
        if 'monkaa' in self.dataset_names:
            # MONKAA
            count = 0
            dataset_location = os.path.join(data_root, 'monkaa')
            for dstype in ['frames_cleanpass_webp', 'frames_finalpass']:
                for cam in ['left', 'right']:
                    for direction in ['into_future', 'into_past']:
                        image_dirs = sorted(glob.glob(os.path.join(dataset_location, dstype, '*')))
                        flow_dirs = sorted(glob.glob(os.path.join(dataset_location, 'optical_flow/*')))
                        image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])
                        flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])
                        for idir, fdir in zip(image_dirs, flow_dirs):
                            if 'webp' in dstype:
                                images = sorted(glob.glob(os.path.join(idir, '*.webp')) )
                            else:
                                images = sorted(glob.glob(os.path.join(idir, '*.png')) )
                            flows = sorted(glob.glob(os.path.join(fdir, '*.pfm')) )
                            for i in range(len(flows)-1):
                                if direction == 'into_future':
                                    self.rgb0_paths.append(images[i])
                                    self.rgb1_paths.append(images[i+1])
                                    self.flow_paths.append(flows[i])
                                    self.valid_paths.append(None)
                                elif direction == 'into_past':
                                    self.rgb0_paths.append(images[i+1])
                                    self.rgb1_paths.append(images[i])
                                    self.flow_paths.append(flows[i+1])
                                    self.valid_paths.append(None)
                                self.dnames.append('monkaa')
                                count += 1
            print('added %d samples from monkaa (%s)' % (count, dset))
        
        if 'spring' in self.dataset_names:
            # SPRING (1080x1920)
            count = 0
            dataset_location = os.path.join(data_root, 'springdataset/spring')
            seq_root = os.path.join(dataset_location, 'train')
            data_list = []
            for scene in sorted(os.listdir(seq_root)):
                for cam in ["left"]: # we did not download "right" 
                    images = sorted(glob.glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                    # forward
                    for frame in range(1, len(images)):
                        data_list.append((frame, scene, cam, "FW"))
                    # # backward # we did not download "BW"
                    # for frame in reversed(range(2, len(images)+1)):
                    #     data_list.append((frame, scene, cam, "BW"))
            for frame_data in data_list:
                frame, scene, cam, direction = frame_data
                img1_path = os.path.join(seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
                if direction == "FW":
                    img2_path = os.path.join(seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame+1:04d}.png")
                else:
                    img2_path = os.path.join(seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame-1:04d}.png")
                flow_path = os.path.join(seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")

                for mult in range(10): # copy sea-raft
                    self.rgb0_paths.append(img1_path)
                    self.rgb1_paths.append(img2_path)
                    self.flow_paths.append(flow_path)
                    self.valid_paths.append(None)
                    self.dnames.append('spring')
                    count += 1
            print('added %d samples from spring (%s)' % (count, dset))
        
        if 'tartanair' in self.dataset_names:
            # TARTANAIR (480x640)
            count = 0
            dataset_location = os.path.join(data_root, 'tartanair')
            seqs = sorted(glob.glob(os.path.join(dataset_location, '*/*/P*')))
            image_dirs = sorted([os.path.join(f, 'image_left') for f in seqs])
            flow_dirs = sorted([os.path.join(f, 'flow') for f in seqs])
            for idir, fdir in zip(image_dirs, flow_dirs):
                images = sorted(glob.glob(os.path.join(idir, '*.png')) )
                flows = sorted(glob.glob(os.path.join(fdir, '*.npy')) )
                if len(flows)==len(images)-1:
                    for i in range(0,len(flows)-1,10): # redundant data so we subsample
                        self.rgb0_paths.append(images[i])
                        self.rgb1_paths.append(images[i+1])
                        self.flow_paths.append(flows[i])
                        self.valid_paths.append(None)
                        self.dnames.append('tartanair')
                        count += 1
            print('added %d samples from tartanair (%s)' % (count, dset))
        
        if 'things' in self.dataset_names:
            # THINGS (540x960)
            count = 0
            dataset_location = os.path.join(data_root, 'flyingthings')
            for dstype in ['frames_cleanpass_webp', 'frames_finalpass']:
                for cam in ['left']:
                    for direction in ['into_future', 'into_past']:
                        if is_training:
                            image_dirs = sorted(glob.glob(os.path.join(dataset_location, dstype, 'TRAIN/*/*')))
                            flow_dirs = sorted(glob.glob(os.path.join(dataset_location, 'optical_flow/TRAIN/*/*')))
                        else:
                            image_dirs = sorted(glob.glob(os.path.join(dataset_location, dstype, 'TEST/*/*')))
                            flow_dirs = sorted(glob.glob(os.path.join(dataset_location, 'optical_flow/TEST/*/*')))
                            
                        image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])
                        flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                        for idir, fdir in zip(image_dirs, flow_dirs):
                            if 'webp' in dstype:
                                images = sorted(glob.glob(os.path.join(idir, '*.webp')) )
                            else:
                                images = sorted(glob.glob(os.path.join(idir, '*.png')) )
                            flows = sorted(glob.glob(os.path.join(fdir, '*.pfm')) )
                            for i in range(len(flows)-1):
                                if direction == 'into_future':
                                    self.rgb0_paths.append(images[i])
                                    self.rgb1_paths.append(images[i+1])
                                    self.flow_paths.append(flows[i])
                                    self.valid_paths.append(None)
                                elif direction == 'into_past':
                                    self.rgb0_paths.append(images[i+1])
                                    self.rgb1_paths.append(images[i])
                                    self.flow_paths.append(flows[i+1])
                                    self.valid_paths.append(None)
                                self.dnames.append('things')
                                count += 1
            print('added %d samples from things (%s)' % (count, dset))
        
        if 'sintel' in self.dataset_names:      
            # SINTEL (436x1024)
            count = 0
            dataset_location = os.path.join(data_root, 'sintel')
            for v in ['clean', 'final']:
                rgb_root = os.path.join(dataset_location, 'training/%s' % v)
                flow_root = os.path.join(dataset_location, 'training/flow')
                valid_root = os.path.join(dataset_location, 'training/invalid')
                folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(rgb_root, "*"))]
                folder_names = sorted(folder_names)

                for ii, folder_name in enumerate(folder_names):
                    cur_rgb_path = os.path.join(rgb_root, folder_name)
                    cur_flow_path = os.path.join(flow_root, folder_name)
                    cur_valid_path = os.path.join(valid_root, folder_name)

                    img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
                    if is_training:
                        img_names = [im for im in img_names if not '000' in im]
                    else:
                        img_names = [im for im in img_names if '000' in im]
                    img_names = sorted(img_names)
                    S_here = len(img_names)

                    for si in range(0, S_here-1):
                        rgb0_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si])
                        rgb1_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si+1])
                        flow_path = os.path.join(cur_flow_path, '%s.flo' % img_names[si])
                        valid_path = os.path.join(cur_valid_path, '%s.png' % img_names[si])

                        for mult in range(20): # copy sea-raft
                            self.rgb0_paths.append(rgb0_path)
                            self.rgb1_paths.append(rgb1_path)
                            self.flow_paths.append(flow_path)
                            self.valid_paths.append(valid_path)
                            self.dnames.append('sintel')
                            count += 1
            print('added %d samples from sintel (%s)' % (count, dset))

        if 'viper' in self.dataset_names:
            # VIPER (1080x1920)
            count = 0
            dataset_location = os.path.join(data_root, 'viper')
            if is_training:
                rgb_root = os.path.join(dataset_location, 'train/img')
                flow_root = os.path.join(dataset_location, 'train/flow')
            else:
                rgb_root = os.path.join(dataset_location, 'val/img')
                flow_root = os.path.join(dataset_location, 'val/flow')
            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(rgb_root, "*"))]
            folder_names = sorted(folder_names)
            for ii, folder_name in enumerate(folder_names):
                cur_rgb_path = os.path.join(rgb_root, folder_name)
                cur_flow_path = os.path.join(flow_root, folder_name)
                flow_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_flow_path, "*"))]
                flow_names = sorted(flow_names)
                S_here = len(flow_names)
                for si in range(0, S_here):
                    flow_name = flow_names[si] # e.g., "001_00010"
                    seq = int(flow_name.split('_')[0])
                    num = int(flow_name.split('_')[1])
                    rgb0_path = os.path.join(cur_rgb_path, '%03d_%05d.jpg' % (seq, num))
                    rgb1_path = os.path.join(cur_rgb_path, '%03d_%05d.jpg' % (seq, num+1))
                    flow_path = os.path.join(cur_flow_path, '%03d_%05d.npz' % (seq, num))
                    if os.path.isfile(rgb0_path) and os.path.isfile(rgb1_path):
                        self.rgb0_paths.append(rgb0_path)
                        self.rgb1_paths.append(rgb1_path)
                        self.flow_paths.append(flow_path)
                        self.valid_paths.append(None)
                        self.dnames.append('viper')
                    count += 1
            print('added %d samples from viper (%s)' % (count, dset))

        # print(len(self.rgb0_paths), len(self.rgb1_paths), len(self.flow_paths), len(self.valid_paths), len(self.dnames))
        assert len(self.rgb0_paths) == len(self.rgb1_paths) == len(self.flow_paths) == len(self.valid_paths) == len(self.dnames)

        if shuffle:
            N = len(self.rgb0_paths)
            shuf = np.random.permutation(N)
            self.dnames = [self.dnames[ind] for ind in shuf]
            self.rgb0_paths = [self.rgb0_paths[ind] for ind in shuf]
            self.rgb1_paths = [self.rgb1_paths[ind] for ind in shuf]
            self.flow_paths = [self.flow_paths[ind] for ind in shuf]
            self.valid_paths = [self.valid_paths[ind] for ind in shuf]
        
        print('found %d samples total (%s); dataset_names' % (len(self.rgb0_paths), dset), self.dataset_names)

    def getitem_helper(self, index):
        cur_rgb0_path = self.rgb0_paths[index]
        cur_rgb1_path = self.rgb1_paths[index]
        cur_flow_path = self.flow_paths[index]
        cur_valid_path = self.valid_paths[index]
        dname = self.dnames[index]

        rgb0 = read_gen(cur_rgb0_path)
        rgb1 = read_gen(cur_rgb1_path)

        rgb0 = np.array(rgb0).astype(np.uint8)
        rgb1 = np.array(rgb1).astype(np.uint8)
        
        # grayscale images
        if len(rgb0.shape) == 2:
            rgb0 = np.tile(rgb0[...,None], (1, 1, 3))
            rgb1 = np.tile(rgb1[...,None], (1, 1, 3))
        else:
            rgb0 = rgb0[..., :3]
            rgb1 = rgb1[..., :3]

        valid = np.ones_like(rgb0[:,:,0]).astype(np.float32)
        
        if dname in ['driving', 'things', 'monkaa', 'spring']:
            flow = read_gen(cur_flow_path)
            if dname=='spring':
                # the flow is 2x the spatial resolution of the rgb
                flow = flow[::2,::2]

            flow = np.array(flow).astype(np.float32)[:,:,:2]
        elif dname in ['kitti', 'hd1k']:
            flow, valid = readFlowKITTI(cur_flow_path)
            flow = np.array(flow).astype(np.float32)[:,:,:2]
            if dname=='hd1k':
                flow_bak = flow.copy()
                flow = cv2.medianBlur(flow, 5)
                match = (np.linalg.norm(flow-flow_bak, axis=2)<1.0).astype(np.float32)
                valid = valid * match
        elif dname=='viper':
            di = dict(np.load(cur_flow_path, allow_pickle=True))
            u = di['u']
            v = di['v']
            u[np.isnan(u)] = 0
            v[np.isnan(v)] = 0
            u[np.isinf(u)] = 0
            v[np.isinf(v)] = 0
            flow = np.stack([u, v], axis=2).astype(np.float32) # H,W,2
            # the data has some weird outliers. we'll filter these away.
            flow_bak = flow.copy()
            flow = cv2.medianBlur(flow, 5)
            match = (np.linalg.norm(flow-flow_bak, axis=2)<1.0).astype(np.float32)
            valid = ((np.abs(flow[:,:,0])*np.abs(flow[:,:,1]))>0).astype(np.float32) * match
        elif dname=='tartanair':
            flow = np.load(cur_flow_path)
        elif dname=='sintel':
            flow = readFlow(cur_flow_path)
            flow = np.array(flow).astype(np.float32)
            valid = read_gen(cur_valid_path)
            valid = 1.0-np.array(valid).astype(np.float32)/255.0
        elif dname=='autoflow':
            flow = readFlow(cur_flow_path)
            flow = np.array(flow).astype(np.float32)
            flow_bak = flow.copy()
            flow = cv2.medianBlur(flow, 3)
            match = (np.linalg.norm(flow-flow_bak, axis=2)<1.0).astype(np.float32)
            valid = valid * match
        else:
            flow = readFlow(cur_flow_path)
            flow = np.array(flow).astype(np.float32)
            
        valid[np.isnan(np.sum(flow, axis=-1))] = 0
        flow[np.isnan(flow)] = 0

        if dname in ['viper','hd1k','spring']: # 1080p datasets (too big)
            sc = 0.5
            rgb0 = cv2.resize(rgb0, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            rgb1 = cv2.resize(rgb1, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            if dname in ['viper','spring']: # dense
                flow = cv2.resize(flow, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST) * sc
                valid = cv2.resize(valid, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            else: # sparse
                flow, valid = self.sparse_augmentor.resize_sparse_flow_map(flow, valid, sc, sc)

        if self.crop_size is not None:
            if self.use_augs:
                if dname=='kitti' or dname=='hd1k':
                    rgb0, rgb1, flow, valid = self.sparse_augmentor(rgb0, rgb1, flow, valid)
                else:
                    rgb0, rgb1, flow, valid = self.augmentor(rgb0, rgb1, flow, valid)
            else:
                print(dname, rgb0.shape)
                rgb0, rgb1, flow, valid = just_crop_flow(rgb0, rgb1, flow, valid, self.crop_size)

        # final isnan check before going to torch
        valid[np.isnan(np.sum(flow, axis=-1))] = 0
        flow[np.isnan(flow)] = 0
        
        rgb0 = torch.from_numpy(rgb0).permute(2,0,1) # 3,H,W
        rgb1 = torch.from_numpy(rgb1).permute(2,0,1) # 3,H,W
        flow = torch.from_numpy(flow).permute(2,0,1) # 2,H,W
        valid = torch.from_numpy(valid) # H,W
        valid = (valid==1.0).float().unsqueeze(0)

        flow_mag = torch.linalg.norm(flow, dim=0, keepdim=True)
        valid = (flow_mag < 400.0).float() * valid

        sample = {
            'dname': dname,
            'rgb0': rgb0,
            'rgb1': rgb1,
            'flow': flow,
            'valid': valid,
        }
        return sample

    def __getitem__(self, index):
        gotit = False
        fails = 0
        while not gotit and fails < 8:
            samp = self.getitem_helper(index)
            if torch.sum(samp['valid']):
                gotit = True
            else:
                fails += 1
                index = np.random.randint(len(self.rgb0_paths))
        if fails > 4:
            print('note: sampling failed %d times' % fails)
        return samp, True
    

    def __len__(self):
        return len(self.rgb0_paths)

