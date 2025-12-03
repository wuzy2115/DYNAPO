from numpy import random
import torch
import numpy as np
import os
import random
import imageio
from pathlib import Path
import matplotlib.pyplot as plt
from utils.basic import print_stats
from PIL import Image
import cv2
import utils.py
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.pointdataset import PointDataset

class ExportDataset(PointDataset):
    def __init__(self,
                 data_root='../datasets/alltrack_export',
                 version='bs',
                 dsets=None,
                 dsets_exclude=None,
                 seq_len=64, 
                 crop_size=(384,512),
                 shuffle_frames=False,
                 shuffle=True,
                 use_augs=False,
                 is_training=True,
                 backwards=False,
                 traj_per_sample=256, # min number of trajs
                 traj_max_factor=24, # multiplier on traj_per_sample
                 random_seq_len=False,
                 random_frame_rate=False,
                 random_number_traj=False,
                 only_first=False,
    ):
        super(ExportDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
        )
        print('loading export...')

        self.shuffle_frames = shuffle_frames
        self.pad_bounds = [10, 100]
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50
        self.only_first = only_first
        self.traj_max_factor = traj_max_factor

        self.S = seq_len

        self.use_augs = use_augs
        self.is_training = is_training

        self.dataset_location = Path(data_root) / version

        dataset_names = self.dataset_location.glob('*/')
        self.dataset_names = [str(fn.stem) for fn in dataset_names]
        self.dataset_names = ['%s%d' % (dname, self.S) for dname in self.dataset_names]
        print('dataset_names', self.dataset_names)

        folder_names = self.dataset_location.glob('*/*/*/')
        folder_names = [str(fn) for fn in folder_names]
        # print('folder_names', folder_names)

        print('found {:d} {} folders in {}'.format(len(folder_names), version, self.dataset_location))

        if dsets is not None:
            print('dsets', dsets)
            new_folder_names = []
            for fn in folder_names:
                for dset in dsets:
                    if dset in fn:
                        new_folder_names.append(fn)
                        break
            folder_names = new_folder_names
            print('filtered to %d folders' % len(folder_names))

        if backwards:
            new_folder_names = []
            for fn in folder_names:
                chunk = fn.split('/')[-2]
                if 'b' in chunk:
                    new_folder_names.append(fn)
            folder_names = new_folder_names
            print('filtered to %d folders with backward motion' % len(folder_names))
            
        if dsets_exclude is not None:
            print('dsets_exclude', dsets_exclude)
            new_folder_names = []
            for fn in folder_names:
                keep = True
                for dset in dsets_exclude:
                    if dset in fn:
                        keep = False
                        break
                if keep:
                    new_folder_names.append(fn)
            folder_names = new_folder_names
            print('filtered to %d folders' % len(folder_names))
            
        # if quick:
        #     folder_names = sorted(folder_names)
        #     folder_names = folder_names[:201]
        #     print('folder_names', folder_names)

        if shuffle:
            random.shuffle(folder_names)
        else:
            folder_names = sorted(list(folder_names))

        self.all_folders = folder_names
        # # step through once and make sure all of the npys are there
        # print('stepping through...')
        # self.all_folders = []
        # for fi, fol in enumerate(folder_names):
        #     npy_path = os.path.join(fol, "annot.npy")
        #     rgb_path = os.path.join(fol, "frames")
        #     if os.path.isdir(rgb_path) and os.path.isfile(npy_path):
        #         img_paths = sorted(os.listdir(rgb_path))
        #         if len(img_paths)>=self.S:
        #             self.all_folders.append(fol)
        #         else:
        #             pass
        #     else:
        #         pass
        # print('ok done stepping; got %d' % len(self.all_folders))
        
        
    def getitem_helper(self, index):
        # cH, cW = self.cH, self.cW
        
        fol = self.all_folders[index]
        npy_path = os.path.join(fol, "annot.npy")
        rgb_path = os.path.join(fol, "frames")

        mid = str(fol)[len(str(self.dataset_location))+1:]
        dname = mid.split('/')[0]
        # print('dname', dname)

        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []
        try:
            for i, img_path in enumerate(img_paths):
                rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))
        except:
            print('some exception when reading rgbs')

        if len(rgbs)<self.S:
            print('some problem with fol', fol)
            return None, False

        rgbs = np.stack(rgbs, axis=0).astype(np.float32) # S,H,W,3
        S,H,W,C = rgbs.shape
        # rgbs = rgbs.transpose(0,3,1,2) # S,3,H,W
        # S,C,H,W = rgbs.shape
        assert(C==3)
        
        try:
            d = np.load(npy_path, allow_pickle=True).item()
        except:
            print('some problem with npy', npy_path)
            return None, False

        trajs = d['point_xys'].astype(np.float32) # S,N,2
        visibs = d['point_visibs'].astype(np.float32) # S,N
        # print('trajs', trajs.shape)
        # print('visibs', visibs.shape)

        if self.use_augs and np.random.rand() < 0.5:
            # time flip
            rgbs = np.flip(rgbs, axis=[0]).copy()
            trajs = np.flip(trajs, axis=[0]).copy()
            visibs = np.flip(visibs, axis=[0]).copy()

        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]

        if trajs.shape[0] > self.S:
            surplus = trajs.shape[0] - self.S
            ind = np.random.randint(surplus)+1
            rgbs = rgbs[ind:ind+self.S]
            trajs = trajs[ind:ind+self.S]
            visibs = visibs[ind:ind+self.S]
        assert(trajs.shape[0] == self.S)

        if self.only_first:
            vis_ok = np.nonzero(visibs[0]==1)[0]
            trajs = trajs[:,vis_ok]
            visibs = visibs[:,vis_ok]

        N = trajs.shape[1]
        if N < self.traj_per_sample:
            print('exp: %s; N after vis0: %d' % (dname, N))
            return None, False

        # print('rgbs', rgbs.shape)
        if H > self.crop_size[0]*2 and W > self.crop_size[1]*2 and np.random.rand() < 0.5:
            scale = 0.5
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            H, W = rgbs[0].shape[:2]
            rgbs = np.stack(rgbs, axis=0) # S,H,W,3
            trajs = trajs * scale
            # print('resized rgbs', rgbs.shape)
            
        if H > self.crop_size[0]*2 and W > self.crop_size[1]*2 and np.random.rand() < 0.5:
            scale = 0.5
            rgbs = [cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            H, W = rgbs[0].shape[:2]
            rgbs = np.stack(rgbs, axis=0) # S,H,W,3
            trajs = trajs * scale
            # print('resized rgbs', rgbs.shape)

        if self.use_augs and np.random.rand() < 0.98:
            rgbs, trajs, visibs = self.add_photometric_augs(
                rgbs, trajs, visibs, replace=False,
            )
            if np.random.rand() < 0.2:
                rgbs, trajs = self.add_spatial_augs(rgbs, trajs, visibs, self.crop_size)
            else:
                rgbs, trajs = self.follow_crop(rgbs, trajs, visibs, self.crop_size)
            if np.random.rand() < self.rot_prob:
                # note this is OK since B==1
                # otw we would do it before this func
                rgbs = [np.transpose(rgb, (1,0,2)).copy() for rgb in rgbs]
                rgbs = np.stack(rgbs)
                trajs = np.flip(trajs, axis=2).copy()
            H, W = rgbs[0].shape[:2]
            if np.random.rand() < self.h_flip_prob:
                rgbs = [rgb[:, ::-1].copy() for rgb in rgbs]
                trajs[:, :, 0] = W - trajs[:, :, 0]
                rgbs = np.stack(rgbs)
            if np.random.rand() < self.v_flip_prob:
                rgbs = [rgb[::-1].copy() for rgb in rgbs]
                trajs[:, :, 1] = H - trajs[:, :, 1]
                rgbs = np.stack(rgbs)
        else:
            rgbs, trajs = self.crop(rgbs, trajs, self.crop_size)

        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames (again)
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]

        H,W = rgbs[0].shape[:2]

        visibs[trajs[:, :, 0] > W-1] = False
        visibs[trajs[:, :, 0] < 0] = False
        visibs[trajs[:, :, 1] > H-1] = False
        visibs[trajs[:, :, 1] < 0] = False

        N = trajs.shape[1]
        # print('N8', N)

        # ensure no crazy values
        all_valid = np.nonzero(np.sum(np.sum(np.abs(trajs), axis=-1)<100000, axis=0)==self.S)[0]
        trajs = trajs[:,all_valid]
        visibs = visibs[:,all_valid]
        
        if self.only_first:
            vis_ok = np.nonzero(visibs[0]==1)[0]
            trajs = trajs[:,vis_ok]
            visibs = visibs[:,vis_ok]
            N = trajs.shape[1]
            
        if N < self.traj_per_sample:
            print('exp: %s; N after aug: %d' % (dname, N))
            return None, False

        N = trajs.shape[1]
        
        seq_len = S
        visibs = torch.from_numpy(visibs)
        trajs = torch.from_numpy(trajs)

        # discard tracks that go far OOB
        crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1) < max(H,W)*2,
            dim=0,
        )
        trajs = trajs[:, close_pts_inds]
        visibs = visibs[:, close_pts_inds]
        
        visible_pts_inds = (visibs[0]).nonzero(as_tuple=False)[:, 0]
        point_inds = torch.randperm(len(visible_pts_inds))[:self.traj_per_sample*self.traj_max_factor]
        if len(point_inds) < self.traj_per_sample:
            # print('not enough trajs')
            # gotit = False
            return None, False

        visible_inds_sampled = visible_pts_inds[point_inds]

        trajs = trajs[:, visible_inds_sampled].float()
        visibs = visibs[:, visible_inds_sampled].float()
        valids = torch.ones_like(visibs).float()

        trajs = trajs[:, :self.traj_per_sample*self.traj_max_factor]
        visibs = visibs[:, :self.traj_per_sample*self.traj_max_factor]
        valids = valids[:, :self.traj_per_sample*self.traj_max_factor]

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()

        dname += '%d' % (self.S)
        
        sample = utils.data.VideoData(
            video=rgbs,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            seq_name=None,
            dname=dname,
        )
        return sample, True
        
    def __len__(self):
        return len(self.all_folders)
