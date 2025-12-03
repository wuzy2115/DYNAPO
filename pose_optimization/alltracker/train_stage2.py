import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from lightning_fabric import Fabric 
import utils.loss
import utils.samp
import utils.data
import utils.improc
import utils.misc
import utils.saveload
from tensorboardX import SummaryWriter
import datetime
import time
from nets.blocks import bilinear_sampler

torch.set_float32_matmul_precision('medium')

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

def get_flow_dataset(args):
    import datasets.metaflowdataset
    dataset = datasets.metaflowdataset.MetaflowDataset(
        data_root=args.data_dir,
        crop_size=args.crop_size_2,
        use_augs=args.use_augs,
        shuffle=True,
        is_training=True,
    )
    dataset_names = dataset.dataset_names
    return dataset, dataset_names

def get_multi_dataset_24(args, crop_size, N, version='aa00', T=56):
    from datasets import exportdataset
    dsets = None
    dsets_exclude = None
    export_dataset = exportdataset.ExportDataset(
        data_root=os.path.join(args.data_dir),
        version=version,
        dsets=dsets,
        dsets_exclude=dsets_exclude,
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        random_seq_len=args.random_seq_len,
        random_frame_rate=args.random_frame_rate,
        random_number_traj=args.random_number_traj,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )

    # sample clips from the S64 pod & kub exports 
    export_dataset2 = exportdataset.ExportDataset(
        data_root=os.path.join(args.data_dir),
        version='ce64',
        dsets=['podlong','kublong'],
        dsets_exclude=dsets_exclude,
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        random_seq_len=args.random_seq_len,
        random_frame_rate=args.random_frame_rate,
        random_number_traj=args.random_number_traj,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )
    
    from datasets import dynrep_dataset
    dyn_dataset = dynrep_dataset.DynamicReplicaDataset(
        os.path.join(args.data_dir, 'dynamicreplica'), 
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )

    # stage1 kubric data
    from datasets import kubric_movif_dataset
    kubric_version = 'kubric_au'
    kub_dataset = kubric_movif_dataset.KubricMovifDataset(
        data_root=os.path.join(args.data_dir, kubric_version),
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        random_seq_len=args.random_seq_len,
        random_frame_rate=args.random_frame_rate,
        random_number_traj=args.random_number_traj,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )

    from torch.utils.data import ConcatDataset

    dataset = ConcatDataset([export_dataset] + [export_dataset2] + [dyn_dataset] + [kub_dataset])
    dataset_names = export_dataset.dataset_names + export_dataset2.dataset_names + [dyn_dataset.dname] + [kub_dataset.dname]
    
    return dataset, dataset_names


def get_multi_dataset_64(args, crop_size, N, version='aa00', T=56):
    from datasets import exportdataset
    dsets = None
    dsets_exclude = None
    export_dataset = exportdataset.ExportDataset(
        data_root=os.path.join(args.data_dir),
        version=version,
        dsets=dsets,
        dsets_exclude=dsets_exclude,
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        random_seq_len=args.random_seq_len,
        random_frame_rate=args.random_frame_rate,
        random_number_traj=args.random_number_traj,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )

    # stage1 kubric data (even though it's included in export_dataset)
    from datasets import kubric_movif_dataset
    kubric_version = 'ce64/kublong'
    kub_dataset = kubric_movif_dataset.KubricMovifDataset(
        data_root=os.path.join(args.data_dir, kubric_version),
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        random_seq_len=args.random_seq_len,
        random_frame_rate=args.random_frame_rate,
        random_number_traj=args.random_number_traj,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )
    
    from datasets import dynrep_dataset
    dyn_dataset = dynrep_dataset.DynamicReplicaDataset(
        os.path.join(args.data_dir, 'dynamicreplica'), 
        crop_size=crop_size,
        seq_len=T,
        traj_per_sample=N,
        use_augs=args.use_augs,
        shuffle_frames=args.shuffle_frames,
        shuffle=True,
        only_first=True,
    )
    
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset([export_dataset] + [kub_dataset] + [dyn_dataset])
    dataset_names = export_dataset.dataset_names + [dyn_dataset.dname]
    
    return dataset, dataset_names


def create_pools(n_pool=50, min_size=10):
    pools = {}

    n_pool = max(n_pool, 10)

    thrs = [1,2,4,8,16]
    for thr in thrs:
        pools['d_%d' % thr] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['d_avg'] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    
    pool_names = [
        'seq_loss_visible',
        'seq_loss_invisible',
        'vis_loss',
        'conf_loss',
        'total_loss',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
        
    return pools

def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if not ("bias" in name)]
    nondecay_parameters = [n for n, p in model.named_parameters() if n not in decay_parameters]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "lr": args.lr,
            "weight_decay": args.wdecay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n in nondecay_parameters],
            "lr": args.lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr, weight_decay=args.wdecay)

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.max_steps+100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
    else:
        scheduler = None
        
    return optimizer, scheduler

def forward_batch_sparse(batch, model, args, sw, inference_iters):
    rgbs = batch.video
    trajs_g = batch.trajs
    vis_g = batch.visibs # B,S,N
    valids = batch.valids
    dname = batch.dname
    
    B, T, C, H, W = rgbs.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = rgbs.device
    S = 16
    
    if N==0 or torch.any(vis_g[:,0].reshape(-1)==0):
        print('vis_g', vis_g.shape)
        return None, None

    __, first_positive_inds = torch.max(vis_g, dim=1)
    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis_g[ind_array_ == first_positive_inds[:, None, :]],
        torch.ones(1, device=device),
    )
    gather = torch.gather(
        trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D)
    )
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)
    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)

    assert B == 1

    if np.any(queries.shape==0):
        return None, None

    try: 
        if (torch.isnan(queries).any()
            or torch.isnan(trajs_g).any()
            or queries.abs().reshape(-1).max() > 1500):
            print("failed_sample")
            print("queries time", queries[..., 0])
            print("queries ", queries[..., 1:])
            queries = torch.ones_like(queries).to(queries.device).float()
            print("new queries", queries)
            valids = torch.zeros_like(valids).to(valids.device).float()
            print("new valids", valids)
    except:
        print('some big problem')
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        print("queries.shape", queries.shape)
        print("trajs_g.shape", trajs_g.shape)
    

    if (torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().reshape(-1).max() > 1500):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        valids = torch.zeros_like(valids).to(valids.device).float()
        print("new valids", valids)

    stride = 8
    inds = list(range(0, T - stride, stride))
    if args.rand_stride and np.random.rand() < 0.1:
        len_normal = len(inds)
        stride = np.random.randint(1,S//2)
        inds = []
        start = 0
        while start + S < T:
            inds.append(start)
            start += stride
        inds = inds[:len_normal]
        T = inds[-1]+S
        rgbs = rgbs[:,:T]
        trajs_g = trajs_g[:,:T]
        vis_g = vis_g[:,:T]
        valids = valids[:,:T]
        
    all_flow_e, all_visconf_e, all_flow_preds, all_visconf_preds = model(
        rgbs, iters=inference_iters, sw=sw, is_training=True,
        stride=stride)

    if torch.isnan(all_flow_e).any():
        print('nan!')
        return None, None
        
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device=device).float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W
    traj_maps_e = all_flow_e + grid_xy # B,T,2,H,W
    xy0 = trajs_g[:,0] # B,N,2
    xy0[:,:,0] = xy0[:,:,0].clamp(0,W-1)
    xy0[:,:,1] = xy0[:,:,1].clamp(0,H-1)

    traj_maps_e_ = traj_maps_e.reshape(B*T,2,H,W)
    xy0_ = xy0.reshape(B,1,N,2).repeat(1,T,1,1).reshape(B*T,N,1,2)
    trajs_e_ = utils.samp.bilinear_sampler(traj_maps_e_, xy0_) # B*T,2,N,1
    trajs_e = trajs_e_.reshape(B,T,2,N).permute(0,1,3,2) # B,T,N,2

    xy0_ = xy0.reshape(B,1,N,2).repeat(1,S,1,1).reshape(B*S,N,1,2)

    coord_predictions = []
    assert(B==1)
    for fpl in all_flow_preds:
        cps = []
        for fp in fpl:
            traj_map = fp + grid_xy # B,S,2,H,W
            traj_map_ = traj_map.reshape(B*S,2,H,W)
            traj_e_ = utils.samp.bilinear_sampler(traj_map_, xy0_) # B*T,2,N,1
            traj_e = traj_e_.reshape(B,S,2,N).permute(0,1,3,2) # B,S,N,2
            cps.append(traj_e)
        coord_predictions.append(cps)
    
    # visconf is upset when i bilinearly sample. says the data is outside [0,1]
    # so here we do NN sampling
    assert(B==1)
    x0, y0 = xy0[0,:,0], xy0[0,:,1] # N
    x0 = torch.clamp(x0, 0, W-1).round().long()
    y0 = torch.clamp(y0, 0, H-1).round().long()
    vis_predictions, confidence_predictions = [], []
    for vcl in all_visconf_preds:
        vps = []
        cps = []
        for vc in vcl:
            vc = vc[:,:,:,y0,x0] # B,S,2,N
            vps.append(vc[:,:,0]) # B,S,N
            cps.append(vc[:,:,1]) # B,S,N
        vis_predictions.append(vps)
        confidence_predictions.append(cps)
        
    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    for ind in inds:
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        valids_gts.append(valids[:, ind : ind + S])

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    metrics = {}
          
    seq_loss_visible = utils.loss.sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=0.8,
        use_huber_loss=args.use_huber_loss,
        loss_only_for_visible=True,
    )
    confidence_loss = utils.loss.sequence_prob_loss(
        coord_predictions, confidence_predictions, traj_gts, vis_gts
    )
    vis_loss = utils.loss.sequence_BCE_loss(vis_predictions, vis_gts)

    seq_loss_invisible = utils.loss.sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=invis_gts,
        gamma=0.8,
        use_huber_loss=args.use_huber_loss,
        loss_only_for_visible=True,
    )
    
    total_loss = seq_loss_visible.mean()*0.05 + seq_loss_invisible.mean()*0.02 + vis_loss.mean() + confidence_loss.mean()

    if sw is not None and sw.save_scalar:
        assert B == 1
        dn = dname[0]
        metrics['dname'] = dn
        metrics['seq_loss_visible'] = seq_loss_visible.mean().item()
        metrics['seq_loss_invisible'] = seq_loss_invisible.mean().item()
        metrics['vis_loss'] = vis_loss.item()
        metrics['conf_loss'] = confidence_loss.item()
        thrs = [1,2,4,8,16]
        sx_ = (W-1) / 255.0
        sy_ = (H-1) / 255.0
        sc_py = np.array([sx_, sy_]).reshape([1,1,1,2])
        sc_pt = torch.from_numpy(sc_py).float().to(device)
        d_sum = 0.0
        for thr in thrs:
            d_ = (torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) < thr).float().mean().item()
            d_sum += d_
            metrics['d_%d' % thr] = d_
        d_avg = d_sum / len(thrs)
        metrics['d_avg'] = d_avg
        metrics['total_loss'] = total_loss.item()
    
    if sw is not None and sw.save_this:
        utils.basic.print_stats('rgbs', rgbs)
        prep_rgbs = utils.basic.normalize(rgbs[0:1])-0.5
        prep_grays = prep_rgbs.mean(dim=2, keepdim=True).repeat(1,1,3,1,1)
        sw.summ_rgb('0_inputs/rgb0', prep_rgbs[:,0], frame_str=dname[0], frame_id=torch.sum(vis_g[0,0]).item())
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb0', trajs_g[0:1], prep_rgbs[:,0], cmap='winter', linewidth=1)
        trajs_clamp = trajs_g.clone()
        trajs_clamp[:,:,:,0] = trajs_clamp[:,:,:,0].clip(0,W-1)
        trajs_clamp[:,:,:,1] = trajs_clamp[:,:,:,1].clip(0,H-1)
        inds = np.random.choice(trajs_g.shape[2], 1024)
        outs = sw.summ_pts_on_rgbs(
            '',
            trajs_clamp[0:1,:,inds],
            prep_grays[0:1],
            valids=valids[0:1,:,inds],
            cmap='winter', linewidth=3, only_return=True)
        sw.summ_pts_on_rgbs(
            '0_inputs/kps_gv_on_rgbs',
            trajs_clamp[0:1,:,inds],
            utils.improc.preprocess_color(outs),
            valids=valids[0:1,:,inds]*vis_g[0:1,:,inds],
            cmap='spring', linewidth=2,
            frame_ids=list(range(T)))
        
        out = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1,:,inds], prep_rgbs[:,0], cmap='winter', linewidth=1, only_return=True))
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_g', trajs_e[0:1,:,inds], out, cmap='spring', linewidth=1)

        trajs_e_clamp = trajs_e.clone()
        trajs_e_clamp[:,:,:,0] = trajs_e_clamp[:,:,:,0].clip(0,W-1)
        trajs_e_clamp[:,:,:,1] = trajs_e_clamp[:,:,:,1].clip(0,H-1)
        inds_e = np.random.choice(trajs_e.shape[2], 1024)
        outs = sw.summ_pts_on_rgbs(
            '',
            trajs_clamp[0:1,:,inds],
            prep_grays[0:1],
            valids=valids[0:1,:,inds]*vis_g[0:1,:,inds],
            cmap='winter', linewidth=2,
            only_return=True)
        sw.summ_pts_on_rgbs(
            '2_outputs/kps_ge_on_rgbs',
            trajs_e_clamp[0:1,:,inds],
            utils.improc.preprocess_color(outs),
            valids=valids[0:1,:,inds]*vis_g[0:1,:,inds],
            cmap='spring', linewidth=2,
            frame_ids=list(range(T)))
                                                                        
    return total_loss, metrics

def forward_batch_flow(batch, model, args, sw, inference_iters, horz_flip=False, vert_flip=False):

    dname = batch['dname'] # B
    rgb0 = batch['rgb0'].float().cuda() # B,3,H,W
    rgb1 = batch['rgb1'].float().cuda() # B,3,H,W
    flow_g = batch['flow'].float().cuda() # B,2,H,W
    flow_valid = batch['valid'].float().cuda() # B,1,H,W
    device = rgb0.device

    B,C,H,W = rgb0.shape
    assert(C==3)

    def flip_and_concat(tensor, dim):
        return torch.cat([tensor, torch.flip(tensor, [dim])], dim=0)

    if horz_flip:
        rgb0 = flip_and_concat(rgb0, dim=-1)
        rgb1 = flip_and_concat(rgb1, dim=-1)
        flow_g = flip_and_concat(flow_g, dim=-1)
        flow_valid = flip_and_concat(flow_valid, dim=-1)
        flow_g[B:,0] = -flow_g[B:,0] # switch the x direction
        dname = dname * 2
        B = B*2
        
    if vert_flip:
        rgb0 = flip_and_concat(rgb0, dim=-2)
        rgb1 = flip_and_concat(rgb1, dim=-2)
        flow_g = flip_and_concat(flow_g, dim=-2)
        flow_valid = flip_and_concat(flow_valid, dim=-2)
        flow_g[B:,1] = -flow_g[B:,1] # switch the y direction
        dname = dname * 2
        B = B*2

    if horz_flip or vert_flip:
        # shuffle so that the vis (on zeroth ind) shows a random one
        perm = np.random.permutation(B)
        rgb0 = rgb0[perm]
        rgb1 = rgb1[perm]
        flow_g = flow_g[perm]
        flow_valid = flow_valid[perm]
        dname = [dname[pi] for pi in perm]
            
    flow_g_mag = torch.linalg.norm(flow_g, dim=1, keepdim=True)

    if sw is not None and sw.save_this:
        prep_rgb0 = utils.basic.normalize(rgb0[0:1])-0.5
        prep_rgb1 = utils.basic.normalize(rgb1[0:1])-0.5
        
        rgb0_vis = sw.summ_rgb('', prep_rgb0, only_return=True, frame_str=dname[0])
        rgb1_vis = sw.summ_rgb('', prep_rgb1, only_return=True)
        flow_vis = sw.summ_flow('', flow_g[0:1], only_return=True, frame_id=flow_g_mag[0].mean().item())
        flow_mask = sw.summ_oned('', flow_valid[0:1], norm=False, only_return=True, frame_id=flow_valid[0].mean().item())
        cat = torch.cat([rgb0_vis, rgb1_vis, flow_vis, flow_mask], axis=-1)
        sw.summ_rgb('0_inputs/rgbs_and_flow', cat)

    all_rgbs = torch.stack([rgb0, rgb1], axis=1)
    flow_e, visconf_e, flow_preds, visconf_preds = model(
        all_rgbs, iters=inference_iters, sw=sw,
        is_training=True)

    if torch.isnan(flow_e).any():
        print('nan!')
        return None, None
    
    _,_,fH,fW = flow_e.shape

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    for ii, flow_e in enumerate(flow_preds):
        flow_e_ = flow_e.permute(0,2,3,1).reshape(B*fH*fW,2)
        flow_g_ = flow_g.permute(0,2,3,1).reshape(B*fH*fW,2)
        flow_valid_ = flow_valid.reshape(B*fH*fW)
        if torch.sum(flow_valid_):
            flow_e_ = flow_e_[flow_valid_>0]
            flow_g_ = flow_g_[flow_valid_>0]
            flow_l1 = torch.sum(torch.abs(flow_e_-flow_g_), dim=1) # -1
            flow_l1 = flow_l1.mean()
        else:
            flow_l1 = torch.tensor(0.0, requires_grad=True, device=device)
        gamma = 0.8
        i_weight = gamma**(inference_iters - ii - 1)
        total_loss = total_loss + flow_l1*i_weight * 0.01

    flow_l1 = torch.sum(torch.abs(flow_e - flow_g), dim=1) # B,fH,fW
    flow_l2 = torch.linalg.vector_norm(flow_e - flow_g, dim=1) # B,fH,fW

    if sw is not None and sw.save_this:
        clip = max(torch.abs(flow_g[0]).max().item(), torch.abs(flow_e[0]).max().item())
        clip = max(clip, 2.0)
        vis = []
        for ii, flow_e in enumerate(flow_preds):
            flow_l2_ = torch.linalg.vector_norm(flow_e[0:1] - flow_g[0:1], dim=1).mean().item()
            flow_g_vis = sw.summ_flow('', flow_g[0:1], clip=clip, only_return=True)
            flow_e_vis = sw.summ_flow('', flow_e[0:1], clip=clip, only_return=True, frame_id=flow_l2_)
            vis.append(torch.cat([flow_g_vis, flow_e_vis], axis=-1))
        sw.summ_rgbs('1_model/flow_ge_anim', vis)
        sw.summ_rgb('1_model/flow_ge', vis[-1])

    metrics = [{} for _ in range(B)]
    if sw is not None and sw.save_scalar:
        for b in range(B):
            dn = dname[b]
            metrics[b]['dname'] = dn
            flow_l1_ = flow_l1[b].reshape(fH*fW)
            flow_l2_ = flow_l2[b].reshape(fH*fW)
            flow_err_valid = flow_valid[b].reshape(fH*fW)
            if torch.sum(flow_err_valid):
                flow_l1_ = flow_l1_[flow_err_valid>0]
                flow_l2_ = flow_l2_[flow_err_valid>0]
                metrics[b]['flow_l1'] = flow_l1_.mean().item()
                metrics[b]['flow_l2'] = flow_l2_.mean().item()

                thrs = [1,2,4,8,16]
                sx_ = W / 256.0
                sy_ = H / 256.0
                sc_py = np.array([sx_, sy_]).reshape([1,2])
                sc = torch.from_numpy(sc_py).float().to(device)
                flow_e_ = flow_e[b].reshape(2,-1).permute(1,0)[flow_err_valid>0]
                flow_g_ = flow_g[b].reshape(2,-1).permute(1,0)[flow_err_valid>0]
                d_sum = 0.0
                for thr in thrs:
                    d_ = (torch.norm(flow_e_/sc - flow_g_/sc, dim=-1) < thr).float().mean().item()
                    d_sum += d_
                    metrics[b]['d_%d' % thr] = d_
                d_avg = d_sum / len(thrs)
                metrics[b]['d_avg'] = d_avg
                
            metrics[b]['total_loss'] = total_loss.item() # [b].item()
        
    return total_loss, metrics


def run(model, args):
    fabric = Fabric(
        devices="auto",
        num_nodes=1,
        strategy="ddp",
        accelerator="cuda",
        precision="bf16-mixed" if args.mixed_precision else "32-true",
    )
    fabric.launch() # enable multi-gpu
    
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    B2_ = args.batch_size_2 * torch.cuda.device_count()
    if args.horz_flip_2:
        B2_ *= 2
    if args.vert_flip_2:
        B2_ *= 2
    model_name = "%d" % (B2_)
    if args.horz_flip_2:
        model_name += "h"
    if args.vert_flip_2:
        model_name += "v"
    B24_ = args.batch_size_24 * torch.cuda.device_count()
    model_name += "%d" % (B24_)
    if args.use_augs:
        model_name += "A"
    model_name += "i%d" % (args.inference_iters_2)
    model_name += "i%d" % (args.inference_iters_24)
    model_name += "i%d" % (args.inference_iters_56)
    if args.shuffle_frames:
        model_name += "s"
    lrn = utils.basic.get_lr_str(args.lr) # e.g., 5e-4
    model_name += "_%s" % lrn
    if args.mixed_precision:
        model_name += "m"
    if args.use_huber_loss:
        model_name += "h"
    model_name += "_%s" % args.exp
    model_date = datetime.datetime.now().strftime('%M%S')
    model_name = model_name + '_' + model_date
    model_name += '%d' % fabric.global_rank
    print('model_name', model_name)

    save_dir = '%s/%s' % (args.ckpt_dir, model_name)

    model.cuda()

    dataset_names = []

    # logic to distribute different-shaped samples across gpus
    if fabric.world_size==2:
        ranks_2 = []
        ranks_24 = [0]
        ranks_56 = [1]
        log_ranks = [0,1]
    elif fabric.world_size==4:
        ranks_2 = [0]
        ranks_24 = [1,2]
        ranks_56 = [3,4]
        log_ranks = [0,1,3]
    elif fabric.world_size==8:
        ranks_2 = [0]
        ranks_24 = [1,2,3]
        ranks_56 = [4,5,6,7]
        log_ranks = [0,2,5]
    else:
        ranks_2 = []
        ranks_24 = []
        ranks_56 = [0]
        log_ranks = [0]
        print('assuming we are debugging with 1 gpu...')
    
    if fabric.global_rank in ranks_2:
        flow_dataset, flow_dataset_names = get_flow_dataset(args)
        flow_loader = torch.utils.data.DataLoader(
            flow_dataset,
            batch_size=args.batch_size_2,
            shuffle=True,
            num_workers=args.num_workers_2,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=False,
            drop_last=True,
        )
        flow_loader = fabric.setup_dataloaders(flow_loader, move_to_device=False, use_distributed_sampler=False)
        print('len(flow_loader)', len(flow_loader))
        flow_iterloader = iter(flow_loader)
        dataset_names += flow_dataset_names
    elif fabric.global_rank in ranks_24:
        sparse_dataset24, sparse_dataset_names24 = get_multi_dataset_24(
            args, crop_size=args.crop_size_24, T=24, N=args.traj_per_sample_24, version='ce24')
        sparse_loader24 = torch.utils.data.DataLoader(
            sparse_dataset24,
            batch_size=args.batch_size_24,
            shuffle=True,
            num_workers=args.num_workers_24,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=False,
            collate_fn=utils.data.collate_fn_train,
            drop_last=True,
        )
        sparse_loader24 = fabric.setup_dataloaders(sparse_loader24, move_to_device=False, use_distributed_sampler=False)
        print('len(sparse_loader24)', len(sparse_loader24))
        sparse_iterloader24 = iter(sparse_loader24)
        dataset_names += sparse_dataset_names24
    elif fabric.global_rank in ranks_56:
        sparse_dataset56, sparse_dataset_names56 = get_multi_dataset_64(
            args, crop_size=args.crop_size_56, T=56, N=args.traj_per_sample_56, version='ce64')
        sparse_loader56 = torch.utils.data.DataLoader(
            sparse_dataset56,
            batch_size=args.batch_size_56,
            shuffle=True,
            num_workers=args.num_workers_56,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=False,
            collate_fn=utils.data.collate_fn_train,
            drop_last=True,
        )
        sparse_loader56 = fabric.setup_dataloaders(sparse_loader56, move_to_device=False, use_distributed_sampler=False)
        print('len(sparse_loader56)', len(sparse_loader56))
        sparse_iterloader56 = iter(sparse_loader56)
        dataset_names += sparse_dataset_names56

    dataset_names = list(set(dataset_names))
    
    optimizer, scheduler = fetch_optimizer(args, model)

    if fabric.global_rank in log_ranks:
        log_dir = './logs_train'
        pools_t = {}
        for dname in dataset_names:
            if not (dname in pools_t):
                print('creating pools for', dname)
                pools_t[dname] = create_pools()
        overpools_t = create_pools()
        writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
        
    global_step = 0
    if args.init_dir:
        load_dir = '%s/%s' % (args.ckpt_dir, args.init_dir)
        loaded_global_step = utils.saveload.load(
            fabric,
            load_dir,
            model,
            optimizer=optimizer if args.load_optimizer else None,
            scheduler=scheduler if args.load_scheduler else None,
            ignore_load=None,
            strict=True)
        if args.load_optimizer and not args.use_scheduler:
            assert(optimizer.param_groups[-1]["lr"] == args.lr)
        if args.load_step:
            global_step = loaded_global_step
        if args.use_scheduler and args.load_step and (not args.load_optimizer):
            # advance the scheduler to catch up with global_step
            for ii in range(global_step):
                scheduler.step()

    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)
    model.train()

    while global_step < args.max_steps+10:
        global_step += 1

        f_start_time = time.time()

        optimizer.zero_grad(set_to_none=True)
        assert model.training

        if fabric.global_rank in log_ranks:
            sw_t = utils.improc.Summ_writer(
                writer=writer_t,
                global_step=global_step,
                log_freq=args.log_freq,
                fps=8,
                scalar_freq=min(53,args.log_freq),
                just_gif=True)
        else:
            sw_t = None

        metrics = None
            
        gotit = [False, False]
        while not all(gotit):
            if fabric.global_rank in ranks_2:
                try:
                    batch = next(flow_iterloader)
                except StopIteration:
                    flow_iterloader = iter(flow_loader)
                    batch = next(flow_iterloader)
            elif fabric.global_rank in ranks_24:
                try:
                    batch = next(sparse_iterloader24)
                except StopIteration:
                    sparse_iterloader24 = iter(sparse_loader24)
                    batch = next(sparse_iterloader24)
            elif fabric.global_rank in ranks_56:
                try:
                    batch = next(sparse_iterloader56)
                except StopIteration:
                    sparse_iterloader56 = iter(sparse_loader56)
                    batch = next(sparse_iterloader56)
            batch, gotit = batch
        
        rtime = time.time()-f_start_time

        if fabric.global_rank in ranks_2:
            inference_iters = args.inference_iters_2
        elif fabric.global_rank in ranks_24:
            inference_iters = args.inference_iters_24
        elif fabric.global_rank in ranks_56:
            inference_iters = args.inference_iters_56
        
        if fabric.global_rank in ranks_2:
            total_loss, metrics = forward_batch_flow(batch, model, args, sw_t, inference_iters, horz_flip=args.horz_flip_2, vert_flip=args.vert_flip_2)
        else:
            utils.data.dataclass_to_cuda_(batch)
            total_loss, metrics = forward_batch_sparse(batch, model, args, sw_t, inference_iters)

        ftime = time.time()-f_start_time
        fabric.barrier() # wait for all gpus to finish their fwd
        
        b_start_time = time.time()
        if metrics is not None:
            if fabric.global_rank in log_ranks and sw_t.save_scalar:
                sw_t.summ_scalar('_/current_lr', optimizer.param_groups[-1]["lr"])

                if fabric.global_rank in ranks_2:
                    sw_t.summ_scalar('total_loss', metrics[0]['total_loss'])
                    # update stats
                    for b in range(len(metrics)):
                        metrics_b = metrics[b]
                        dname = metrics_b['dname']
                        if dname in dataset_names:
                            for key in list(pools_t[dname].keys()):
                                if key in metrics_b:
                                    pools_t[dname][key].update([metrics_b[key]])
                                    overpools_t[key].update([metrics_b[key]])
                    
                else:
                    sw_t.summ_scalar('total_loss', metrics['total_loss'])
                    # update stats
                    dname = metrics['dname']
                    if dname in dataset_names:
                        for key in list(pools_t[dname].keys()):
                            if key in metrics:
                                pools_t[dname][key].update([metrics[key]])
                                overpools_t[key].update([metrics[key]])
                # plot stats
                for key in list(overpools_t.keys()):
                    for dname in dataset_names:
                        sw_t.summ_scalar('%s/%s' % (dname, key), pools_t[dname][key].mean())
                    sw_t.summ_scalar('_/%s' % (key), overpools_t[key].mean())

            if args.mixed_precision:
                fabric.backward(total_loss)
                fabric.clip_gradients(model, optimizer, max_norm=1.0, norm_type=2, error_if_nonfinite=False)
                optimizer.step()
                if args.use_scheduler:
                    scheduler.step()
            else:
                fabric.backward(total_loss)
                fabric.clip_gradients(model, optimizer, max_norm=1.0, norm_type=2, error_if_nonfinite=False)
                optimizer.step()
                if args.use_scheduler:
                    scheduler.step()
        btime = time.time()-b_start_time
        fabric.barrier() # wait for all gpus to finish their bwd

        itime = ftime + btime
        
        if global_step % args.save_freq == 0:
            if metrics is not None: # otw something weird happened
                if fabric.global_rank == 0:
                    utils.saveload.save(save_dir, model.module, optimizer, scheduler, global_step, keep_latest=2)

        if fabric.global_rank in log_ranks:
            info_str = '%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, args.max_steps, rtime, itime)
            if overpools_t['total_loss'].have_min_size():
                info_str += '; loss_t %.2f; d_avg %.1f' % (
                    overpools_t['total_loss'].mean(),
                    overpools_t['d_avg'].mean()*100.0,
                )
            info_str += '; (rank %d)' % fabric.global_rank
            print(info_str)
    print('done!')

    if fabric.global_rank in log_ranks:
        writer_t.close()

if __name__ == "__main__":
    init_dir = ''

    # this file is for training alltracker in "stage 2", 
    # which involves mixing flow datasets with point tracking datasets.
    
    from nets.alltracker import Net; exp = 'stage2' # clean up for release
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=exp)
    parser.add_argument("--init_dir", type=str, default=init_dir)
    parser.add_argument("--load_optimizer", default=False, action='store_true')
    parser.add_argument("--load_scheduler", default=False, action='store_true')
    parser.add_argument("--load_step", default=False, action='store_true')
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints')
    parser.add_argument("--data_dir", type=str, default='/data')
    parser.add_argument("--batch_size_2", type=int, default=8)
    parser.add_argument("--batch_size_24", type=int, default=1)
    parser.add_argument("--batch_size_56", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers_2", type=int, default=2)
    parser.add_argument("--num_workers_24", type=int, default=2)
    parser.add_argument("--num_workers_56", type=int, default=6)
    parser.add_argument("--mixed_precision", default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wdecay", type=float, default=0.0005)
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--use_scheduler", default=False, action='store_true')
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=997) # prime number
    parser.add_argument("--traj_per_sample_24", type=int, default=256) # note we allow 1-24x this amount
    parser.add_argument("--traj_per_sample_56", type=int, default=256) # note we allow 1-24x this amount
    parser.add_argument("--inference_iters_2", type=int, default=4)
    parser.add_argument("--inference_iters_24", type=int, default=4)
    parser.add_argument("--inference_iters_56", type=int, default=3)
    parser.add_argument("--random_frame_rate", default=False, action='store_true')
    parser.add_argument("--use_augs", default=True)
    parser.add_argument("--seqlen", type=int, default=16)
    parser.add_argument("--crop_size_2", nargs="+", default=[384,768])
    parser.add_argument("--crop_size_24", nargs="+", default=[384,512])
    parser.add_argument("--crop_size_56", nargs="+", default=[256,384])
    parser.add_argument("--random_number_traj", default=False, action='store_true')
    parser.add_argument("--rand_stride", default=False, action='store_true')
    parser.add_argument("--use_huber_loss", default=False, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--random_seq_len", default=False, action='store_true')
    parser.add_argument("--no_attn", default=False, action='store_true')
    parser.add_argument("--no_short", default=False, action='store_true')
    parser.add_argument("--use_mixer", default=False, action='store_true')
    parser.add_argument("--use_conv", default=False, action='store_true')
    parser.add_argument("--use_convb", default=False, action='store_true')
    parser.add_argument("--use_basicencoder", default=False, action='store_true')
    parser.add_argument("--no_space", default=False, action='store_true')
    parser.add_argument("--no_time", default=False, action='store_true')
    parser.add_argument("--no_split", default=False, action='store_true')
    parser.add_argument("--full_split", default=False, action='store_true')
    parser.add_argument("--use_relmotion", default=False, action='store_true')
    parser.add_argument("--use_sinrelmotion", default=False, action='store_true')
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--corr_levels", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--horz_flip_2", default=False, action='store_true')
    parser.add_argument("--vert_flip_2", default=False, action='store_true')
    parser.add_argument("--shuffle_frames", default=False, action='store_true')

    args = parser.parse_args()

    # fix up integers
    args.crop_size_2 = (int(args.crop_size_2[0]), int(args.crop_size_2[1]))
    args.crop_size_24 = (int(args.crop_size_24[0]), int(args.crop_size_24[1]))
    args.crop_size_56 = (int(args.crop_size_56[0]), int(args.crop_size_56[1]))

    model = Net(
        16,
        dim=args.dim,
        hdim=args.hdim,
        use_attn=(not args.no_attn),
        use_mixer=args.use_mixer,
        use_conv=args.use_conv,
        use_convb=args.use_convb,
        use_basicencoder=args.use_basicencoder,
        no_space=args.no_space,
        no_time=args.no_time,
        use_relmotion=args.use_relmotion,
        use_sinrelmotion=args.use_sinrelmotion,
        no_split=args.no_split,
        full_split=args.full_split,
        num_blocks=args.num_blocks,
        corr_radius=args.corr_radius,
        corr_levels=args.corr_levels,
    )
    
    run(model, args)
    
