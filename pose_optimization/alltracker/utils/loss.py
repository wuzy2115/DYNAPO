import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
import utils.basic


def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    use_huber_loss=False,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if use_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * utils.basic.reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)

def sequence_loss_dense(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    use_huber_loss=False,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        # print('flow_gt[j]', flow_gt[j].shape)
        B, S, D, H, W = flow_gt[j].shape
        B, S2, _, H, W = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        # import ipdb; ipdb.set_trace()
        for i in range(n_predictions):
            # print('flow_e[j][i]', flow_preds[j][i].shape)
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i] # B,S,2,H,W
            if use_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0) # B,S,2,H,W
            else:
                i_loss = (flow_pred - flow_gt[j]).abs() # B,S,2,H,W
            i_loss_ = torch.mean(i_loss, dim=2) # B,S,H,W
            valid_ = valids[j].reshape(B,S,H,W)
            # print(' (%d,%d) i_loss_' % (i,j), i_loss_.shape)
            # print(' (%d,%d) valid_' % (i,j), valid_.shape)
            if loss_only_for_visible:
                valid_ = valid_ * vis[j].reshape(B,-1,H,W) # usually B,S,H,W, but maybe B,1,H,W
            flow_loss += i_weight * utils.basic.reduce_masked_mean(i_loss_, valid_, broadcast=True)
            # import ipdb; ipdb.set_trace()
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_BCE_loss(vis_preds, vis_gts, valids=None, use_logits=False):
    total_bce_loss = 0.0
    # all_vis_preds = [torch.stack(vp) for vp in vis_preds]
    # all_vis_preds = torch.stack(all_vis_preds)
    # utils.basic.print_stats('all_vis_preds', all_vis_preds)
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            # utils.basic.print_stats('vis_preds[%d][%d]' % (j,i), vis_preds[j][i])
            # utils.basic.print_stats('vis_gts[%d]' % (i), vis_gts[i])
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(vis_preds[j][i], vis_gts[j], reduction='none')
            else:
                loss = F.binary_cross_entropy(vis_preds[j][i], vis_gts[j], reduction='none')
            if valids is None:
                bce_loss += loss.mean()
            else:
                bce_loss += (loss * valids[j]).mean()
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


# def sequence_BCE_loss_dense(vis_preds, vis_gts):
#     total_bce_loss = 0.0
#     for j in range(len(vis_preds)):
#         n_predictions = len(vis_preds[j])
#         bce_loss = 0.0
#         for i in range(n_predictions):
#             vis_e = vis_preds[j][i]
#             vis_g = vis_gts[j]
#             print('vis_e', vis_e.shape, 'vis_g', vis_g.shape)
#             vis_loss = F.binary_cross_entropy(vis_e, vis_g)
#             bce_loss += vis_loss
#         bce_loss = bce_loss / n_predictions
#         total_bce_loss += bce_loss
#     return total_bce_loss / len(vis_preds)


def sequence_prob_loss(
        tracks: torch.Tensor,
        confidence: torch.Tensor,
        target_points: torch.Tensor,
        visibility: torch.Tensor,
        expected_dist_thresh: float = 12.0,
        use_logits=False,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(confidence[j][i], valid, reduction="none")
            else:
                loss = F.binary_cross_entropy(confidence[j][i], valid, reduction="none")
            loss *= visibility[j]
            loss = torch.mean(loss, dim=[1, 2])
            logprob_loss += loss
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)

def sequence_prob_loss_dense(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 12.0,
    use_logits=False,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.

    # all_confidence = [torch.stack(vp) for vp in confidence]
    # all_confidence = torch.stack(all_confidence)
    # utils.basic.print_stats('all_confidence', all_confidence)
    
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            # print('trajs_e', tracks[j][i].shape)
            # print('trajs_g', target_points[j].shape)
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=2)
            positive = (err <= expected_dist_thresh**2).float()
            # print('conf', confidence[j][i].shape, 'positive', positive.shape)
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(confidence[j][i].squeeze(2), positive, reduction="none")
            else:
                loss = F.binary_cross_entropy(confidence[j][i].squeeze(2), positive, reduction="none")
            loss *= visibility[j].squeeze(2) # B,S,H,W
            loss = torch.mean(loss, dim=[1,2,3])
            logprob_loss += loss
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def masked_mean(data, mask, dim):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)
