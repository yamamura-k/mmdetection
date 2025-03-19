# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor
import ot

from mmdet.registry import TASK_UTILS
from .sim_ota_assigner import SimOTAAssigner

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class OTAAssigner(SimOTAAssigner):

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        self.assigner_info['dynamic_ks'].append(dynamic_ks.cpu().tolist())

        mu = pairwise_ious.new_ones(num_gt + 1)
        mu[:-1] = dynamic_ks.float()
        mu[-1] = cost.shape[0] - mu[:-1].sum()
        nu = pairwise_ious.new_ones(cost.shape[0])
        matching_matrix = ot.emd(nu, mu, cost, EPS)
        matching_matrix = matching_matrix[:, :-1]
        del topk_ious, dynamic_ks

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        num_matched_preds_per_gt = matching_matrix.sum(0).cpu().tolist()
        return matched_pred_ious, matched_gt_inds, num_matched_preds_per_gt

