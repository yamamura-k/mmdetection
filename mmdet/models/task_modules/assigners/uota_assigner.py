# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor
import torch_extension as te

from mmdet.registry import TASK_UTILS
from .sim_ota_assigner import SimOTAAssigner

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class UOTAAssigner(SimOTAAssigner):

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        pairwise_ious_cpu = pairwise_ious.detach().cpu()
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious_cpu, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        self.assigner_info['dynamic_ks'].append(dynamic_ks.cpu().tolist())

        n_pred, n_gt = cost.shape
        nu = pairwise_ious_cpu.new_ones(n_pred).int()
        mu = pairwise_ious_cpu.new_ones(n_gt).int() # n_gt = num_gt + 1
        # mu[:-1] = (dynamic_ks - 2).clamp(min=1)
        mu[:-1] = dynamic_ks
        mu[-1] = n_pred
        matching_matrix = torch.from_numpy(
            te.solve_qubo_with_lemon_prune(n_pred, n_gt, nu.numpy(), mu.numpy(), cost.cpu().numpy().ravel(), pairwise_ious_cpu.numpy().ravel())
        ).to(valid_mask.device)
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
