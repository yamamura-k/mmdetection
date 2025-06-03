# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS, TASK_UTILS


@HOOKS.register_module()
class YOLOXAssignerSwitchHook(Hook):
    """Switch the assigner of YOLOX during training.

    FIXME: This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head. 

    FIXME: Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
       skip_type_keys (Sequence[str], optional): Sequence of type string to be
            skip pipeline. Defaults to ('Mosaic', 'RandomAffine', 'MixUp').
    """

    def __init__(
        self,
        num_warmup_epochs: int = 100,
    ) -> None:
        self.num_warmup_epochs = num_warmup_epochs
        self._has_switched = False

    def before_train_epoch(self, runner) -> None:
        """FIXME: Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
            
        epoch_to_be_switched = (epoch + 1) >= self.num_warmup_epochs
        if epoch_to_be_switched and not self._has_switched:
            runner.logger.info('Assigner is changed now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(model, 'detector'):
                model.detector.bbox_head.assigner = TASK_UTILS.build(model.train_cfg['assigner_latter'])
            else:
                model.bbox_head.assigner = TASK_UTILS.build(model.train_cfg['assigner_latter'])
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            pass
