import os
import os.path as osp
import time
import json  # または import yaml
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS

@HOOKS.register_module()
class YOLOXAssignerINFOHook(Hook):

    def __init__(self,
                 filename,
                 interval,
                 out_dir=None,):
        self.filename = filename
        self.interval = interval
        self.out_dir = out_dir

        # フォーマットの決定
        if self.filename.endswith('.json'):
            self.dump_func = json.dump
            self.file_ext = '.json'
        elif self.filename.endswith('.yaml') or self.filename.endswith('.yml'):
            try:
                import yaml
                self.dump_func = yaml.dump
                self.file_ext = '.yaml'
            except ImportError:
                raise ImportError('Please install PyYAML to use YAML format.')
        else:
            raise ValueError('filename must end with .json, .yaml, or .yml')


    def _dump_log(self, log_dict, filepath):
        with open(filepath, 'w') as f:
            self.dump_func(log_dict, f, indent=4)


    def after_train_epoch(self, runner, *args, **kwargs):
        if not self.every_n_epochs(runner, self.interval):
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'detector'):
            _assigner = model.detector.bbox_head.assigner
        else:
            _assigner = model.bbox_head.assigner
        log_dict = _assigner.assigner_info

        filename = f'epoch_{runner.epoch:04d}{self.file_ext}'
        filepath = osp.join(self.log_dir, filename)
        self._dump_log(log_dict, filepath)
        _assigner.clean_assigner_info()
            

    def before_run(self, runner):
        if self.out_dir is None:
            self.log_dir = osp.join(runner.work_dir, str(time.time()))
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = self.out_dir
