from mmcv.runner.hooks import HOOKS, Hook, LrUpdaterHook
import time


class CustomLog(Hook):
    def __init__(self, batch_size, when_defrost=None, out_file=None):
        self.avg_loss = -1
        self.batch_size = batch_size
        self.out_file = out_file
        self.when_defrost = when_defrost
        self.out = None
        self.t = None

    def before_run(self, runner):
        if self.out_file:
            self.out = open(self.out_file, 'a')

    def before_train_epoch(self, runner):
        self.t = time.time()
        if self.when_defrost is not None:
            if runner.epoch == self.when_defrost:
                log_line = 'Defrosting backbone'
                print(log_line)
                if self.out:
                    self.out.write(log_line + '\n')

    def after_train_iter(self, runner):
        if self.avg_loss < 0:
            self.avg_loss = runner.outputs['loss']
        else:
            self.avg_loss = self.avg_loss * 0.9 + runner.outputs['loss'] * 0.1
        log_line = '{}: {:.5}, {:.5} avg, {:.3} rate, {:.3} seconds, {} images'.format(runner.iter+1,
            runner.outputs['loss'], self.avg_loss, runner.current_lr()[0], time.time()-self.t,
            (runner.iter+1)*self.batch_size)
        print(log_line)
        if self.out:
            self.out.write(log_line + '\n')
        self.t = time.time()

    def after_train_epoch(self, runner):
        log_line = 'Saving weights (epoch {})'.format(runner.epoch+1)
        print(log_line)
        if self.out:
            self.out.write(log_line + '\n')

    def after_run(self, runner):
        if self.out:
            self.out.close()


class DefrostBackbone(Hook):
    def __init__(self, when_defrost, frozen_stages=-1):
        self.when_defrost = when_defrost
        self.frozen_stages = frozen_stages
        self.frozen = True

    def before_run(self, runner):
        if (runner.epoch >= self.when_defrost) and self.frozen:
            runner.model.module.backbone.frozen_stages = self.frozen_stages
            print('Backbone was defrosted (frozen_stages {})'.format(self.frozen_stages))
            self.frozen = False

    def after_train_epoch(self, runner):
        if ((runner.epoch+1) >= self.when_defrost) and self.frozen:
            runner.model.module.backbone.frozen_stages = self.frozen_stages
            print('Backbone was defrosted (frozen_stages {})'.format(self.frozen_stages))
            self.frozen = False


@HOOKS.register_module
class MultistepsLrUpdaterHook(LrUpdaterHook):
    def __init__(self, steps, scales, **kwargs):
        assert isinstance(steps, (list, tuple))
        assert isinstance(scales, (list, tuple))
        assert len(steps) == len(scales)
        for s in steps:
            assert isinstance(s, int) and s > 0
        for g in scales:
            assert g > 0
        self.steps = steps
        self.scales = scales
        super(MultistepsLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        factor = 1
        for s, g in zip(self.steps, self.scales):
            if progress >= s:
                factor *= g
        return base_lr * factor
