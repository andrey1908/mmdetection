from mmcv.runner.hooks import HOOKS, Hook, LrUpdaterHook


class CustomLog(Hook):
    def __init__(self, batch_size, when_defrost=0, out_file=None):
        self.avg_loss = -1
        self.batch_size = batch_size
        self.out_file = out_file
        self.when_defrost = when_defrost
        self.out = None

    def before_run(self, runner):
        if self.out_file:
            self.out = open(self.out_file, 'a')
        if self.when_defrost > 0:
            if runner.epoch == 0:
                log_line = 'Freezing backbone'
                print(log_line)
                if self.out:
                    self.out.write(log_line + '\n')
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
        log_line = '{}: {:.5}, {:.5} avg, {:.3} rate, {} images'.format(runner.iter+1, runner.outputs['loss'], self.avg_loss,
                                                                        runner.current_lr()[0], (runner.iter+1)*self.batch_size)
        print(log_line)
        if self.out:
            self.out.write(log_line + '\n')

    def after_train_epoch(self, runner):
        log_line = 'Saving weights (epoch {})'.format(runner.epoch+1)
        print(log_line)
        if self.out:
            self.out.write(log_line + '\n')
        if (runner.epoch+1) == self.when_defrost:
            log_line = 'Defrosting backbone'
            print(log_line)
            if self.out:
                self.out.write(log_line + '\n')

    def after_run(self, runner):
        if self.out:
            self.out.close()


class DefrostBackbone(Hook):
    def __init__(self, when_defrost, defrosted_stages=-1):
        self.when_defrost = when_defrost
        self.defrosted_stages = defrosted_stages

    def before_run(self, runner):
        if runner.epoch >= self.when_defrost:
            runner.model.module.backbone.frozen_stages = self.defrosted_stages

    def after_train_epoch(self, runner):
        if (runner.epoch+1) >= self.when_defrost:
            runner.model.module.backbone.frozen_stages = self.defrosted_stages


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
