from mmcv.runner.hooks import Hook


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
            log_line = 'Freezing backbone'
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
        if ((runner.epoch+1) >= self.when_defrost) and (self.when_defrost > 0):
            log_line = 'Defrosting backbone'
            print(log_line)
            if self.out:
                self.out.write(log_line + '\n')

    def after_run(self, runner):
        if self.out:
            self.out.close()


class DefrostBackbone(Hook):
    def __init__(self, when_defrost=0):
        self.when_defrost = when_defrost

    def before_run(self, runner):
        if self.when_defrost == 0:
            runner.model.module.backbone.frozen_stages = 1

    def after_train_epoch(self, runner):
        if (runner.epoch+1) >= self.when_defrost:
            runner.model.module.backbone.frozen_stages = 1