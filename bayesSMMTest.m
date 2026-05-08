cfg = struct();
cfg.experiment.seed = 42;

run('bayesSMMUnifiedMMCompare.m')

cfg.task = 'predict';
cfg.noise.inputNoise = true;
cfg.noise.distribution = 'gaussian';

run('bayesSMMUnifiedMMCompare.m')
