cfg = struct();
cfg.experiment.Ne = 10;
cfg.task = 'control';
cfg.experiment.seed = 42;
cfg.system.model = 'controlToeplitz';
cfg.noise.distribution = 'gaussian';
cfg.noise.inputNoise = false;
cfg.noise.correlation = 0;
cfg.noise.u_data_var = 0;
cfg.noise.u_var = 0;
cfg.noise.y_data_var = 1e-2;
cfg.noise.y_var = 1e-2;

run('bayesSMMCompare.m');
