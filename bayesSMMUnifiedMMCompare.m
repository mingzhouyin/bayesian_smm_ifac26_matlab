% Unified data-driven SMM MM comparison script.
%
% The experiment is configured by a single cfg struct. Smooth and predict
% share the same Bayesian trajectory-estimation pipeline; the task only
% changes Phi, the measured vector zeta, exact-input constraints, and the
% target output indices used for RMSE/coverage.
%
% Unified observation model:
%   zeta = Phi*z0 + epsilon,     z0 = H0*g = H*g - DeltaH*g.
% Here z = [u; y] is the stacked trajectory. Phi selects the noisy
% observations used by the current task, while Cz/AeqG enforce exact input
% information when cfg.noise.inputNoise is false.
%
% Example:
%   cfg = struct();
%   cfg.task = 'predict';
%   cfg.noise.inputNoise = true;
%   cfg.noise.distribution = 'gaussian';
%   cfg.experiment.seed = 3;
%   run('bayesSMMUnifiedMMCompare.m')
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin

clc; close all;
if ~exist('cfg', 'var') || isempty(cfg)
    cfg = default_bayes_smm_mm_compare_config();
else
    cfg = merge_structs(default_bayes_smm_mm_compare_config(), cfg);
end
cfg = finalize_bayes_smm_mm_compare_config(cfg);
results = run_bayes_smm_mm_compare(cfg);
assignin('base', 'results', results);

%% Main experiment
function results = run_bayes_smm_mm_compare(cfg)
    setup_bayes_smm_paths(cfg);
    rng(cfg.experiment.seed);

    options = optimoptions('fmincon', 'Algorithm', cfg.optim.Algorithm, ...
        'StepTolerance', cfg.optim.StepTolerance, ...
        'MaxFunctionEvaluations', cfg.optim.MaxFunctionEvaluations, ...
        'MaxIterations', cfg.optim.MaxIterations, ...
        'Display', cfg.optim.Display);

    dims = make_dimensions(cfg);
    mm = cfg.mm;
    methodNames = {'EB-fmincon', 'Convex', cfg.labels.subspace, 'Proj', ...
        'EmpBayes-MM', 'HierBayes-MM'};
    numMethods = numel(methodNames);
    Ne = cfg.experiment.Ne;

    errory = zeros(Ne, numMethods);
    t_calc = zeros(Ne, numMethods);
    ebMMIter = nan(Ne, 1);
    ebMMHitMaxIter = false(Ne, 1);
    ebMMFinalRelStep = nan(Ne, 1);
    hbMMIter = nan(Ne, 1);
    hbMMHitMaxIter = false(Ne, 1);
    hbMMFinalRelStep = nan(Ne, 1);
    hbRankGap = nan(Ne, 1);
    hbOriginalGradNorm = nan(Ne, 1);
    hbOriginalGradInfNorm = nan(Ne, 1);
    hbLaplaceDamping = nan(Ne, 1);
    hbLaplaceSolved = false(Ne, 1);
    hbLaplaceDamped = false(Ne, 1);
    hbLaplaceTime = zeros(Ne, 1);
    hbMinEigHtheta = nan(Ne, 1);
    hbNumNegEigHtheta = nan(Ne, 1);
    hbMinEigJzz = nan(Ne, 1);
    hbMinEigJgg = nan(Ne, 1);
    hbMinEigSchurG = nan(Ne, 1);

    hbLaplaceCovTheta = cell(Ne, 1);
    hbLaplaceCovZ = cell(Ne, 1);
    hbLaplaceCovTarget = cell(Ne, 1);
    ebCondCovZ = cell(Ne, 1);
    ebCondCovTarget = cell(Ne, 1);
    ebMMCondCovZ = cell(Ne, 1);
    ebMMCondCovTarget = cell(Ne, 1);
    hbLaplaceMeanStdTarget = nan(Ne, 1);
    ebCondMeanStdTarget = nan(Ne, 1);
    ebMMCondMeanStdTarget = nan(Ne, 1);
    hbLaplaceTarget90Inside = false(Ne, 1);
    hbLaplaceTarget90Stat = nan(Ne, 1);
    hbLaplaceTarget90Threshold = nan(Ne, 1);
    ebCondTarget90Inside = false(Ne, 1);
    ebCondTarget90Stat = nan(Ne, 1);
    ebCondTarget90Threshold = nan(Ne, 1);
    ebMMCondTarget90Inside = false(Ne, 1);
    ebMMCondTarget90Stat = nan(Ne, 1);
    ebMMCondTarget90Threshold = nan(Ne, 1);
    covCompareTraceRatioTarget = nan(Ne, 2);
    covCompareMedianDiagRatioTarget = nan(Ne, 2);
    covCompareRelFrobTarget = nan(Ne, 2);

    % CVX cannot call nonlinear covariance routines directly, so the
    % Toeplitz covariance is pre-expanded into basis matrices and combined
    % linearly with G inside the SDP relaxations.
    bases = make_covar_bases(cfg, dims);

    for ii = 1:Ne
        fprintf('Experiment %d/%d\n', ii, Ne);
        
        % Each repetition uses one simulated system/trajectory pair. The
        % task model below is the only place where smooth and predict split.
        data = simulate_bayes_smm_data(cfg, dims);
        model = build_task_model(cfg, dims, data);

        g0 = initial_g(model, dims, mm.jitter);

        % Empirical Bayes reference: direct fmincon minimization of the
        % marginal likelihood in g, followed by p(z | zeta, g).
        tic
        mmlObj = @(g) marginal_objective(g, model, dims, cfg);
        g_hat1 = solve_fmincon_g(mmlObj, g0, model, options);
        [z_est1, ebCondCovZ{ii}] = posterior_given_g(g_hat1, model, dims, cfg, options);
        t_calc(ii, 1) = toc;
        ebCondCovTarget{ii} = ebCondCovZ{ii}(model.targetIdx, model.targetIdx);
        ebCondMeanStdTarget(ii) = mean_std_from_cov(ebCondCovTarget{ii});

        % Convex SQP baseline iterates the quadratic approximation from
        % extended.tex until the coefficient update is small.
        tic
        g_hat2 = convex_baseline_g(g0, model, dims, cfg, mm.jitter);
        z_est2 = posterior_given_g(g_hat2, model, dims, cfg, options);
        t_calc(ii, 2) = toc;

        % Classical data-driven baseline retained for comparison with the
        % Bayesian SMM estimators.
        tic
        z_est3 = subspace_baseline(model, data, dims, cfg);
        t_calc(ii, 3) = toc;

        % Projection baseline solves the deterministic SMM equations with
        % the task-specific observation and exact-input constraints.
        tic
        z_est4 = projection_baseline(model, data, dims, cfg, mm.jitter);
        t_calc(ii, 4) = toc;

        % Empirical-Bayes MM majorizes the marginal likelihood and solves
        % one SDP per iteration for the lifted variables (g, G).
        tic
        [z_est5, g_hat5, ebInfo] = empirical_bayes_mm(model, dims, cfg, bases, g0, options);
        [~, ebMMCondCovZ{ii}] = posterior_given_g(g_hat5, model, dims, cfg, options, z_est5);
        ebMMIter(ii) = ebInfo.iter;
        ebMMHitMaxIter(ii) = ebInfo.hitMaxIter;
        ebMMFinalRelStep(ii) = ebInfo.finalRelStep;
        t_calc(ii, 5) = toc;
        ebMMCondCovTarget{ii} = ebMMCondCovZ{ii}(model.targetIdx, model.targetIdx);
        ebMMCondMeanStdTarget(ii) = mean_std_from_cov(ebMMCondCovTarget{ii});

        % Hierarchical-Bayes MM jointly estimates z and g, then the Laplace
        % block below approximates the local posterior covariance.
        tic
        lambda_g = max(sum(g0.^2), cfg.mm.jitter);
        [z_est6, g_hat6, G_hat6, hbInfo] = hierarchical_bayes_mm( ...
            model, dims, cfg, bases, g0, lambda_g);
        hbMMIter(ii) = hbInfo.iter;
        hbMMHitMaxIter(ii) = hbInfo.hitMaxIter;
        hbMMFinalRelStep(ii) = hbInfo.finalRelStep;
        hbRankGap(ii) = trace(G_hat6) - sum(g_hat6.^2);
        [hbOriginalGradNorm(ii), hbOriginalGradInfNorm(ii)] = ...
            original_map_gradient_norm_generic(z_est6, g_hat6, model, dims, cfg, lambda_g);
        t_calc(ii, 6) = toc;

        tic
        [hbLaplaceCovZ{ii}, hbLaplaceCovTheta{ii}, laplaceInfo] = ...
            laplace_posterior_covariance_generic(z_est6, g_hat6, model, dims, cfg, lambda_g);
        hbLaplaceTime(ii) = toc;
        hbLaplaceCovTarget{ii} = hbLaplaceCovZ{ii}(model.targetIdx, model.targetIdx);
        hbLaplaceMeanStdTarget(ii) = mean_std_from_cov(hbLaplaceCovTarget{ii});
        hbLaplaceDamping(ii) = laplaceInfo.damping;
        hbLaplaceSolved(ii) = laplaceInfo.solved;
        hbLaplaceDamped(ii) = laplaceInfo.damping > 0;
        hbMinEigHtheta(ii) = laplaceInfo.minEigHtheta;
        hbNumNegEigHtheta(ii) = laplaceInfo.numNegEigHtheta;
        hbMinEigJzz(ii) = laplaceInfo.minEigJzz;
        hbMinEigJgg(ii) = laplaceInfo.minEigJgg;
        hbMinEigSchurG(ii) = laplaceInfo.minEigSchurG;

        targetTrue = data.z_true(model.targetIdx);
        [hbLaplaceTarget90Inside(ii), hbLaplaceTarget90Stat(ii), hbLaplaceTarget90Threshold(ii)] = ...
            confidence_region_contains(targetTrue - z_est6(model.targetIdx), ...
            hbLaplaceCovTarget{ii}, 0.90, cfg, mm.jitter);
        [ebCondTarget90Inside(ii), ebCondTarget90Stat(ii), ebCondTarget90Threshold(ii)] = ...
            confidence_region_contains(targetTrue - z_est1(model.targetIdx), ...
            ebCondCovTarget{ii}, 0.90, cfg, mm.jitter);
        [ebMMCondTarget90Inside(ii), ebMMCondTarget90Stat(ii), ebMMCondTarget90Threshold(ii)] = ...
            confidence_region_contains(targetTrue - z_est5(model.targetIdx), ...
            ebMMCondCovTarget{ii}, 0.90, cfg, mm.jitter);
        [covCompareTraceRatioTarget(ii, 1), covCompareMedianDiagRatioTarget(ii, 1), ...
            covCompareRelFrobTarget(ii, 1)] = covariance_compare( ...
            hbLaplaceCovTarget{ii}, ebCondCovTarget{ii});
        [covCompareTraceRatioTarget(ii, 2), covCompareMedianDiagRatioTarget(ii, 2), ...
            covCompareRelFrobTarget(ii, 2)] = covariance_compare( ...
            hbLaplaceCovTarget{ii}, ebMMCondCovTarget{ii});

        estimates = {z_est1, z_est2, z_est3, z_est4, z_est5, z_est6};
        for jj = 1:numMethods
            errory(ii, jj) = norm(estimates{jj}(model.targetIdx) - targetTrue) ...
                /sqrt(numel(model.targetIdx));
        end

        if ~ebInfo.solved
            warning('EmpBayes-MM did not fully solve in experiment %d. Last CVX status: %s', ...
                ii, ebInfo.status);
        end
        if ~hbInfo.solved
            warning('HierBayes-MM did not fully solve in experiment %d. Last CVX status: %s', ...
                ii, hbInfo.status);
        end
    end

    results = build_results(cfg, dims, methodNames, errory, t_calc, ...
        ebMMIter, ebMMHitMaxIter, ebMMFinalRelStep, hbMMIter, ...
        hbMMHitMaxIter, hbMMFinalRelStep, hbRankGap, hbOriginalGradNorm, ...
        hbOriginalGradInfNorm, hbLaplaceDamping, hbLaplaceSolved, ...
        hbLaplaceDamped, hbLaplaceTime, hbMinEigHtheta, hbNumNegEigHtheta, ...
        hbMinEigJzz, hbMinEigJgg, hbMinEigSchurG, hbLaplaceCovTheta, ...
        hbLaplaceCovZ, hbLaplaceCovTarget, hbLaplaceMeanStdTarget, ...
        ebCondCovZ, ebCondCovTarget, ebCondMeanStdTarget, ebMMCondCovZ, ...
        ebMMCondCovTarget, ebMMCondMeanStdTarget, hbLaplaceTarget90Inside, ...
        hbLaplaceTarget90Stat, hbLaplaceTarget90Threshold, ebCondTarget90Inside, ...
        ebCondTarget90Stat, ebCondTarget90Threshold, ebMMCondTarget90Inside, ...
        ebMMCondTarget90Stat, ebMMCondTarget90Threshold, covCompareTraceRatioTarget, ...
        covCompareMedianDiagRatioTarget, covCompareRelFrobTarget);

    if cfg.save.enabled
        save(cfg.resultFile, 'results');
    end
    print_results(results);
    if cfg.plot.enabled
        plot_results(results);
    end
end

%% Configuration and setup
function cfg = default_bayes_smm_mm_compare_config()
    cfg = struct();
    cfg.task = 'smooth';
    cfg.experiment.seed = 1;
    cfg.experiment.Ne = 100;
    cfg.system.nx = 10;
    cfg.system.nu = 1;
    cfg.system.ny = 1;
    cfg.system.maxPoleMagnitude = 0.95;
    cfg.data.N = 100;
    cfg.data.L = 40;
    cfg.noise.distribution = 'studentT';
    cfg.noise.inputNoise = false;
    cfg.noise.dof = 10;
    cfg.noise.u_data_var = 1e-4;
    cfg.noise.y_data_var = 1e-4;
    cfg.noise.u_var = 1e-2;
    cfg.noise.y_var = 1e-2;
    cfg.mm.maxIter = 100;
    cfg.mm.tol = 1e-3;
    cfg.mm.eta = 1e-4;
    cfg.mm.jitter = 1e-9;
    cfg.sqp.maxIter = 100;
    cfg.sqp.tol = 1e-6;
    cfg.sqp.minRegularization = 1e-9;
    cfg.optim.Algorithm = 'interior-point';
    cfg.optim.StepTolerance = 1e-14;
    cfg.optim.MaxFunctionEvaluations = 1e5;
    cfg.optim.MaxIterations = 1e4;
    cfg.optim.Display = 'off';
    cfg.paths.resultDir = pwd;
    cfg.paths.cvxDir = fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'cvx');
    cfg.paths.rtsDir = fullfile(pwd, 'rts');
    cfg.save.enabled = true;
    cfg.save.resultFile = '';
    cfg.plot.enabled = true;
    cfg.labels.subspace = '';
end

function cfg = finalize_bayes_smm_mm_compare_config(cfg)
    cfg.task = validatestring(cfg.task, {'smooth', 'predict'});
    cfg.noise.distribution = validatestring(cfg.noise.distribution, ...
        {'gaussian', 'studentT', 'ellipticalT'});
    if strcmpi(cfg.noise.distribution, 'ellipticalT')
        cfg.noise.distribution = 'studentT';
    end
    if isempty(cfg.labels.subspace)
        if strcmpi(cfg.task, 'smooth')
            cfg.labels.subspace = 'N4SID+KF';
        else
            cfg.labels.subspace = 'N4SID';
        end
    end
    if ~isfield(cfg.save, 'resultFile') || isempty(cfg.save.resultFile)
        inputTag = bool_tag(cfg.noise.inputNoise, 'inputNoise', 'exactInput');
        cfg.save.resultFile = fullfile(cfg.paths.resultDir, sprintf( ...
            'bayesSMMUnifiedMMCompare_%s_%s_%s_results.mat', ...
            lower(cfg.task), lower(cfg.noise.distribution), inputTag));
    end
    cfg.resultFile = cfg.save.resultFile;
end

function setup_bayes_smm_paths(cfg)
    if exist('cvx_begin', 'file') == 0 && exist(fullfile(cfg.paths.cvxDir, 'cvx_startup.m'), 'file')
        run(fullfile(cfg.paths.cvxDir, 'cvx_startup.m'));
    end
    if exist('KF_RTS', 'file') == 0 && exist(cfg.paths.rtsDir, 'dir')
        addpath(genpath(cfg.paths.rtsDir));
    end
end

function out = merge_structs(defaults, overrides)
    out = defaults;
    names = fieldnames(overrides);
    for ii = 1:numel(names)
        name = names{ii};
        if isfield(out, name) && isstruct(out.(name)) && isstruct(overrides.(name))
            out.(name) = merge_structs(out.(name), overrides.(name));
        else
            out.(name) = overrides.(name);
        end
    end
end

function tag = bool_tag(value, trueTag, falseTag)
    if value
        tag = trueTag;
    else
        tag = falseTag;
    end
end

function dims = make_dimensions(cfg)
    % Indices are defined on the common stacked vector z = [u; y]. For
    % prediction, y is split into past yp and future yf by L0 = nx.
    dims.nx = cfg.system.nx;
    dims.nu = cfg.system.nu;
    dims.ny = cfg.system.ny;
    dims.N = cfg.data.N;
    dims.L = cfg.data.L;
    dims.M = dims.N - dims.L + 1;
    dims.L0 = dims.nx;
    dims.Lf = dims.L - dims.L0;
    dims.idx_u = dims.L*dims.nu;
    dims.idx_y = dims.L*dims.ny;
    dims.idx_yp = dims.L0*dims.ny;
    dims.idx_yf = dims.Lf*dims.ny;
    dims.nz = dims.idx_u + dims.idx_y;
    dims.uIdx = 1:dims.idx_u;
    dims.yIdx = dims.idx_u + (1:dims.idx_y);
    dims.ypIdx = dims.idx_u + (1:dims.idx_yp);
    dims.yfIdx = dims.idx_u + dims.idx_yp + (1:dims.idx_yf);
end

%% Model construction
function data = simulate_bayes_smm_data(cfg, dims)
    % Generate one stable LTI system plus offline data (ud, yd) and online
    % task data (u, y). The *_dist variables are the actually observed
    % noisy trajectories used to build Hankel matrices and zeta.
    trueSys = drss(dims.nx, dims.ny, dims.nu);
    trueSys.D = 0;
    while max(abs(pole(trueSys))) > cfg.system.maxPoleMagnitude
        trueSys = drss(dims.nx, dims.ny, dims.nu);
        trueSys.D = 0;
    end
    trueSys = trueSys/norm(trueSys);

    ud = randn(dims.N, dims.nu);
    yd = lsim(trueSys, ud);
    u = randn(dims.L, dims.nu);
    y = lsim(trueSys, u);

    ud_dist = ud + draw_noise(size(ud), effective_var(cfg, 'u_data'), cfg);
    yd_dist = yd + draw_noise(size(yd), cfg.noise.y_data_var, cfg);
    u_dist = u + draw_noise(size(u), effective_var(cfg, 'u_online'), cfg);
    y_dist = y + draw_noise(size(y), cfg.noise.y_var, cfg);

    data.trueSys = trueSys;
    data.ud = ud;
    data.yd = yd;
    data.u = u;
    data.y = y;
    data.ud_dist = ud_dist;
    data.yd_dist = yd_dist;
    data.u_dist = u_dist(:);
    data.y_dist = y_dist(:);
    data.Hu = GenHankel(ud_dist, dims.L);
    data.Hy = GenHankel(yd_dist, dims.L);
    data.H = [data.Hu; data.Hy];
    data.z_true = [u(:); y(:)];
    data.z_dist = [data.u_dist; data.y_dist];
end

function value = effective_var(cfg, whichVar)
    % Input-noise-off experiments treat input measurements as exact, so the
    % corresponding offline/online input variances are zeroed consistently.
    switch whichVar
        case 'u_data'
            value = cfg.noise.u_data_var*double(cfg.noise.inputNoise);
        case 'u_online'
            value = cfg.noise.u_var*double(cfg.noise.inputNoise);
        otherwise
            error('Unknown variance selector: %s', whichVar);
    end
end

function noise = draw_noise(sz, variance, cfg)
    if variance == 0
        noise = zeros(sz);
        return
    end
    noise = sqrt(variance)*randn(sz);
    if strcmpi(cfg.noise.distribution, 'studentT')
        noise = noise/sqrt(chi2rnd(cfg.noise.dof)/cfg.noise.dof);
    end
end

function model = build_task_model(cfg, dims, data)
    % Build the task-dependent selectors for the unified model. All later
    % estimators consume model.Phi/model.zeta/model.Cz/model.AeqG and do not
    % need separate smooth/predict branches.
    H = data.H;
    spec = task_observation_spec(cfg, dims, data);

    if cfg.noise.inputNoise
        obsIdx = [dims.uIdx, spec.yObsIdx];
        zeta = [data.u_dist; spec.yZeta];
        sigmaE = diag([cfg.noise.u_var*ones(dims.idx_u, 1); spec.yNoiseVar]);
        exactIdx = [];
        exactValue = zeros(0, 1);
        AeqG = zeros(0, dims.M);
        beqG = zeros(0, 1);
    else
        obsIdx = spec.yObsIdx;
        zeta = spec.yZeta;
        sigmaE = diag(spec.yNoiseVar);
        exactIdx = dims.uIdx;
        exactValue = data.u_dist;
        AeqG = data.Hu;
        beqG = data.u_dist;
    end

    % Phi and Cz are row-selection matrices on z. AeqG/beqG are the matching
    % exact constraints in coefficient space because z = H*g.
    Phi = selection_matrix(obsIdx, dims.nz);
    Cz = selection_matrix(exactIdx, dims.nz);
    hasExactG = ~isempty(exactIdx);
    model.H = H;
    model.Phi = Phi;
    model.PhiH = Phi*H;
    model.zeta = zeta;
    model.SigmaE = make_spd(sigmaE, cfg.mm.jitter);
    model.Robs = chol(model.SigmaE\eye(size(model.SigmaE)));
    model.obsIdx = obsIdx;
    model.targetIdx = spec.targetIdx;
    model.targetName = spec.targetName;
    model.Cz = Cz;
    model.exactZValue = exactValue;
    model.AeqG = AeqG;
    model.beqG = beqG;
    model.hasExactZ = ~isempty(exactIdx);
    model.hasExactG = hasExactG;
    model.freeZDim = dims.nz - size(Cz, 1);
    model.task = cfg.task;
end

function spec = task_observation_spec(cfg, dims, data)
    % The task controls which output samples are observed and which output
    % samples are scored. Input-noise handling is added by build_task_model.
    switch lower(cfg.task)
        case 'smooth'
            spec.yObsIdx = dims.yIdx;
            spec.yZeta = data.y_dist;
            spec.yNoiseVar = cfg.noise.y_var*ones(dims.idx_y, 1);
            spec.targetIdx = dims.yIdx;
            spec.targetName = 'smoothed output y';
        case 'predict'
            spec.yObsIdx = dims.ypIdx;
            spec.yZeta = data.y_dist(1:dims.idx_yp);
            spec.yNoiseVar = cfg.noise.y_var*ones(dims.idx_yp, 1);
            spec.targetIdx = dims.yfIdx;
            spec.targetName = 'future output y_f';
        otherwise
            error('Unsupported task: %s', cfg.task);
    end
end

function S = selection_matrix(indices, width)
    if isempty(indices)
        S = zeros(0, width);
        return
    end
    S = zeros(numel(indices), width);
    for ii = 1:numel(indices)
        S(ii, indices(ii)) = 1;
    end
end

%% Estimation methods
function g0 = initial_g(model, dims, jitter)
    % A small least-squares estimate gives every nonlinear/MM method the
    % same feasible starting point.
    Q = model.PhiH'*model.PhiH + jitter*eye(dims.M);
    c = model.PhiH'*model.zeta;
    g0 = solve_model_quadratic(Q, c, model, jitter);
end

function g = solve_fmincon_g(obj, g0, model, options)
    g = fmincon(obj, g0, [], [], model.AeqG, model.beqG, [], [], [], options);
end

function g = solve_model_quadratic(Q, c, model, jitter)
    if model.hasExactG
        g = equality_constrained_quadratic(Q, c, model.AeqG, model.beqG, jitter);
    else
        g = solve_regularized_quadratic(Q, c, jitter);
    end
end

function value = marginal_objective(g, model, dims, cfg)
    % Marginal likelihood of zeta after integrating out the latent
    % trajectory uncertainty induced by noisy Hankel data.
    Psi = model.Phi*trajectory_covar(g, dims, cfg)*model.Phi' + model.SigmaE;
    Psi = make_spd(Psi, cfg.mm.jitter);
    residual = model.zeta - model.PhiH*g;
    q = max(residual'*(Psi\residual), 0);
    logdetPsi = stable_logdet(Psi);
    if strcmpi(cfg.noise.distribution, 'gaussian')
        % Gaussian negative log-likelihood up to constants.
        value = logdetPsi + q;
    else
        % Elliptical Student-t version uses the same Mahalanobis residual
        % with a heavy-tailed log penalty.
        value = logdetPsi + (cfg.noise.dof + numel(model.zeta))*log(1 + q/cfg.noise.dof);
    end
end

function g = convex_baseline_g(g0, model, dims, cfg, jitter)
    g = g0;
    iter = 0;
    relStep = inf;
    while iter < cfg.sqp.maxIter && relStep > cfg.sqp.tol
        iter = iter + 1;
        [Q, c] = convex_sqp_quadratic_model(g, model, dims, cfg, jitter);
        gNext = solve_model_quadratic(Q, c, model, jitter);
        relStep = norm(gNext - g)/max(1, norm(g));
        g = gNext;
    end
end

function [Q, c] = convex_sqp_quadratic_model(g, model, dims, cfg, jitter)
    Lambda = sqp_observation_data_covariance(model, dims, cfg);
    Psi = sum(g.^2)*Lambda + model.SigmaE;
    Psi = make_spd(Psi, jitter);
    W = Psi\eye(size(Psi));

    residual = model.zeta - model.PhiH*g;
    weightedResidual = W*residual;
    regularization = real(trace(Lambda*W) - weightedResidual'*Lambda*weightedResidual);
    regularizationFloor = max(cfg.sqp.minRegularization, jitter);
    if ~isfinite(regularization)
        regularization = regularizationFloor;
    end
    regularization = max(regularization, regularizationFloor);

    Q = model.PhiH'*W*model.PhiH + regularization*eye(dims.M);
    c = model.PhiH'*W*model.zeta;
end

function Lambda = sqp_observation_data_covariance(model, dims, cfg)
    % Under the Page/zero-lag approximation used by the SQP derivation,
    % trajectory uncertainty is ||g||^2 times the selected data covariance.
    dataVariance = [effective_var(cfg, 'u_data')*ones(dims.idx_u, 1); ...
        cfg.noise.y_data_var*ones(dims.idx_y, 1)];
    Lambda = model.Phi*diag(dataVariance)*model.Phi';
    Lambda = (Lambda + Lambda')/2;
end

function z = subspace_baseline(~, data, dims, cfg)
    sys = n4sid(data.ud_dist, data.yd_dist, dims.nx);
    switch lower(cfg.task)
        case 'smooth'
            X_filt = KF_RTS(data.y_dist', sys.A, sys.C, ...
                cfg.noise.y_var*sys.K*sys.K', cfg.noise.y_var*ones(dims.ny, 1), ...
                B=sys.B, u=data.u_dist);
            y_est = (sys.C*X_filt)';
            z = data.z_dist;
            z(dims.yIdx) = y_est(:);
        case 'predict'
            y_est = compare(iddata(data.y_dist, data.u_dist), sys);
            z = data.z_dist;
            z(dims.yfIdx) = y_est(dims.idx_yp+1:end).OutputData;
    end
end

function z = projection_baseline(model, data, dims, cfg, jitter)
    if strcmpi(cfg.task, 'smooth') && model.hasExactG
        H1 = [data.Hu; data.Hy(1:dims.idx_yp, :)];
        Hyf = data.Hy(dims.idx_yp+1:end, :);
        Hyfhat = Hyf*H1'*((H1*H1')\H1);
        Hyhat = [data.Hy(1:dims.idx_yp, :); Hyfhat];
        g = equality_constrained_quadratic(Hyhat'*Hyhat, ...
            Hyhat'*data.y_dist, model.AeqG, model.beqG, jitter);
        z = data.z_dist;
        z(dims.yIdx) = Hyhat*g;
        return
    end

    g = projection_g_from_observations(model, jitter);
    z = model.H*g;
    z = enforce_exact_z(z, model);
end

function g = projection_g_from_observations(model, jitter)
    Q = model.PhiH'*model.PhiH;
    c = model.PhiH'*model.zeta;
    if model.hasExactG || size(model.PhiH, 1) > size(model.PhiH, 2)
        g = solve_model_quadratic(Q, c, model, jitter);
        return
    end

    gram = make_spd(model.PhiH*model.PhiH', jitter);
    g = model.PhiH'*(gram\model.zeta);
end

function [z_est, cov_z] = posterior_given_g(g, model, dims, cfg, options, z_init)
    % Estimate z conditional on g. Gaussian noise gives a constrained
    % quadratic closed form; Student-t noise is solved by fmincon and uses a
    % local Hessian covariance approximation.
    if nargin < 6 || isempty(z_init)
        z_init = model.H*g;
        if model.hasExactZ
            z_init = enforce_exact_z(z_init, model);
        end
    end

    S = make_spd(trajectory_covar(g, dims, cfg), cfg.mm.jitter);
    P = S\eye(dims.nz);
    R = model.SigmaE\eye(size(model.SigmaE));

    HzzGaussian = P + model.Phi'*R*model.Phi;
    rhs = P*(model.H*g) + model.Phi'*R*model.zeta;
    if strcmpi(cfg.noise.distribution, 'gaussian')
        [z_est, cov_z] = solve_quadratic_z(HzzGaussian, rhs, model, cfg.mm.jitter);
        return
    end

    [z_mean, ~] = solve_quadratic_z(HzzGaussian, rhs, model, cfg.mm.jitter);
    if nargin < 6 || isempty(z_init)
        z_init = z_mean;
    end
    obj = @(z) posterior_z_objective_student_t(z, g, model, S, cfg);
    z_est = fmincon(obj, z_init, [], [], model.Cz, model.exactZValue, ...
        [], [], [], options);
    Hzz = posterior_z_hessian_elliptical(z_est, g, model, S, cfg);
    cov_z = constrained_covariance_from_hessian(Hzz, model.Cz, cfg.mm.jitter);
    cov_z = distribution_laplace_scale(cfg, model.freeZDim)*cov_z;
end

function value = posterior_z_objective_student_t(z, g, model, S, cfg)
    % Conditional z objective for the heavy-tailed model: one Student-t term
    % for measurement residuals and one for Hankel/model uncertainty.
    obs = model.zeta - model.Phi*z;
    qObs = max(obs'*(model.SigmaE\obs), 0);
    modelRes = z - model.H*g;
    qModel = max(modelRes'*(S\modelRes), 0);
    value = log(1 + qObs/cfg.noise.dof) + log(1 + qModel/cfg.noise.dof);
end

function [z, cov_z] = solve_quadratic_z(Hzz, rhs, model, jitter)
    % Solve min 0.5*z'*Hzz*z - rhs'*z with optional exact constraints Cz*z.
    % The nullspace form keeps constrained covariance consistent with the
    % returned constrained mean.
    Hzz = make_spd(Hzz, jitter);
    if model.hasExactZ
        z0 = model.Cz'*(model.Cz*model.Cz'\model.exactZValue);
        Nz = null(model.Cz);
        Hred = make_spd(Nz'*Hzz*Nz, jitter);
        z = z0 + Nz*(Hred\(Nz'*(rhs - Hzz*z0)));
        cov_z = constrained_covariance_from_hessian(Hzz, model.Cz, jitter);
    else
        z = Hzz\rhs;
        cov_z = Hzz\eye(size(Hzz));
        cov_z = (cov_z + cov_z')/2;
    end
end

%% MM solvers
function [z_est, g_opt, info] = empirical_bayes_mm(model, dims, cfg, bases, g_init, options)
    % MM for the marginal-likelihood estimator. The lifted matrix G
    % represents g*g' in the covariance term and is relaxed by [G g; g' 1].
    g_prev = g_init;
    info = init_mm_info(cfg.mm);

    for iter = 1:cfg.mm.maxIter
        % Reweight only the scalar residual term for Student-t noise; in the
        % Gaussian case w is constant and the update reduces to the Gaussian
        % majorizer.
        A_prev = model.Phi*trajectory_covar(g_prev, dims, cfg)*model.Phi' + model.SigmaE;
        A_prev = make_spd(A_prev, cfg.mm.jitter);
        residual_prev = model.zeta - model.PhiH*g_prev;
        q_prev = max(residual_prev'*(A_prev\residual_prev), 0);
        w = mml_mm_weight(q_prev, numel(model.zeta), cfg);

        cvx_begin quiet sdp
            variable g(dims.M)
            variable G(dims.M, dims.M) symmetric
            variable t
            B = cvx_trajectory_covar(G, bases, dims);
            A = model.Phi*B*model.Phi' + model.SigmaE + 0*G(1, 1)*eye(numel(model.zeta));
            residual = model.zeta - model.PhiH*g;
            minimize(0.5*trace(A_prev\A) + w*t ...
                + cfg.mm.eta*trace(G) - 2*cfg.mm.eta*transpose(g_prev)*g)
            subject to
                if model.hasExactG
                    model.AeqG*g == model.beqG;
                    model.AeqG*G == model.beqG*transpose(g);
                end
                t >= 0;
                [A residual; residual' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        [g_prev, info, stopNow] = update_mm_info(g, g_prev, cvx_status, cvx_optval, iter, cfg.mm, info);
        if stopNow
            break
        end
    end

    g_opt = g_prev;
    z_est = posterior_given_g(g_opt, model, dims, cfg, options);
end

function [z_est, g_opt, G_opt, info] = hierarchical_bayes_mm(model, dims, cfg, bases, g_init, lambda_g)
    % Joint MAP/MM estimator for (z, g). This keeps the same task selectors
    % as empirical Bayes but optimizes the latent trajectory explicitly.
    g_prev = g_init;
    z_est = model.H*g_prev;
    if model.hasExactZ
        z_est = enforce_exact_z(z_est, model);
    end
    G_opt = g_prev*g_prev';
    info = init_mm_info(cfg.mm);
    info.rankGap = nan(cfg.mm.maxIter, 1);

    for iter = 1:cfg.mm.maxIter
        % Observation and model residuals get separate Student-t weights
        % because they live in different dimensions and covariance spaces.
        B_prev = make_spd(trajectory_covar(g_prev, dims, cfg), cfg.mm.jitter);
        obs_prev = model.zeta - model.Phi*z_est;
        model_prev = z_est - model.H*g_prev;
        q_obs = max(obs_prev'*(model.SigmaE\obs_prev), 0);
        q_model = max(model_prev'*(B_prev\model_prev), 0);
        [w_obs, w_model] = hierarchical_mm_weights(q_obs, q_model, ...
            numel(model.zeta), dims.nz, cfg);

        cvx_begin quiet sdp
            variable z(dims.nz)
            variable g(dims.M)
            variable G(dims.M, dims.M) symmetric
            variable t
            B = cvx_trajectory_covar(G, bases, dims);
            Breg = B + cfg.mm.jitter*eye(dims.nz);
            obs_res = model.zeta - model.Phi*z;
            model_res = z - model.H*g;
            minimize(trace(B_prev\Breg) + w_obs*sum_square(model.Robs*obs_res) ...
                + w_model*t + (1/lambda_g)*sum_square(g) ...
                + cfg.mm.eta*trace(G) - 2*cfg.mm.eta*transpose(g_prev)*g)
            subject to
                if model.hasExactZ
                    model.Cz*z == model.exactZValue;
                end
                if model.hasExactG
                    model.AeqG*g == model.beqG;
                    model.AeqG*G == model.beqG*transpose(g);
                end
                t >= 0;
                [Breg model_res; model_res' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        if ~contains(cvx_status, 'Solved') || any(~isfinite(g)) || any(~isfinite(z))
            info.solved = false;
            info.status = cvx_status;
            info.iter = iter;
            break
        end

        g_new = full(g);
        z_est = full(z);
        G_opt = full(G);
        info.rankGap(iter) = trace(G_opt) - sum(g_new.^2);
        [g_prev, info, stopNow] = update_mm_info(g_new, g_prev, cvx_status, cvx_optval, iter, cfg.mm, info);
        if stopNow
            break
        end
    end

    g_opt = g_prev;
    info.finalRankGap = trace(G_opt) - sum(g_opt.^2);
end

function z = enforce_exact_z(z, model)
    if model.hasExactZ
        z = z + model.Cz'*(model.exactZValue - model.Cz*z);
    end
end

function w = mml_mm_weight(q, dim, cfg)
    % Weight of the quadratic residual majorizer. Student-t residuals are
    % down-weighted when their Mahalanobis distance is large.
    w = elliptical_rho_derivatives(q, dim, cfg);
end

function [wObs, wModel] = hierarchical_mm_weights(qObs, qModel, obsDim, modelDim, cfg)
    % Separate robust weights for measurement and model residual blocks.
    wObs = 2*elliptical_rho_derivatives(qObs, obsDim, cfg);
    wModel = 2*elliptical_rho_derivatives(qModel, modelDim, cfg);
end

function [rho1, rho2] = elliptical_rho_derivatives(q, dim, cfg)
    if strcmpi(cfg.noise.distribution, 'gaussian')
        rho1 = 0.5;
        rho2 = 0;
    else
        rho1 = 0.5*(cfg.noise.dof + dim)/(cfg.noise.dof + q);
        rho2 = -0.5*(cfg.noise.dof + dim)/(cfg.noise.dof + q)^2;
    end
end

function scale = distribution_laplace_scale(cfg, dimension)
    if strcmpi(cfg.noise.distribution, 'gaussian')
        scale = 1;
    else
        scale = elliptical_t_laplace_scale(cfg.noise.dof, dimension);
    end
end

function info = init_mm_info(mm)
    info.solved = true;
    info.status = 'Not started';
    info.iter = 0;
    info.obj = nan(mm.maxIter, 1);
    info.finalRelStep = nan;
    info.converged = false;
    info.hitMaxIter = false;
end

function [g_new, info, stopNow] = update_mm_info(g_value, g_prev, status, optval, iter, mm, info)
    info.status = status;
    info.obj(iter) = optval;
    info.iter = iter;
    stopNow = false;
    if ~contains(status, 'Solved') || any(~isfinite(g_value))
        info.solved = false;
        stopNow = true;
        g_new = g_prev;
        return
    end
    g_new = full(g_value);
    info.finalRelStep = norm(g_new - g_prev)/max(1, norm(g_prev));
    if info.finalRelStep < mm.tol
        info.converged = true;
        stopNow = true;
    end
    info.hitMaxIter = info.solved && iter == mm.maxIter && ~info.converged;
end

%% Laplace covariance and gradients
function [cov_z, cov_theta, info] = laplace_posterior_covariance_generic(z, g, model, dims, cfg, lambda_g)
    % Laplace covariance of theta = [z; g] around the HB-MM solution. Exact
    % z/g constraints are projected out before inverting the Hessian.
    [Jzz, Jzg, Jgg] = hierarchical_hessian(z, g, model, dims, cfg, lambda_g);
    scale = distribution_laplace_scale(cfg, model.freeZDim);
    Htheta = [Jzz Jzg; Jzg' Jgg];
    Htheta = (Htheta + Htheta')/2;
    Ctheta = blkdiag(model.Cz, model.AeqG);
    [cov_theta, Hred, damping] = constrained_covariance_theta(Htheta, Ctheta, cfg.mm.jitter);
    cov_theta = scale*cov_theta;
    cov_z = cov_theta(1:dims.nz, 1:dims.nz);
    cov_z = (cov_z + cov_z')/2;

    eigHred = eig((Hred + Hred')/2);
    eigJzz = eig((Jzz + Jzz')/2);
    Ng = null(model.AeqG);
    if isempty(Ng)
        eigJgg = NaN;
        eigSchurG = NaN;
    else
        eigJgg = eig((Ng'*Jgg*Ng + Ng'*Jgg'*Ng)/2);
        SchurG = Jgg - Jzg'*(Jzz\Jzg);
        SchurG = (SchurG + SchurG')/2;
        eigSchurG = eig((Ng'*SchurG*Ng + Ng'*SchurG'*Ng)/2);
    end
    info.Htheta = Htheta;
    info.Jzz = Jzz;
    info.Jzg = Jzg;
    info.Jgg = Jgg;
    info.Hred = Hred;
    info.damping = damping;
    info.solved = damping == 0;
    info.minEigHtheta = min(eigHred);
    info.numNegEigHtheta = sum(eigHred < -1e-8);
    info.minEigJzz = min(eigJzz);
    info.minEigJgg = min(eigJgg);
    info.minEigSchurG = min(eigSchurG);
end

function [gradNorm, gradInfNorm, grad] = original_map_gradient_norm_generic(z, g, model, dims, cfg, lambda_g)
    % Report stationarity in the feasible directions only. This avoids
    % counting gradients normal to exact-input constraints as MM error.
    [grad_z, grad_g] = hierarchical_gradient(z, g, model, dims, cfg, lambda_g);
    Nz = null(model.Cz);
    Ng = null(model.AeqG);
    grad = [Nz'*grad_z; Ng'*grad_g];
    gradNorm = norm(grad);
    gradInfNorm = norm(grad, inf);
end

function [Jzz, Jzg, Jgg] = hierarchical_hessian(z, g, model, dims, cfg, lambda_g)
    % Hessian blocks for the Gaussian/elliptical-t hierarchical objective.
    % Distribution dependence enters only through rho'(q) and rho''(q).
    S = make_spd(trajectory_covar(g, dims, cfg), cfg.mm.jitter);
    P = S\eye(dims.nz);
    a = z - model.H*g;
    b = P*a;
    qm = max(a'*b, 0);
    [rho1m, rho2m] = elliptical_rho_derivatives(qm, dims.nz, cfg);

    obs = model.Phi*z - model.zeta;
    R = model.SigmaE\eye(size(model.SigmaE));
    ro = model.Phi'*(R*obs);
    qo = max(obs'*(R*obs), 0);
    [rho1o, rho2o] = elliptical_rho_derivatives(qo, numel(model.zeta), cfg);

    [dS, PS, dSb, PSb] = covariance_derivative_cache(g, P, b, dims, cfg);
    PH = P*model.H;
    HPH = model.H'*PH;
    HPSb = model.H'*PSb;
    dSbPdSb = dSb'*P*dSb;
    qz = 2*b;
    qg = zeros(dims.M, 1);
    for ii = 1:dims.M
        qg(ii) = -2*model.H(:, ii)'*b - b'*dS{ii}*b;
    end

    Jzz = 2*rho1o*(model.Phi'*R*model.Phi) + 4*rho2o*(ro*ro') ...
        + 2*rho1m*P + rho2m*(qz*qz');
    Jzg = 2*rho1m*(-PH - PSb) + rho2m*(qz*qg');
    Jgg = zeros(dims.M, dims.M);
    for ii = 1:dims.M
        for jj = ii:dims.M
            Sij = trajectory_covar_hessian(ii, jj, dims, cfg);
            logdetHess = 0.5*(trace(P*Sij) - trace(PS{jj}*PS{ii}));
            qgg = 2*(HPH(ii, jj) + HPSb(ii, jj) + HPSb(jj, ii) ...
                + dSbPdSb(jj, ii) - 0.5*(b'*Sij*b));
            value = logdetHess + rho1m*qgg + rho2m*qg(ii)*qg(jj);
            if ii == jj
                value = value + 1/lambda_g;
            end
            Jgg(ii, jj) = value;
            Jgg(jj, ii) = value;
        end
    end
    Jzz = (Jzz + Jzz')/2;
    Jgg = (Jgg + Jgg')/2;
end

function [grad_z, grad_g] = hierarchical_gradient(z, g, model, dims, cfg, lambda_g)
    % Gradient of the hierarchical objective before constraint projection.
    S = make_spd(trajectory_covar(g, dims, cfg), cfg.mm.jitter);
    P = S\eye(dims.nz);
    a = z - model.H*g;
    b = P*a;
    qm = max(a'*b, 0);
    rho1m = elliptical_rho_derivatives(qm, dims.nz, cfg);

    obs = model.Phi*z - model.zeta;
    R = model.SigmaE\eye(size(model.SigmaE));
    qo = max(obs'*(R*obs), 0);
    rho1o = elliptical_rho_derivatives(qo, numel(model.zeta), cfg);

    grad_z = 2*rho1o*(model.Phi'*(R*obs)) + 2*rho1m*b;
    grad_g = (1/lambda_g)*g;
    for ii = 1:dims.M
        Si = trajectory_covar_gradient(g, ii, dims, cfg);
        qg_i = -2*model.H(:, ii)'*b - b'*Si*b;
        grad_g(ii) = grad_g(ii) + 0.5*trace(P*Si) + rho1m*qg_i;
    end
end

function Hzz = posterior_z_hessian_elliptical(z, g, model, S, cfg)
    % Local z-only Hessian used for p(z | zeta, g) under elliptical noise.
    n = length(z);
    R = model.SigmaE\eye(size(model.SigmaE));
    obs = model.Phi*z - model.zeta;
    ro = model.Phi'*(R*obs);
    qo = max(obs'*(R*obs), 0);
    [rho1o, rho2o] = elliptical_rho_derivatives(qo, numel(model.zeta), cfg);

    P = make_spd(S, cfg.mm.jitter)\eye(n);
    modelRes = z - model.H*g;
    b = P*modelRes;
    qm = max(modelRes'*b, 0);
    [rho1m, rho2m] = elliptical_rho_derivatives(qm, n, cfg);

    Hzz = 2*rho1o*(model.Phi'*R*model.Phi) + 4*rho2o*(ro*ro') ...
        + 2*rho1m*P + 4*rho2m*(b*b');
    Hzz = (Hzz + Hzz')/2;
end

function [dS, PS, dSb, PSb] = covariance_derivative_cache(g, P, b, dims, cfg)
    % Shared cache for covariance derivative products used repeatedly in
    % gradient/Hessian formulas.
    dS = cell(dims.M, 1);
    PS = cell(dims.M, 1);
    dSb = zeros(dims.nz, dims.M);
    PSb = zeros(dims.nz, dims.M);
    for ii = 1:dims.M
        dS{ii} = trajectory_covar_gradient(g, ii, dims, cfg);
        PS{ii} = P*dS{ii};
        dSb(:, ii) = dS{ii}*b;
        PSb(:, ii) = PS{ii}*b;
    end
end

%% Covariance helpers
function bases = make_covar_bases(cfg, dims)
    % Basis(:,:,lag+1) contains the Toeplitz mask for one autocorrelation
    % lag, allowing CVX to form S(G) linearly from lifted correlations.
    bases.u = toeplitz_covar_basis(effective_var(cfg, 'u_data'), dims.idx_u);
    bases.y = toeplitz_covar_basis(cfg.noise.y_data_var, dims.idx_y);
end

function S = trajectory_covar(g, dims, cfg)
    % Covariance of the stacked trajectory error induced by noisy Hankel
    % data for a fixed coefficient vector g.
    Su = covar_data(g, effective_var(cfg, 'u_data'), dims.idx_u);
    Sy = covar_data(g, cfg.noise.y_data_var, dims.idx_y);
    S = blkdiag(Su, Sy);
    S = (S + S')/2;
end

function S = trajectory_covar_gradient(g, index, dims, cfg)
    Su = covar_data_gradient(g, index, effective_var(cfg, 'u_data'), dims.idx_u);
    Sy = covar_data_gradient(g, index, cfg.noise.y_data_var, dims.idx_y);
    S = blkdiag(Su, Sy);
    S = (S + S')/2;
end

function S = trajectory_covar_hessian(ii, jj, dims, cfg)
    Su = covar_data_hessian(ii, jj, effective_var(cfg, 'u_data'), dims.idx_u);
    Sy = covar_data_hessian(ii, jj, cfg.noise.y_data_var, dims.idx_y);
    S = blkdiag(Su, Sy);
    S = (S + S')/2;
end

function B = cvx_trajectory_covar(G, bases, dims)
    % CVX-compatible version of trajectory_covar where autocorrelations of
    % g are replaced by linear functions of the lifted matrix G.
    Sig_u = cvx_toeplitz_covar_from_G(G, bases.u, dims.idx_u);
    Sig_y = cvx_toeplitz_covar_from_G(G, bases.y, dims.idx_y);
    B = [Sig_u, 0*G(1, 1)*ones(dims.idx_u, dims.idx_y); ...
        0*G(1, 1)*ones(dims.idx_y, dims.idx_u), Sig_y];
end

function S = cvx_toeplitz_covar_from_G(G, basis, covSize)
    if covSize == 0
        S = [];
        return
    end
    S = 0*G(1, 1)*eye(covSize);
    for lag = 0:covSize-1
        if lag <= size(G, 1)-1
            corr = sum(diag(G, lag));
        else
            corr = 0*G(1, 1);
        end
        S = S + basis(:, :, lag+1)*corr;
    end
end

function sigma_g = covar_data(g, var, cov_size)
    if cov_size == 0 || var == 0
        sigma_g = zeros(cov_size, cov_size);
        return
    end
    gcorr = xcorr(g, cov_size-1);
    gcorr = gcorr(cov_size:end);
    sigma_g = var*toeplitz(gcorr);
    sigma_g = (sigma_g + sigma_g')/2;
end

function sigma_i = covar_data_gradient(g, index, var, cov_size)
    if cov_size == 0 || var == 0
        sigma_i = zeros(cov_size, cov_size);
        return
    end
    M = length(g);
    dcorr = zeros(cov_size, 1);
    for lag = 0:cov_size-1
        if index + lag <= M
            dcorr(lag+1) = dcorr(lag+1) + g(index+lag);
        end
        if index - lag >= 1
            dcorr(lag+1) = dcorr(lag+1) + g(index-lag);
        end
    end
    sigma_i = var*toeplitz(dcorr);
    sigma_i = (sigma_i + sigma_i')/2;
end

function sigma_ij = covar_data_hessian(ii, jj, var, cov_size)
    if cov_size == 0 || var == 0
        sigma_ij = zeros(cov_size, cov_size);
        return
    end
    dcorr = zeros(cov_size, 1);
    lag = abs(ii - jj);
    if lag <= cov_size-1
        if lag == 0
            dcorr(lag+1) = 2;
        else
            dcorr(lag+1) = 1;
        end
    end
    sigma_ij = var*toeplitz(dcorr);
    sigma_ij = (sigma_ij + sigma_ij')/2;
end

function basis = toeplitz_covar_basis(var, cov_size)
    idx = (1:cov_size)';
    basis = zeros(cov_size, cov_size, cov_size);
    if var == 0
        return
    end
    for lag = 0:cov_size-1
        basis(:, :, lag+1) = var*(abs(idx - idx') == lag);
    end
end

%% Result handling
function results = build_results(cfg, dims, methodNames, errory, t_calc, ...
        ebMMIter, ebMMHitMaxIter, ebMMFinalRelStep, hbMMIter, ...
        hbMMHitMaxIter, hbMMFinalRelStep, hbRankGap, hbOriginalGradNorm, ...
        hbOriginalGradInfNorm, hbLaplaceDamping, hbLaplaceSolved, ...
        hbLaplaceDamped, hbLaplaceTime, hbMinEigHtheta, hbNumNegEigHtheta, ...
        hbMinEigJzz, hbMinEigJgg, hbMinEigSchurG, hbLaplaceCovTheta, ...
        hbLaplaceCovZ, hbLaplaceCovTarget, hbLaplaceMeanStdTarget, ...
        ebCondCovZ, ebCondCovTarget, ebCondMeanStdTarget, ebMMCondCovZ, ...
        ebMMCondCovTarget, ebMMCondMeanStdTarget, hbLaplaceTarget90Inside, ...
        hbLaplaceTarget90Stat, hbLaplaceTarget90Threshold, ebCondTarget90Inside, ...
        ebCondTarget90Stat, ebCondTarget90Threshold, ebMMCondTarget90Inside, ...
        ebMMCondTarget90Stat, ebMMCondTarget90Threshold, covCompareTraceRatioTarget, ...
        covCompareMedianDiagRatioTarget, covCompareRelFrobTarget)

    % Keep raw arrays and covariance objects in results, while console
    % output below prints only compact final summaries.
    Ne = cfg.experiment.Ne;
    rmseSummary = table(methodNames(:), median(errory, 1)', mean(errory, 1)', ...
        mean(t_calc, 1)', 'VariableNames', ...
        {'Method', 'MedianRMSE', 'MeanRMSE', 'MeanTimeSec'});

    mmSummary = struct();
    mmSummary.ebMeanIter = mean(ebMMIter, 'omitnan');
    mmSummary.ebHitMaxIter = sum(ebMMHitMaxIter);
    mmSummary.ebMedianFinalRelStep = median(ebMMFinalRelStep, 'omitnan');
    mmSummary.hbMeanIter = mean(hbMMIter, 'omitnan');
    mmSummary.hbHitMaxIter = sum(hbMMHitMaxIter);
    mmSummary.hbMedianFinalRelStep = median(hbMMFinalRelStep, 'omitnan');
    mmSummary.hbMeanRankGap = mean(hbRankGap, 'omitnan');
    mmSummary.hbMedianRankGap = median(hbRankGap, 'omitnan');

    laplaceSummary = struct();
    laplaceSummary.meanStdTarget = mean(hbLaplaceMeanStdTarget, 'omitnan');
    laplaceSummary.medianStdTarget = median(hbLaplaceMeanStdTarget, 'omitnan');
    laplaceSummary.meanTimeSec = mean(hbLaplaceTime, 'omitnan');
    laplaceSummary.undampedFraction = mean(hbLaplaceSolved);
    laplaceSummary.dampedCount = sum(hbLaplaceDamped);
    laplaceSummary.meanDamping = mean(hbLaplaceDamping, 'omitnan');
    laplaceSummary.maxDamping = max(hbLaplaceDamping);
    laplaceSummary.coverageTarget90 = mean(hbLaplaceTarget90Inside);
    laplaceSummary.coverageTarget90Count = sum(hbLaplaceTarget90Inside);
    laplaceSummary.medianTargetMahalanobisRatio90 = ...
        median(hbLaplaceTarget90Stat./hbLaplaceTarget90Threshold, 'omitnan');

    covarianceSummary = table( ...
        {'Laplace HB'; 'EB conditioned on fmincon g'; 'EB conditioned on EB-MM g'}, ...
        [mean(hbLaplaceMeanStdTarget, 'omitnan'); ...
            mean(ebCondMeanStdTarget, 'omitnan'); ...
            mean(ebMMCondMeanStdTarget, 'omitnan')], ...
        [mean(hbLaplaceTarget90Inside); mean(ebCondTarget90Inside); mean(ebMMCondTarget90Inside)], ...
        [sum(hbLaplaceTarget90Inside); sum(ebCondTarget90Inside); sum(ebMMCondTarget90Inside)], ...
        'VariableNames', {'Covariance', 'MeanStd', 'Coverage90', 'Coverage90Count'});
    covarianceComparison = table( ...
        {'Laplace/EB-fmincon'; 'Laplace/EB-MM'}, ...
        mean(covCompareTraceRatioTarget, 1, 'omitnan')', ...
        median(covCompareMedianDiagRatioTarget, 1, 'omitnan')', ...
        mean(covCompareRelFrobTarget, 1, 'omitnan')', ...
        'VariableNames', {'Comparison', 'MeanTraceRatio', ...
            'MedianDiagRatio', 'MeanRelFrobDiff'});

    diagnostics = struct();
    diagnostics.mmIterations = table((1:Ne)', ebMMIter, ebMMHitMaxIter, ...
        ebMMFinalRelStep, hbMMIter, hbMMHitMaxIter, hbMMFinalRelStep, ...
        'VariableNames', {'Experiment', 'EBIter', 'EBHitMaxIter', ...
            'EBFinalRelStep', 'HBIter', 'HBHitMaxIter', 'HBFinalRelStep'});
    diagnostics.hessian = table((1:Ne)', hbLaplaceDamping, hbNumNegEigHtheta, ...
        hbMinEigHtheta, hbMinEigJzz, hbMinEigJgg, hbMinEigSchurG, ...
        hbOriginalGradNorm, hbOriginalGradInfNorm, hbRankGap, ...
        'VariableNames', {'Experiment', 'Damping', 'NumNegEigHtheta', ...
            'MinEigHtheta', 'MinEigJzz', 'MinEigJgg', 'MinEigSchurG', ...
            'OriginalGradNorm', 'OriginalGradInfNorm', 'RankGap'});
    diagnostics.coverage = struct( ...
        'hbLaplaceTarget90Inside', hbLaplaceTarget90Inside, ...
        'hbLaplaceTarget90Stat', hbLaplaceTarget90Stat, ...
        'hbLaplaceTarget90Threshold', hbLaplaceTarget90Threshold, ...
        'ebCondTarget90Inside', ebCondTarget90Inside, ...
        'ebCondTarget90Stat', ebCondTarget90Stat, ...
        'ebCondTarget90Threshold', ebCondTarget90Threshold, ...
        'ebMMCondTarget90Inside', ebMMCondTarget90Inside, ...
        'ebMMCondTarget90Stat', ebMMCondTarget90Stat, ...
        'ebMMCondTarget90Threshold', ebMMCondTarget90Threshold);
    diagnostics.covarianceComparison = struct( ...
        'traceRatioTarget', covCompareTraceRatioTarget, ...
        'medianDiagRatioTarget', covCompareMedianDiagRatioTarget, ...
        'relFrobTarget', covCompareRelFrobTarget);

    posterior = struct();
    posterior.hbLaplaceCovTheta = hbLaplaceCovTheta;
    posterior.hbLaplaceCovZ = hbLaplaceCovZ;
    posterior.hbLaplaceCovTarget = hbLaplaceCovTarget;
    posterior.ebCondCovZ = ebCondCovZ;
    posterior.ebCondCovTarget = ebCondCovTarget;
    posterior.ebMMCondCovZ = ebMMCondCovZ;
    posterior.ebMMCondCovTarget = ebMMCondCovTarget;

    settings = struct();
    settings.script = mfilename;
    settings.generatedAt = char(datetime('now'));
    settings.cfg = cfg;
    settings.dims = dims;
    settings.methodNames = methodNames;

    results = struct();
    results.settings = settings;
    results.rmseSummary = rmseSummary;
    results.mmSummary = mmSummary;
    results.laplaceSummary = laplaceSummary;
    results.covarianceSummary = covarianceSummary;
    results.covarianceComparison = covarianceComparison;
    results.diagnostics = diagnostics;
    results.posterior = posterior;
    results.errory = errory;
    results.t_calc = t_calc;
end

function print_results(results)
    cfg = results.settings.cfg;
    Ne = cfg.experiment.Ne;
    fprintf('\n%s MM comparison settings\n', upper(cfg.task));
    fprintf('Noise: %s, inputNoise: %d, seed: %d, Ne: %d\n', ...
        cfg.noise.distribution, cfg.noise.inputNoise, cfg.experiment.seed, Ne);
    fprintf('MM maxIter: %d, tol: %.4g, eta: %.4g\n', ...
        cfg.mm.maxIter, cfg.mm.tol, cfg.mm.eta);
    fprintf('\nRMSE summary\n');
    fprintf('%16s %12s %12s %12s\n', 'Method', 'Median', 'Mean', 'Time [s]');
    for jj = 1:height(results.rmseSummary)
        fprintf('%16s %12.4g %12.4g %12.4g\n', results.rmseSummary.Method{jj}, ...
            results.rmseSummary.MedianRMSE(jj), results.rmseSummary.MeanRMSE(jj), ...
            results.rmseSummary.MeanTimeSec(jj));
    end
    fprintf('\nMM convergence summary\n');
    fprintf('EB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
        results.mmSummary.ebMeanIter, results.mmSummary.ebHitMaxIter, Ne, ...
        results.mmSummary.ebMedianFinalRelStep);
    fprintf('HB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
        results.mmSummary.hbMeanIter, results.mmSummary.hbHitMaxIter, Ne, ...
        results.mmSummary.hbMedianFinalRelStep);
    fprintf('HB-MM median rank gap trace(G)-||g||^2: %.4g\n', ...
        results.mmSummary.hbMedianRankGap);
    fprintf('\nPosterior covariance summary on target output\n');
    fprintf('%28s %12s %12s\n', 'Covariance', 'MeanStd', 'Coverage90');
    for jj = 1:height(results.covarianceSummary)
        fprintf('%28s %12.4g %9.2f (%d/%d)\n', ...
            results.covarianceSummary.Covariance{jj}, ...
            results.covarianceSummary.MeanStd(jj), ...
            results.covarianceSummary.Coverage90(jj), ...
            results.covarianceSummary.Coverage90Count(jj), Ne);
    end
    fprintf('Laplace covariance time [s]: mean %.4g; damped Hessian count: %d/%d\n', ...
        results.laplaceSummary.meanTimeSec, results.laplaceSummary.dampedCount, Ne);
    if cfg.save.enabled
        fprintf('\nSaved results: %s\n', cfg.resultFile);
    end
end

function plot_results(results)
    Ne = results.settings.cfg.experiment.Ne;
    groupIdx = repelem(1:numel(results.settings.methodNames), Ne)';
    figure(10)
    boxplot(results.errory(:), groupIdx, 'Labels', results.settings.methodNames)
    ylabel('Target-output RMSE')
    grid on
    figure(11)
    boxplot(results.t_calc(:), groupIdx, 'Labels', results.settings.methodNames)
    ylabel('Calculation time [s]')
    grid on
end

%% Numerical utilities
function [inside, stat, threshold] = confidence_region_contains(error, covar, level, cfg, jitter)
    covar = make_spd(covar, jitter);
    stat = error'*(covar\error);
    p = length(error);
    if strcmpi(cfg.noise.distribution, 'gaussian')
        threshold = chi2inv(level, p);
    else
        threshold = p*finv(level, p, cfg.noise.dof);
    end
    inside = stat <= threshold;
end

function covar = constrained_covariance_from_hessian(H, C, jitter)
    [covar, ~, ~] = constrained_covariance_theta(H, C, jitter);
end

function [covar, Hred, damping] = constrained_covariance_theta(H, C, jitter)
    H = (H + H')/2;
    N = null(C);
    if isempty(N)
        covar = zeros(size(H));
        Hred = zeros(0);
        damping = 0;
        return
    end
    Hred = N'*H*N;
    Hred = (Hred + Hred')/2;
    [HredDamped, damping] = make_spd_with_damping(Hred, jitter);
    covRed = HredDamped\eye(size(HredDamped));
    covar = N*covRed*N';
    covar = (covar + covar')/2;
end

function [traceRatio, medianDiagRatio, relFrobDiff] = covariance_compare(A, B)
    diagB = max(diag(B), eps);
    traceRatio = trace(A)/max(trace(B), eps);
    medianDiagRatio = median(diag(A)./diagB);
    relFrobDiff = norm(A - B, 'fro')/max(norm(B, 'fro'), eps);
end

function scale = elliptical_t_laplace_scale(dof, dimension)
    scale = (dof + dimension)/dof;
end

function value = mean_std_from_cov(covar)
    value = sqrt(mean(max(diag(covar), 0)));
end

function value = stable_logdet(A)
    [~, U, P] = lu(A);
    du = diag(U);
    c = det(P)*prod(sign(du));
    value = log(c) + sum(log(abs(du)));
end

function x = equality_constrained_quadratic(Q, c, Aeq, beq, jitter)
    n = size(Q, 1);
    KKT = [Q + jitter*eye(n), Aeq'; Aeq, zeros(size(Aeq, 1))];
    rhs = [c; beq];
    sol = KKT\rhs;
    x = sol(1:n);
end

function x = solve_regularized_quadratic(Q, c, jitter)
    Q = make_spd(Q, jitter);
    x = Q\c;
end

function H = GenHankel(X, window)
    [N, d] = size(X);
    numCols = N - window + 1;
    H = zeros(d*window, numCols);
    for ii = 1:numCols
        block = X(ii:ii+window-1, :)';
        H(:, ii) = block(:);
    end
end

function S = make_spd(S, jitter)
    S = full((S + S')/2);
    eyeS = eye(size(S));
    lambdaMin = min(eig(S));
    delta = max(0, -lambdaMin + jitter);
    if delta > 0
        S = S + delta*eyeS;
    end
    [~, p] = chol(S);
    step = max(jitter, eps(norm(S, 'fro')));
    while p ~= 0
        S = S + step*eyeS;
        step = 10*step;
        [~, p] = chol(S);
    end
end

function [S, damping] = make_spd_with_damping(S, jitter)
    S = full((S + S')/2);
    lambdaMin = min(eig(S));
    damping = max(0, -lambdaMin + jitter);
    if damping > 0
        S = S + damping*eye(size(S));
    end
    [~, p] = chol(S);
    step = max(jitter, eps(norm(S, 'fro')));
    while p ~= 0
        S = S + step*eye(size(S));
        damping = damping + step;
        step = 10*step;
        [~, p] = chol(S);
    end
end
