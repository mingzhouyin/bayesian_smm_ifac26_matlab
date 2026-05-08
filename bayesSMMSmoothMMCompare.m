% Compare data-driven optimal smoothing algorithms with MM variants
% Elliptical Student-t output noise, no input errors
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all;
experimentSeed = 1;
rng(experimentSeed);
resultFile = fullfile(pwd, 'bayesSMMSmoothMMCompare_results.mat');

if exist('cvx_begin', 'file') == 0
    cvxDir = fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'cvx');
    if exist(fullfile(cvxDir, 'cvx_startup.m'), 'file')
        run(fullfile(cvxDir, 'cvx_startup.m'));
    end
end
if exist('KF_RTS', 'file') == 0 && exist(fullfile(pwd, 'rts'), 'dir')
    addpath(genpath(fullfile(pwd, 'rts')));
end

options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'StepTolerance', 1e-14, 'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 1e4, 'Display', 'off');

Ne = 100;
methodNames = {'EB-fmincon', 'Convex', 'N4SID+KF', 'Proj', ...
    'EmpBayes-MM', 'HierBayes-MM'};
numMethods = numel(methodNames);
errory = zeros(Ne, numMethods);
t_calc = zeros(Ne, numMethods);
ebMMIter = nan(Ne, 1);
ebMMHitMaxIter = false(Ne, 1);
ebMMFinalRelStep = nan(Ne, 1);
hbMMIter = nan(Ne, 1);
hbMMHitMaxIter = false(Ne, 1);
hbMMFinalRelStep = nan(Ne, 1);
hbRankGap = nan(Ne, 1);
hbLaplaceCovTheta = cell(Ne, 1);
hbLaplaceCovY = cell(Ne, 1);
hbLaplaceMeanStdY = nan(Ne, 1);
hbLaplaceDamping = nan(Ne, 1);
hbLaplaceSolved = false(Ne, 1);
hbLaplaceDamped = false(Ne, 1);
hbLaplaceTime = zeros(Ne, 1);
hbLaplaceY90Stat = nan(Ne, 1);
hbLaplaceY90Threshold = nan(Ne, 1);
hbLaplaceY90Inside = false(Ne, 1);
ebCondCovY = cell(Ne, 1);
ebCondMeanStdY = nan(Ne, 1);
ebCondY90Stat = nan(Ne, 1);
ebCondY90Threshold = nan(Ne, 1);
ebCondY90Inside = false(Ne, 1);
ebMMCondCovY = cell(Ne, 1);
ebMMCondMeanStdY = nan(Ne, 1);
ebMMCondY90Stat = nan(Ne, 1);
ebMMCondY90Threshold = nan(Ne, 1);
ebMMCondY90Inside = false(Ne, 1);
covCompareTraceRatioY = nan(Ne, 2);
covCompareMedianDiagRatioY = nan(Ne, 2);
covCompareRelFrobY = nan(Ne, 2);
hbOriginalGradNorm = nan(Ne, 1);
hbOriginalGradInfNorm = nan(Ne, 1);
hbMinEigHtheta = nan(Ne, 1);
hbNumNegEigHtheta = nan(Ne, 1);
hbMinEigJzz = nan(Ne, 1);
hbMinEigJgg = nan(Ne, 1);
hbMinEigSchurG = nan(Ne, 1);

%% Parameters
dof = 10;
nx = 10;    % States
nu = 1;     % Inputs
ny = 1;     % Outputs
N = 100;    % Data points
L = 40;     % N > (nu+1)(L+nx)-1
M = N-L+1;  % Number of columns
y_data_var = 1e-4;
y_var = 1e-2;

mm.maxIter = 100;
mm.tol = 1e-3;
mm.eta = 1e-4;
mm.jitter = 1e-9;

idx_y_full = L*ny;
covBasisY = toeplitz_covar_basis(y_data_var, idx_y_full);

for ii = 1:Ne
    %% Define system
    trueSys = drss(nx, ny, nu);
    trueSys.D = 0;
    while max(abs(pole(trueSys))) > 0.95
        trueSys = drss(nx, ny, nu);
    end
    trueSys = trueSys/norm(trueSys);

    %% Create OFFLINE input/output data
    ud = randn(N, nu);
    ud_dist = ud;
    tScale = sqrt(chi2rnd(dof)/dof);

    yd = lsim(trueSys, ud);
    yd_dist = yd + sqrt(y_data_var)*randn(N, ny)/tScale;

    %% Create ONLINE input/output data
    u = randn(L, nu);
    u_dist = u(:);

    y = lsim(trueSys, u);
    y_dist = y + sqrt(y_var)*randn(L, ny)/tScale;
    y_dist = y_dist(:);

    idx_u = L*nu;
    idx_y = L*ny;

    %% Hankel matrix of offline trajectories
    Hu = GenHankel(ud_dist, L);
    Hy = GenHankel(yd_dist, L);

    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, y_dist, Hy, y_var, ...
        y_data_var, idx_y, dof);

    %% Empirical Bayes with direct nonconvex MML
    tic
    g0 = pinv(Hu)*u_dist;
    g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hu, u_dist, [], [], [], options);
    sigmag = covar_data(g_opt, y_data_var, idx_y);
    Pr_zeta_given_g = @(yy) pr_zeta_given_g(yy, y_dist, Hy, y_var, sigmag, g_opt, dof);
    y_est1 = fmincon(Pr_zeta_given_g, y_dist, [], [], [], [], [], [], [], options);
    [ebCondCovY{ii}, ebCondInfo] = student_t_conditioned_laplace_covariance( ...
        y_est1, y_dist, Hy, y_var, sigmag, g_opt, dof, mm.jitter);
    ebCondMeanStdY(ii) = sqrt(mean(max(diag(ebCondCovY{ii}), 0)));
    t_calc(ii, 1) = toc;

    %% Convex approximation
    tic
    lambda = L*y_data_var;
    Q_conv = lambda*eye(M) + Hy'*Hy;
    c_conv = Hy'*y_dist;
    g_opt2 = equality_constrained_quadratic(Q_conv, c_conv, Hu, u_dist, mm.jitter);

    sigmag2 = covar_data(g_opt2, y_data_var, idx_y);
    Pr_zeta_given_g = @(yy) pr_zeta_given_g(yy, y_dist, Hy, y_var, sigmag2, g_opt2, dof);
    y_est2 = fmincon(Pr_zeta_given_g, y_dist, [], [], [], [], [], [], [], options);
    t_calc(ii, 2) = toc;

    %% N4SID + KF
    tic
    sys = n4sid(ud_dist, yd_dist, nx);
    X_filt = KF_RTS(y_dist', sys.A, sys.C, y_var*sys.K*sys.K', ...
        y_var*ones(ny, 1), B=sys.B, u=u_dist);
    y_est3 = (sys.C*X_filt)';
    t_calc(ii, 3) = toc;

    %% Projection
    tic
    H1 = [Hu; Hy(1:nx, :)];
    Hyf = Hy(nx+1:end, :);
    Hyfhat = Hyf*H1'*((H1*H1')\H1);
    Hyhat = [Hy(1:nx, :); Hyfhat];
    g_proj = equality_constrained_quadratic(Hyhat'*Hyhat, Hyhat'*y_dist, ...
        Hu, u_dist, mm.jitter);
    y_est4 = Hyhat*g_proj;
    t_calc(ii, 4) = toc;

    %% Empirical Bayes MM
    tic
    [y_est5, g_hat5, ebInfo] = empirical_bayes_mm_smooth(y_dist, Hu, Hy, u_dist, ...
        g0, y_var, y_data_var, idx_y, dof, options, covBasisY, mm);
    sigmag5 = covar_data(g_hat5, y_data_var, idx_y);
    [ebMMCondCovY{ii}, ebMMCondInfo] = student_t_conditioned_laplace_covariance( ...
        y_est5, y_dist, Hy, y_var, sigmag5, g_hat5, dof, mm.jitter);
    ebMMCondMeanStdY(ii) = sqrt(mean(max(diag(ebMMCondCovY{ii}), 0)));
    ebMMIter(ii) = ebInfo.iter;
    ebMMHitMaxIter(ii) = ebInfo.hitMaxIter;
    ebMMFinalRelStep(ii) = ebInfo.finalRelStep;
    t_calc(ii, 5) = toc;

    %% Hierarchical Bayes MM
    tic
    lambda_g = max(sum(g0.^2), 1e-6);
    [y_est6, z_hat6, g_hat6, G_hat6, hbInfo] = hierarchical_bayes_mm_smooth(y_dist, Hu, Hy, ...
        u_dist, g0, lambda_g, y_var, y_data_var, idx_y, dof, covBasisY, mm);
    hbMMIter(ii) = hbInfo.iter;
    hbMMHitMaxIter(ii) = hbInfo.hitMaxIter;
    hbMMFinalRelStep(ii) = hbInfo.finalRelStep;
    hbRankGap(ii) = trace(G_hat6) - sum(g_hat6.^2);
    [hbOriginalGradNorm(ii), hbOriginalGradInfNorm(ii)] = ...
        original_map_gradient_norm_smooth(z_hat6, g_hat6, y_dist, Hy, ...
        Hu, lambda_g, y_var, y_data_var, idx_y, dof, mm.jitter);
    t_calc(ii, 6) = toc;

    tic
    [hbLaplaceCovY{ii}, hbLaplaceCovTheta{ii}, laplaceInfo] = ...
        laplace_posterior_covariance_smooth(z_hat6, g_hat6, y_dist, Hy, ...
        Hu, lambda_g, y_var, y_data_var, idx_y, dof, mm.jitter);
    hbLaplaceMeanStdY(ii) = sqrt(mean(max(diag(hbLaplaceCovY{ii}), 0)));
    hbLaplaceDamping(ii) = laplaceInfo.damping;
    hbLaplaceSolved(ii) = laplaceInfo.solved;
    hbLaplaceDamped(ii) = laplaceInfo.damping > 0;
    hbMinEigHtheta(ii) = laplaceInfo.minEigHtheta;
    hbNumNegEigHtheta(ii) = laplaceInfo.numNegEigHtheta;
    hbMinEigJzz(ii) = laplaceInfo.minEigJzz;
    hbMinEigJgg(ii) = laplaceInfo.minEigJgg;
    hbMinEigSchurG(ii) = laplaceInfo.minEigSchurG;
    hbLaplaceTime(ii) = toc;

    [hbLaplaceY90Inside(ii), hbLaplaceY90Stat(ii), hbLaplaceY90Threshold(ii)] = ...
        elliptical_t_confidence_region_contains(y - y_est6, hbLaplaceCovY{ii}, ...
        0.90, dof, mm.jitter);
    [ebCondY90Inside(ii), ebCondY90Stat(ii), ebCondY90Threshold(ii)] = ...
        elliptical_t_confidence_region_contains(y - y_est1, ebCondCovY{ii}, ...
        0.90, dof, mm.jitter);
    [ebMMCondY90Inside(ii), ebMMCondY90Stat(ii), ebMMCondY90Threshold(ii)] = ...
        elliptical_t_confidence_region_contains(y - y_est5, ebMMCondCovY{ii}, ...
        0.90, dof, mm.jitter);
    [covCompareTraceRatioY(ii, 1), covCompareMedianDiagRatioY(ii, 1), ...
        covCompareRelFrobY(ii, 1)] = covariance_compare(hbLaplaceCovY{ii}, ebCondCovY{ii});
    [covCompareTraceRatioY(ii, 2), covCompareMedianDiagRatioY(ii, 2), ...
        covCompareRelFrobY(ii, 2)] = covariance_compare(hbLaplaceCovY{ii}, ebMMCondCovY{ii});

    %% Error
    errory(ii, 1) = norm(y_est1 - y)/sqrt(idx_y);
    errory(ii, 2) = norm(y_est2 - y)/sqrt(idx_y);
    errory(ii, 3) = norm(y_est3 - y)/sqrt(idx_y);
    errory(ii, 4) = norm(y_est4 - y)/sqrt(idx_y);
    errory(ii, 5) = norm(y_est5 - y)/sqrt(idx_y);
    errory(ii, 6) = norm(y_est6 - y)/sqrt(idx_y);
end

%% Results
settings = struct();
settings.script = mfilename;
settings.generatedAt = char(datetime('now'));
settings.resultFile = resultFile;
settings.seed = experimentSeed;
settings.Ne = Ne;
settings.methodNames = methodNames;
settings.nx = nx;
settings.nu = nu;
settings.ny = ny;
settings.N = N;
settings.L = L;
settings.M = M;
settings.dof = dof;
settings.y_data_var = y_data_var;
settings.y_var = y_var;
settings.mm = mm;

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
laplaceSummary.studentScale = elliptical_t_laplace_scale(dof, idx_y);
laplaceSummary.meanStdY = mean(hbLaplaceMeanStdY, 'omitnan');
laplaceSummary.medianStdY = median(hbLaplaceMeanStdY, 'omitnan');
laplaceSummary.meanTimeSec = mean(hbLaplaceTime, 'omitnan');
laplaceSummary.undampedFraction = mean(hbLaplaceSolved);
laplaceSummary.dampedCount = sum(hbLaplaceDamped);
laplaceSummary.meanDamping = mean(hbLaplaceDamping, 'omitnan');
laplaceSummary.maxDamping = max(hbLaplaceDamping);
laplaceSummary.coverage90 = mean(hbLaplaceY90Inside);
laplaceSummary.coverage90Count = sum(hbLaplaceY90Inside);
laplaceSummary.medianMahalanobisRatio90 = ...
    median(hbLaplaceY90Stat./hbLaplaceY90Threshold, 'omitnan');

covarianceSummary = table( ...
    {'Laplace HB'; 'EB conditioned on fmincon g'; 'EB conditioned on EB-MM g'}, ...
    [mean(hbLaplaceMeanStdY, 'omitnan'); ...
        mean(ebCondMeanStdY, 'omitnan'); ...
        mean(ebMMCondMeanStdY, 'omitnan')], ...
    [mean(hbLaplaceY90Inside); mean(ebCondY90Inside); mean(ebMMCondY90Inside)], ...
    [sum(hbLaplaceY90Inside); sum(ebCondY90Inside); sum(ebMMCondY90Inside)], ...
    'VariableNames', {'Covariance', 'MeanStd', 'Coverage90', 'Coverage90Count'});
covarianceComparison = table( ...
    {'Laplace/EB-fmincon'; 'Laplace/EB-MM'}, ...
    mean(covCompareTraceRatioY, 1, 'omitnan')', ...
    median(covCompareMedianDiagRatioY, 1, 'omitnan')', ...
    mean(covCompareRelFrobY, 1, 'omitnan')', ...
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
    'hbLaplaceY90Inside', hbLaplaceY90Inside, ...
    'hbLaplaceY90Stat', hbLaplaceY90Stat, ...
    'hbLaplaceY90Threshold', hbLaplaceY90Threshold, ...
    'ebCondY90Inside', ebCondY90Inside, ...
    'ebCondY90Stat', ebCondY90Stat, ...
    'ebCondY90Threshold', ebCondY90Threshold, ...
    'ebMMCondY90Inside', ebMMCondY90Inside, ...
    'ebMMCondY90Stat', ebMMCondY90Stat, ...
    'ebMMCondY90Threshold', ebMMCondY90Threshold);
diagnostics.covarianceComparison = struct( ...
    'traceRatioY', covCompareTraceRatioY, ...
    'medianDiagRatioY', covCompareMedianDiagRatioY, ...
    'relFrobY', covCompareRelFrobY);

posterior = struct();
posterior.hbLaplaceCovTheta = hbLaplaceCovTheta;
posterior.hbLaplaceCovY = hbLaplaceCovY;
posterior.ebCondCovY = ebCondCovY;
posterior.ebMMCondCovY = ebMMCondCovY;

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

save(resultFile, 'results');

fprintf('\nSmoothing MM comparison settings\n');
fprintf('Seed: %d, Ne: %d, MM maxIter: %d, tol: %.4g, eta: %.4g\n', ...
    experimentSeed, Ne, mm.maxIter, mm.tol, mm.eta);
fprintf('\nSmoothing RMSE summary\n');
fprintf('%16s %12s %12s %12s\n', 'Method', 'Median', 'Mean', 'Time [s]');
for jj = 1:numMethods
    fprintf('%16s %12.4g %12.4g %12.4g\n', methodNames{jj}, ...
        rmseSummary.MedianRMSE(jj), rmseSummary.MeanRMSE(jj), ...
        rmseSummary.MeanTimeSec(jj));
end
fprintf('\nMM convergence summary\n');
fprintf('EB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
    mmSummary.ebMeanIter, mmSummary.ebHitMaxIter, Ne, mmSummary.ebMedianFinalRelStep);
fprintf('HB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
    mmSummary.hbMeanIter, mmSummary.hbHitMaxIter, Ne, mmSummary.hbMedianFinalRelStep);
fprintf('HB-MM median rank gap trace(G)-||g||^2: %.4g\n', mmSummary.hbMedianRankGap);

fprintf('\nPosterior covariance summary on smoothed trajectory y\n');
fprintf('%28s %12s %12s\n', 'Covariance', 'MeanStd', 'Coverage90');
for jj = 1:height(covarianceSummary)
    fprintf('%28s %12.4g %9.2f (%d/%d)\n', covarianceSummary.Covariance{jj}, ...
        covarianceSummary.MeanStd(jj), covarianceSummary.Coverage90(jj), ...
        covarianceSummary.Coverage90Count(jj), Ne);
end
fprintf('Laplace covariance time [s]: mean %.4g; damped Hessian count: %d/%d\n', ...
    laplaceSummary.meanTimeSec, laplaceSummary.dampedCount, Ne);
fprintf('Laplace/EB-fmincon trace ratio: %.4g, median diag ratio: %.4g, rel Frobenius diff: %.4g\n', ...
    covarianceComparison.MeanTraceRatio(1), covarianceComparison.MedianDiagRatio(1), ...
    covarianceComparison.MeanRelFrobDiff(1));
fprintf('Laplace/EB-MM trace ratio: %.4g, median diag ratio: %.4g, rel Frobenius diff: %.4g\n', ...
    covarianceComparison.MeanTraceRatio(2), covarianceComparison.MedianDiagRatio(2), ...
    covarianceComparison.MeanRelFrobDiff(2));
fprintf('\nSaved results: %s\n', resultFile);

figure(10)
groupIdx = repelem(1:numMethods, Ne)';
boxplot(errory(:), groupIdx, 'Labels', methodNames)
ylabel('Smoothing RMSE')
grid on

figure(11)
boxplot(t_calc(:), groupIdx, 'Labels', methodNames)
ylabel('Calculation time [s]')
grid on

%% FUNCTIONS
function [y_est, g_opt, info] = empirical_bayes_mm_smooth(y_dist, Hu, Hy, u_dist, ...
        g_init, y_var, y_data_var, idx_y, dof, options, covBasisY, mm)
    M = length(g_init);
    g_prev = g_init;
    info = init_mm_info(mm);

    for iter = 1:mm.maxIter
        A_prev = make_spd(covar_data(g_prev, y_data_var, idx_y) + y_var*eye(idx_y), mm.jitter);
        res_prev = y_dist - Hy*g_prev;
        t_prev = max(res_prev'*(A_prev\res_prev), 0);
        w = 0.5*(dof + idx_y)/(dof + t_prev);

        cvx_begin quiet sdp
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_y(idx_y, 1)
            expression Sig_y(idx_y, idx_y)
            expression A(idx_y, idx_y)

            for lag = 0:idx_y-1
                if lag <= M-1
                    corr_y(lag+1) = sum(diag(G, lag));
                else
                    corr_y(lag+1) = 0*G(1, 1);
                end
            end
            Sig_y = 0*G(1, 1)*eye(idx_y);
            for lag = 1:idx_y
                Sig_y = Sig_y + covBasisY(:, :, lag)*corr_y(lag);
            end
            A = y_var*eye(idx_y) + 0*G(1, 1)*eye(idx_y) + Sig_y;
            residual = y_dist - Hy*g;

            minimize(0.5*trace(A_prev\A) + w*t ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                Hu*g == u_dist;
                Hu*G == u_dist*transpose(g);
                t >= 0;
                [A residual; residual' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        [g_prev, info, stopNow] = update_mm_info(g, g_prev, cvx_status, cvx_optval, iter, mm, info);
        if stopNow
            break
        end
    end

    g_opt = g_prev;
    sigmag = covar_data(g_opt, y_data_var, idx_y);
    Pr_zeta_given_g = @(yy) pr_zeta_given_g(yy, y_dist, Hy, y_var, sigmag, g_opt, dof);
    y_est = fmincon(Pr_zeta_given_g, y_dist, [], [], [], [], [], [], [], options);
end

function [y_est, z_opt, g_opt, G_opt, info] = hierarchical_bayes_mm_smooth(y_dist, Hu, ...
        Hy, u_dist, g_init, lambda_g, y_var, y_data_var, idx_y, dof, covBasisY, mm)
    M = length(g_init);
    g_prev = g_init;
    z_opt = Hy*g_prev;
    G_opt = g_prev*g_prev';
    info = init_mm_info(mm);
    info.rankGap = nan(mm.maxIter, 1);

    for iter = 1:mm.maxIter
        B_prev = make_spd(covar_data(g_prev, y_data_var, idx_y), mm.jitter);
        obs_prev = z_opt - y_dist;
        model_prev = z_opt - Hy*g_prev;
        obs_t_prev = max(obs_prev'*obs_prev/y_var, 0);
        model_t_prev = max(model_prev'*(B_prev\model_prev), 0);
        w_obs = (dof + idx_y)/(dof + obs_t_prev);
        w_model = (dof + idx_y)/(dof + model_t_prev);

        cvx_begin quiet sdp
            variable z(idx_y)
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_y(idx_y, 1)
            expression B(idx_y, idx_y)
            expression Breg(idx_y, idx_y)

            for lag = 0:idx_y-1
                if lag <= M-1
                    corr_y(lag+1) = sum(diag(G, lag));
                else
                    corr_y(lag+1) = 0*G(1, 1);
                end
            end
            B = 0*G(1, 1)*eye(idx_y);
            for lag = 1:idx_y
                B = B + covBasisY(:, :, lag)*corr_y(lag);
            end
            Breg = B + mm.jitter*eye(idx_y);
            obs_res = z - y_dist;
            model_res = z - Hy*g;

            minimize(trace(B_prev\Breg) + w_obs*sum_square(obs_res)/y_var ...
                + w_model*t + (1/lambda_g)*sum_square(g) ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                Hu*g == u_dist;
                Hu*G == u_dist*transpose(g);
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
        z_opt = full(z);
        G_opt = full(G);
        info.rankGap(iter) = trace(G_opt) - sum(g_new.^2);
        [g_prev, info, stopNow] = update_mm_info(g_new, g_prev, cvx_status, cvx_optval, iter, mm, info);
        if stopNow
            break
        end
    end

    g_opt = g_prev;
    info.finalRankGap = trace(G_opt) - sum(g_opt.^2);
    y_est = z_opt;
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

function sigma_g = covar_data(g, var, cov_size)
    gcorr = xcorr(g, cov_size-1);
    gcorr = gcorr(cov_size:end);
    sigma_g = var*toeplitz(gcorr);
    sigma_g = (sigma_g + sigma_g')/2;
end

function sigma_i = covar_data_gradient(g, index, var, cov_size)
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

function MLE = pr_zetam_given_g(g, y_dist, Hy, y_var, y_data_var, idx_y, dof)
    Psi = make_spd(covar_data(g, y_data_var, idx_y) + y_var*eye(idx_y), 1e-10);
    res = y_dist - Hy*g;
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P)*prod(sign(du));
    logdetPsi = log(c) + sum(log(abs(du)));
    MLE = logdetPsi + (dof + idx_y)*log(1 + res'*(Psi\res)/dof);
end

function MAP = pr_zeta_given_g(yy, y_dist, Hy, y_var, sigmag, g_opt, dof)
    delta1 = yy - y_dist;
    delta2 = yy - Hy*g_opt;
    MAP = log(1 + delta1'*delta1/y_var/dof) ...
        + log(1 + delta2'*(sigmag\delta2)/dof);
end

function [cov_z, cov_theta, info] = laplace_posterior_covariance_smooth(z_hat, ...
        g_hat, y_dist, H, Aeq, lambda_g, y_var, y_data_var, idx_y, dof, jitter)
    [Jzz, Jzg, Jgg] = student_t_hierarchical_hessian(z_hat, g_hat, y_dist, H, ...
        lambda_g, y_var, y_data_var, idx_y, dof, jitter);

    Htheta = [Jzz Jzg; Jzg' Jgg];
    Htheta = (Htheta + Htheta')/2;
    eigJzz = eig((Jzz + Jzz')/2);
    Ng = null(Aeq);
    if isempty(Ng)
        eigJgg = NaN;
        eigSchurG = NaN;
    else
        eigJgg = eig((Ng'*Jgg*Ng + Ng'*Jgg'*Ng)/2);
        SchurG = Jgg - Jzg'*(Jzz\Jzg);
        SchurG = (SchurG + SchurG')/2;
        eigSchurG = eig((Ng'*SchurG*Ng + Ng'*SchurG'*Ng)/2);
    end

    [cov_z, cov_theta, Hred, HredDamped, thetaDamping] = ...
        constrained_laplace_covariance(Htheta, idx_y, Aeq, jitter);
    studentScale = elliptical_t_laplace_scale(dof, idx_y);
    cov_z = studentScale*cov_z;
    cov_theta = studentScale*cov_theta;
    eigHred = eig(Hred);

    info.Htheta = Htheta;
    info.Hred = Hred;
    info.Jzz = Jzz;
    info.Jzg = Jzg;
    info.Jgg = Jgg;
    info.HredDamped = HredDamped;
    info.damping = thetaDamping;
    info.solved = info.damping == 0;
    info.studentScale = studentScale;
    info.minEigHtheta = min(eigHred);
    info.numNegEigHtheta = sum(eigHred < -1e-8);
    info.minEigJzz = min(eigJzz);
    info.minEigJgg = min(eigJgg);
    info.minEigSchurG = min(eigSchurG);
end

function [gradNorm, gradInfNorm, grad] = original_map_gradient_norm_smooth(z_hat, ...
        g_hat, y_dist, H, Aeq, lambda_g, y_var, y_data_var, idx_y, dof, jitter)
    M = length(g_hat);
    S = make_spd(covar_data(g_hat, y_data_var, idx_y), jitter);
    P = S\eye(idx_y);
    a = z_hat - H*g_hat;
    b = P*a;
    qm = max(a'*b, 0);
    wm = 0.5*(dof + idx_y)/(dof + qm);
    obs_res = z_hat - y_dist;
    Ro = eye(idx_y)/y_var;
    qo = max(obs_res'*(Ro*obs_res), 0);
    wo = 0.5*(dof + idx_y)/(dof + qo);

    grad_z = 2*wo*(Ro*obs_res) + 2*wm*b;
    grad_g = (1/lambda_g)*g_hat;
    for ii = 1:M
        Si = covar_data_gradient(g_hat, ii, y_data_var, idx_y);
        qg_i = -2*H(:, ii)'*b - b'*Si*b;
        grad_g(ii) = grad_g(ii) + 0.5*trace(P*Si) + wm*qg_i;
    end

    grad = projected_gradient(grad_z, grad_g, Aeq);
    gradNorm = norm(grad);
    gradInfNorm = norm(grad, inf);
end

function [cov_y, info] = student_t_conditioned_laplace_covariance(y_hat, y_dist, ...
        H, y_var, sigmag, g_hat, dof, jitter)
    n = length(y_hat);
    R1 = eye(n)/y_var;
    e1 = y_hat - y_dist;
    q1 = max(e1'*(R1*e1), 0);
    w1 = 0.5*(dof + n)/(dof + q1);
    rho21 = -0.5*(dof + n)/(dof + q1)^2;
    r1 = R1*e1;

    S = make_spd(sigmag, jitter);
    P = S\eye(n);
    e2 = y_hat - H*g_hat;
    b2 = P*e2;
    q2 = max(e2'*b2, 0);
    w2 = 0.5*(dof + n)/(dof + q2);
    rho22 = -0.5*(dof + n)/(dof + q2)^2;

    Hyy = 2*w1*R1 + 4*rho21*(r1*r1') + 2*w2*P + 4*rho22*(b2*b2');
    Hyy = (Hyy + Hyy')/2;
    eigHyy = eig(Hyy);
    [HyyDamped, damping] = make_spd_with_damping(Hyy, jitter);
    cov_y = HyyDamped\eye(n);
    studentScale = elliptical_t_laplace_scale(dof, n);
    cov_y = studentScale*cov_y;
    cov_y = (cov_y + cov_y')/2;

    info.damping = damping;
    info.solved = damping == 0;
    info.studentScale = studentScale;
    info.minEig = min(eigHyy);
    info.numNegEig = sum(eigHyy < -1e-8);
end

function [Jzz, Jzg, Jgg] = student_t_hierarchical_hessian(z_hat, g_hat, y_dist, ...
        H, lambda_g, y_var, y_data_var, idx_y, dof, jitter)
    M = length(g_hat);
    S = make_spd(covar_data(g_hat, y_data_var, idx_y), jitter);
    P = S\eye(idx_y);
    a = z_hat - H*g_hat;
    b = P*a;
    qm = max(a'*b, 0);
    wm = 0.5*(dof + idx_y)/(dof + qm);
    rho2m = -0.5*(dof + idx_y)/(dof + qm)^2;

    obs_res = z_hat - y_dist;
    Ro = eye(idx_y)/y_var;
    qo = max(obs_res'*(Ro*obs_res), 0);
    wo = 0.5*(dof + idx_y)/(dof + qo);
    rho2o = -0.5*(dof + idx_y)/(dof + qo)^2;
    ro = Ro*obs_res;

    dS = cell(M, 1);
    PS = cell(M, 1);
    dSb = zeros(idx_y, M);
    PSb = zeros(idx_y, M);
    qg = zeros(M, 1);
    for ii = 1:M
        dS{ii} = covar_data_gradient(g_hat, ii, y_data_var, idx_y);
        PS{ii} = P*dS{ii};
        dSb(:, ii) = dS{ii}*b;
        PSb(:, ii) = PS{ii}*b;
        qg(ii) = -2*H(:, ii)'*b - b'*dS{ii}*b;
    end

    PH = P*H;
    HPH = H'*PH;
    HPSb = H'*PSb;
    dSbPdSb = dSb'*P*dSb;
    qz = 2*b;

    Jzz = 2*wo*Ro + 4*rho2o*(ro*ro') + 2*wm*P + rho2m*(qz*qz');
    Jzg = wm*2*(-PH - PSb) + rho2m*(qz*qg');
    Jgg = zeros(M, M);
    for ii = 1:M
        for jj = ii:M
            Sij = covar_data_hessian(ii, jj, y_data_var, idx_y);
            logdetHess = 0.5*(trace(P*Sij) - trace(PS{jj}*PS{ii}));
            qggHalf = HPH(ii, jj) + HPSb(ii, jj) + HPSb(jj, ii) ...
                + dSbPdSb(jj, ii) - 0.5*(b'*Sij*b);
            value = logdetHess + wm*2*qggHalf + rho2m*qg(ii)*qg(jj);
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

function scale = elliptical_t_laplace_scale(dof, dimension)
    scale = (dof + dimension)/dof;
end

function [inside, stat, threshold] = elliptical_t_confidence_region_contains( ...
        error, covar, level, dof, jitter)
    covar = make_spd(covar, jitter);
    stat = error'*(covar\error);
    p = length(error);
    threshold = p*finv(level, p, dof);
    inside = stat <= threshold;
end

function [cov_z, cov_theta, Hred, HredDamped, damping] = ...
        constrained_laplace_covariance(Htheta, nz, Aeq, jitter)
    M = size(Aeq, 2);
    Ng = null(Aeq);
    Ntheta = [eye(nz), zeros(nz, size(Ng, 2)); zeros(M, nz), Ng];
    Hred = Ntheta'*Htheta*Ntheta;
    Hred = (Hred + Hred')/2;
    [HredDamped, damping] = make_spd_with_damping(Hred, jitter);
    cov_red = HredDamped\eye(size(HredDamped));
    cov_red = (cov_red + cov_red')/2;
    cov_theta = Ntheta*cov_red*Ntheta';
    cov_theta = (cov_theta + cov_theta')/2;
    cov_z = cov_theta(1:nz, 1:nz);
    cov_z = (cov_z + cov_z')/2;
    if isempty(Ng) && M > 0
        cov_theta(nz+1:end, nz+1:end) = 0;
    end
end

function grad = projected_gradient(grad_z, grad_g, Aeq)
    Ng = null(Aeq);
    grad = [grad_z; Ng'*grad_g];
end

function [traceRatio, medianDiagRatio, relFrobDiff] = covariance_compare(A, B)
    diagB = max(diag(B), eps);
    traceRatio = trace(A)/max(trace(B), eps);
    medianDiagRatio = median(diag(A)./diagB);
    relFrobDiff = norm(A - B, 'fro')/max(norm(B, 'fro'), eps);
end

function basis = toeplitz_covar_basis(var, cov_size)
    idx = (1:cov_size)';
    basis = zeros(cov_size, cov_size, cov_size);
    for lag = 0:cov_size-1
        basis(:, :, lag+1) = var*(abs(idx - idx') == lag);
    end
end

function x = equality_constrained_quadratic(Q, c, Aeq, beq, jitter)
    n = size(Q, 1);
    KKT = [Q + jitter*eye(n), Aeq'; Aeq, zeros(size(Aeq, 1))];
    rhs = [c; beq];
    sol = KKT\rhs;
    x = sol(1:n);
end

function H = GenHankel(X, window)
    [N, d] = size(X);
    numCols = N - window + 1;
    H = zeros(d*window, numCols);
    for i = 1:numCols
        block = X(i:i+window-1, :)';
        H(:, i) = block(:);
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
end

function [S, damping] = make_spd_with_damping(S, jitter)
    S = full((S + S')/2);
    lambdaMin = min(eig(S));
    damping = max(0, -lambdaMin + jitter);
    if damping > 0
        S = S + damping*eye(size(S));
    end
end
