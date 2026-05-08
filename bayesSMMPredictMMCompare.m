% Compare data-driven optimal prediction algorithms with MM variants
% Gaussian uncertainties, input errors, i.i.d. noise
%
% The two MM variants follow the empirical Bayes and hierarchical Bayesian
% semidefinite MM formulations in main.tex for the Gaussian case.
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all;
experimentSeed = 3;
rng(experimentSeed);
resultFile = fullfile(pwd, 'bayesSMMPredictMMCompare_results.mat');

if exist('cvx_begin', 'file') == 0
    cvxDir = fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'cvx');
    if exist(fullfile(cvxDir, 'cvx_startup.m'), 'file')
        run(fullfile(cvxDir, 'cvx_startup.m'));
    end
end

options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'StepTolerance', 1e-14, 'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 1e4, 'Display', 'off');

% The SDP-based MM methods are much more expensive than the original
% fmincon/CVX baselines. Increase Ne to 100 for paper-scale experiments.
Ne = 100;
methodNames = {'EB-fmincon', 'Convex', 'N4SID', 'Proj', ...
    'EmpBayes-MM', 'HierBayes-MM'};
numMethods = numel(methodNames);
errory = zeros(Ne, numMethods);
t_calc = zeros(Ne, numMethods);
hbLaplaceCovTheta = cell(Ne, 1);
hbLaplaceCovZ = cell(Ne, 1);
hbLaplaceCovYf = cell(Ne, 1);
hbLaplaceMeanStdYf = nan(Ne, 1);
hbLaplaceDamping = nan(Ne, 1);
hbLaplaceSolved = false(Ne, 1);
hbLaplaceDamped = false(Ne, 1);
hbLaplaceTime = zeros(Ne, 1);
hbLaplaceZ90Stat = nan(Ne, 1);
hbLaplaceZ90Threshold = nan(Ne, 1);
hbLaplaceZ90Inside = false(Ne, 1);
hbLaplaceYf90Stat = nan(Ne, 1);
hbLaplaceYf90Threshold = nan(Ne, 1);
hbLaplaceYf90Inside = false(Ne, 1);
ebCondCovZ = cell(Ne, 1);
ebCondCovYf = cell(Ne, 1);
ebCondMeanStdYf = nan(Ne, 1);
ebCondYf90Stat = nan(Ne, 1);
ebCondYf90Threshold = nan(Ne, 1);
ebCondYf90Inside = false(Ne, 1);
ebMMCondCovZ = cell(Ne, 1);
ebMMCondCovYf = cell(Ne, 1);
ebMMCondMeanStdYf = nan(Ne, 1);
ebMMCondYf90Stat = nan(Ne, 1);
ebMMCondYf90Threshold = nan(Ne, 1);
ebMMCondYf90Inside = false(Ne, 1);
covCompareTraceRatioYf = nan(Ne, 2);
covCompareMedianDiagRatioYf = nan(Ne, 2);
covCompareRelFrobYf = nan(Ne, 2);
hbRankGap = nan(Ne, 1);
hbOriginalGradNorm = nan(Ne, 1);
hbOriginalGradInfNorm = nan(Ne, 1);
hbMinEigHtheta = nan(Ne, 1);
hbNumNegEigHtheta = nan(Ne, 1);
hbMinEigJzz = nan(Ne, 1);
hbMinEigJgg = nan(Ne, 1);
hbMinEigSchurG = nan(Ne, 1);
ebMMIter = nan(Ne, 1);
ebMMHitMaxIter = false(Ne, 1);
ebMMFinalRelStep = nan(Ne, 1);
hbMMIter = nan(Ne, 1);
hbMMHitMaxIter = false(Ne, 1);
hbMMFinalRelStep = nan(Ne, 1);

%% Parameters
nx = 10;    % States
nu = 1;     % Inputs
ny = 1;     % Outputs
N = 100;    % Data points
L = 40;     % N > (nu+1)(L+nx)-1
M = N-L+1;  % Number of columns
LL = L-nx;
u_data_var = 1e-4;
y_data_var = 1e-4;
u_var = 1e-2;
y_var = 1e-2;

mm.maxIter = 100;
mm.tol = 1e-3;
mm.eta = 1e-4;
mm.jitter = 1e-9;

idx_u = L * nu;
idx_y = L * ny;
idx_yp = nx * ny;
idx_yf = LL * ny;
covBasisU = toeplitz_covar_basis(u_data_var, idx_u);
covBasisYp = toeplitz_covar_basis(y_data_var, idx_yp);
covBasisY = toeplitz_covar_basis(y_data_var, idx_y);

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
    ud_dist = ud + sqrt(u_data_var) * randn(N, nu);

    yd = lsim(trueSys, ud);
    yd_dist = yd + sqrt(y_data_var) * randn(N, ny);

    %% Create ONLINE input/output data
    u = randn(L, nu);
    u_dist = u + sqrt(u_var) * randn(L, nu);
    u_dist = u_dist(:);

    y = lsim(trueSys, u);
    y_dist = y + sqrt(y_var) * randn(L, ny);
    y_dist = y_dist(:);

    %% Auxiliary indices
    idx_u = L * nu;
    idx_y = L * ny;
    idx_yp = nx * ny;
    idx_yf = LL * ny;

    yp_dist = y_dist(1:idx_yp);
    yf = y(idx_yp+1:end);

    %% Hankel matrix of offline trajectories
    Hu = GenHankel(ud_dist, L);
    Hy = GenHankel(yd_dist, L);
    Hyp = Hy(1:idx_yp, :);
    Hyf = Hy(idx_yp+1:end, :);

    %% Common prediction model
    zetam = [u_dist; yp_dist];
    H1 = [Hu; Hyp];
    H = [Hu; Hy];
    sigma_m = diag([u_var*ones(idx_u, 1); y_var*ones(idx_yp, 1)]);

    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, ...
        u_data_var, y_data_var, idx_u, idx_yp);

    g0 = pinv(H1)*zetam;
    yfIdx = idx_u+idx_yp+1:idx_u+idx_y;

    %% Empirical Bayes with direct nonconvex MML
    tic
    g_opt = fmincon(Pr_zetam_given_g, g0, [], [], [], [], [], [], [], options);
    sigmag = blkdiag(covar_data(g_opt, u_data_var, idx_u), ...
        covar_data(g_opt, y_data_var, idx_y));
    zeta_opt = pr_zeta_given_g(zetam, H, sigma_m, sigmag, g_opt, idx_yf, mm.jitter);
    yf_est = zeta_opt(idx_u+idx_yp+1:end);
    t_calc(ii, 1) = toc;
    ebCondCovZ{ii} = empirical_bayes_posterior_covariance(g_opt, sigma_m, ...
        u_data_var, y_data_var, idx_u, idx_yp, idx_y, mm.jitter);
    ebCondCovYf{ii} = ebCondCovZ{ii}(yfIdx, yfIdx);
    ebCondMeanStdYf(ii) = sqrt(mean(max(diag(ebCondCovYf{ii}), 0)));

    %% Convex approximation
    tic
    lambda_1 = 1/(sum(g0.^2)*y_data_var + y_var);
    lambda_2 = 1/(sum(g0.^2)*u_data_var + u_var);
    lambda = nx*y_data_var*lambda_1 + L*u_data_var*lambda_2;
    Q_conv = lambda*eye(M) + lambda_1*(Hyp'*Hyp) + lambda_2*(Hu'*Hu);
    c_conv = lambda_1*(Hyp'*yp_dist) + lambda_2*(Hu'*u_dist);
    g_opt2 = solve_regularized_quadratic(Q_conv, c_conv, mm.jitter);

    sigmag2 = blkdiag(covar_data(g_opt2, u_data_var, idx_u), ...
        covar_data(g_opt2, y_data_var, idx_y));
    zeta_opt2 = pr_zeta_given_g(zetam, H, sigma_m, sigmag2, g_opt2, idx_yf, mm.jitter);
    yf_est2 = zeta_opt2(idx_u+idx_yp+1:end);
    t_calc(ii, 2) = toc;

    %% N4SID
    tic
    sys = n4sid(ud_dist, yd_dist, nx);
    y_est = compare(iddata(y_dist, u_dist), sys);
    yf_est3 = y_est(idx_yp+1:end).OutputData;
    t_calc(ii, 3) = toc;

    %% Projection
    tic
    yf_est4 = Hyf*H1'*((H1*H1')\zetam);
    t_calc(ii, 4) = toc;

    %% Empirical Bayes MM from marginal likelihood
    tic
    [yf_est5, g_hat5, ebInfo] = empirical_bayes_mm_predict(zetam, H1, H, sigma_m, ...
        g0, u_data_var, y_data_var, idx_u, idx_yp, idx_y, idx_yf, ...
        covBasisU, covBasisYp, mm);
    ebMMIter(ii) = ebInfo.iter;
    ebMMHitMaxIter(ii) = ebInfo.hitMaxIter;
    ebMMFinalRelStep(ii) = ebInfo.finalRelStep;
    t_calc(ii, 5) = toc;
    ebMMCondCovZ{ii} = empirical_bayes_posterior_covariance(g_hat5, sigma_m, ...
        u_data_var, y_data_var, idx_u, idx_yp, idx_y, mm.jitter);
    ebMMCondCovYf{ii} = ebMMCondCovZ{ii}(yfIdx, yfIdx);
    ebMMCondMeanStdYf(ii) = sqrt(mean(max(diag(ebMMCondCovYf{ii}), 0)));

    %% Hierarchical Bayes MM from joint MAP
    tic
    hb.lambda_g = max(sum(g0.^2), 1e-6);
    [yf_est6, z_hat6, g_hat6, G_hat6, hbInfo] = hierarchical_bayes_mm_predict(zetam, H, sigma_m, ...
        g0, hb.lambda_g, u_data_var, y_data_var, idx_u, idx_yp, idx_y, ...
        covBasisU, covBasisY, mm);
    hbMMIter(ii) = hbInfo.iter;
    hbMMHitMaxIter(ii) = hbInfo.hitMaxIter;
    hbMMFinalRelStep(ii) = hbInfo.finalRelStep;
    hbRankGap(ii) = trace(G_hat6) - sum(g_hat6.^2);
    [hbOriginalGradNorm(ii), hbOriginalGradInfNorm(ii)] = ...
        original_map_gradient_norm(z_hat6, g_hat6, zetam, H, sigma_m, hb.lambda_g, ...
        u_data_var, y_data_var, idx_u, idx_yp, idx_y, mm.jitter);
    t_calc(ii, 6) = toc;

    tic
    [hbLaplaceCovZ{ii}, hbLaplaceCovTheta{ii}, laplaceInfo] = ...
        laplace_posterior_covariance(z_hat6, g_hat6, H, sigma_m, hb.lambda_g, ...
        u_data_var, y_data_var, idx_u, idx_yp, idx_y, idx_yf, mm.jitter);
    hbLaplaceCovYf{ii} = hbLaplaceCovZ{ii}(yfIdx, yfIdx);
    hbLaplaceMeanStdYf(ii) = sqrt(mean(max(diag(hbLaplaceCovYf{ii}), 0)));
    hbLaplaceDamping(ii) = laplaceInfo.damping;
    hbLaplaceSolved(ii) = laplaceInfo.solved;
    hbLaplaceDamped(ii) = laplaceInfo.damping > 0;
    hbMinEigHtheta(ii) = laplaceInfo.minEigHtheta;
    hbNumNegEigHtheta(ii) = laplaceInfo.numNegEigHtheta;
    hbMinEigJzz(ii) = laplaceInfo.minEigJzz;
    hbMinEigJgg(ii) = laplaceInfo.minEigJgg;
    hbMinEigSchurG(ii) = laplaceInfo.minEigSchurG;
    hbLaplaceTime(ii) = toc;

    z_true = [u(:); y(:)];
    [hbLaplaceZ90Inside(ii), hbLaplaceZ90Stat(ii), hbLaplaceZ90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(z_true - z_hat6, hbLaplaceCovZ{ii}, 0.90, mm.jitter);
    [hbLaplaceYf90Inside(ii), hbLaplaceYf90Stat(ii), hbLaplaceYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(yf - yf_est6, hbLaplaceCovYf{ii}, 0.90, mm.jitter);
    [ebCondYf90Inside(ii), ebCondYf90Stat(ii), ebCondYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(yf - yf_est, ebCondCovYf{ii}, 0.90, mm.jitter);
    [ebMMCondYf90Inside(ii), ebMMCondYf90Stat(ii), ebMMCondYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(yf - yf_est5, ebMMCondCovYf{ii}, 0.90, mm.jitter);
    [covCompareTraceRatioYf(ii, 1), covCompareMedianDiagRatioYf(ii, 1), ...
        covCompareRelFrobYf(ii, 1)] = covariance_compare(hbLaplaceCovYf{ii}, ebCondCovYf{ii});
    [covCompareTraceRatioYf(ii, 2), covCompareMedianDiagRatioYf(ii, 2), ...
        covCompareRelFrobYf(ii, 2)] = covariance_compare(hbLaplaceCovYf{ii}, ebMMCondCovYf{ii});

    if ~ebInfo.solved
        warning('EmpBayes-MM did not fully solve in experiment %d. Last CVX status: %s', ...
            ii, ebInfo.status);
    end
    if ~hbInfo.solved
        warning('HierBayes-MM did not fully solve in experiment %d. Last CVX status: %s', ...
            ii, hbInfo.status);
    end
    %% Error
    errory(ii, 1) = norm(yf_est - yf)/sqrt(idx_yf);
    errory(ii, 2) = norm(yf_est2 - yf)/sqrt(idx_yf);
    errory(ii, 3) = norm(yf_est3 - yf)/sqrt(idx_yf);
    errory(ii, 4) = norm(yf_est4 - yf)/sqrt(idx_yf);
    errory(ii, 5) = norm(yf_est5 - yf)/sqrt(idx_yf);
    errory(ii, 6) = norm(yf_est6 - yf)/sqrt(idx_yf);
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
settings.LL = LL;
settings.u_data_var = u_data_var;
settings.y_data_var = y_data_var;
settings.u_var = u_var;
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
laplaceSummary.meanStdYf = mean(hbLaplaceMeanStdYf, 'omitnan');
laplaceSummary.medianStdYf = median(hbLaplaceMeanStdYf, 'omitnan');
laplaceSummary.meanTimeSec = mean(hbLaplaceTime, 'omitnan');
laplaceSummary.undampedFraction = mean(hbLaplaceSolved);
laplaceSummary.dampedCount = sum(hbLaplaceDamped);
laplaceSummary.meanDamping = mean(hbLaplaceDamping, 'omitnan');
laplaceSummary.maxDamping = max(hbLaplaceDamping);
laplaceSummary.coverageZ90 = mean(hbLaplaceZ90Inside);
laplaceSummary.coverageZ90Count = sum(hbLaplaceZ90Inside);
laplaceSummary.coverageYf90 = mean(hbLaplaceYf90Inside);
laplaceSummary.coverageYf90Count = sum(hbLaplaceYf90Inside);
laplaceSummary.medianZMahalanobisRatio90 = ...
    median(hbLaplaceZ90Stat./hbLaplaceZ90Threshold, 'omitnan');
laplaceSummary.medianYfMahalanobisRatio90 = ...
    median(hbLaplaceYf90Stat./hbLaplaceYf90Threshold, 'omitnan');

covarianceSummary = table( ...
    {'Laplace HB'; 'EB conditioned on fmincon g'; 'EB conditioned on EB-MM g'}, ...
    [mean(hbLaplaceMeanStdYf, 'omitnan'); ...
        mean(ebCondMeanStdYf, 'omitnan'); ...
        mean(ebMMCondMeanStdYf, 'omitnan')], ...
    [mean(hbLaplaceYf90Inside); mean(ebCondYf90Inside); mean(ebMMCondYf90Inside)], ...
    [sum(hbLaplaceYf90Inside); sum(ebCondYf90Inside); sum(ebMMCondYf90Inside)], ...
    'VariableNames', {'Covariance', 'MeanStd', 'Coverage90', 'Coverage90Count'});
covarianceComparison = table( ...
    {'Laplace/EB-fmincon'; 'Laplace/EB-MM'}, ...
    mean(covCompareTraceRatioYf, 1, 'omitnan')', ...
    median(covCompareMedianDiagRatioYf, 1, 'omitnan')', ...
    mean(covCompareRelFrobYf, 1, 'omitnan')', ...
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
    'hbLaplaceZ90Inside', hbLaplaceZ90Inside, ...
    'hbLaplaceZ90Stat', hbLaplaceZ90Stat, ...
    'hbLaplaceZ90Threshold', hbLaplaceZ90Threshold, ...
    'hbLaplaceYf90Inside', hbLaplaceYf90Inside, ...
    'hbLaplaceYf90Stat', hbLaplaceYf90Stat, ...
    'hbLaplaceYf90Threshold', hbLaplaceYf90Threshold, ...
    'ebCondYf90Inside', ebCondYf90Inside, ...
    'ebCondYf90Stat', ebCondYf90Stat, ...
    'ebCondYf90Threshold', ebCondYf90Threshold, ...
    'ebMMCondYf90Inside', ebMMCondYf90Inside, ...
    'ebMMCondYf90Stat', ebMMCondYf90Stat, ...
    'ebMMCondYf90Threshold', ebMMCondYf90Threshold);
diagnostics.covarianceComparison = struct( ...
    'traceRatioYf', covCompareTraceRatioYf, ...
    'medianDiagRatioYf', covCompareMedianDiagRatioYf, ...
    'relFrobYf', covCompareRelFrobYf);

posterior = struct();
posterior.hbLaplaceCovTheta = hbLaplaceCovTheta;
posterior.hbLaplaceCovZ = hbLaplaceCovZ;
posterior.hbLaplaceCovYf = hbLaplaceCovYf;
posterior.ebCondCovZ = ebCondCovZ;
posterior.ebCondCovYf = ebCondCovYf;
posterior.ebMMCondCovZ = ebMMCondCovZ;
posterior.ebMMCondCovYf = ebMMCondCovYf;

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

fprintf('\nPrediction MM comparison settings\n');
fprintf('Seed: %d, Ne: %d, MM maxIter: %d, tol: %.4g, eta: %.4g\n', ...
    experimentSeed, Ne, mm.maxIter, mm.tol, mm.eta);
fprintf('\nPrediction RMSE summary\n');
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

fprintf('\nPosterior covariance summary on future output y_f\n');
fprintf('%28s %12s %12s\n', 'Covariance', 'MeanStd', 'Coverage90');
for jj = 1:height(covarianceSummary)
    fprintf('%28s %12.4g %9.2f (%d/%d)\n', covarianceSummary.Covariance{jj}, ...
        covarianceSummary.MeanStd(jj), covarianceSummary.Coverage90(jj), ...
        covarianceSummary.Coverage90Count(jj), Ne);
end
fprintf('HB full trajectory 90%% coverage: %.2f (%d/%d)\n', ...
    laplaceSummary.coverageZ90, laplaceSummary.coverageZ90Count, Ne);
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
ylabel('Prediction RMSE')
grid on

figure(11)
boxplot(t_calc(:), groupIdx, 'Labels', methodNames)
ylabel('Calculation time [s]')
grid on

figure(12)
boxplot(hbLaplaceMeanStdYf)
ylabel('HierBayes-MM Laplace posterior std of y_f')
grid on

%% FUNCTIONS
function [yf_est, g_opt, info] = empirical_bayes_mm_predict(zetam, H1, H, ...
        sigma_m, g_init, u_data_var, y_data_var, idx_u, idx_yp, idx_y, ...
        idx_yf, covBasisU, covBasisYp, mm)
    M = length(g_init);
    nobs = length(zetam);
    g_prev = g_init;
    info.solved = true;
    info.status = 'Not started';
    info.iter = 0;
    info.obj = nan(mm.maxIter, 1);
    info.finalRelStep = nan;
    info.converged = false;
    info.hitMaxIter = false;

    for iter = 1:mm.maxIter
        A_prev = psi(g_prev, sigma_m, u_data_var, y_data_var, idx_u, idx_yp);
        A_prev = make_spd(A_prev, mm.jitter);

        cvx_begin quiet sdp
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_u(idx_u, 1)
            expression corr_yp(idx_yp, 1)
            expression Sig_u(idx_u, idx_u)
            expression Sig_yp(idx_yp, idx_yp)
            expression A(nobs, nobs)

            for lag = 0:idx_u-1
                if lag <= M-1
                    corr_u(lag+1) = sum(diag(G, lag));
                else
                    corr_u(lag+1) = 0*G(1, 1);
                end
            end
            Sig_u = 0*G(1, 1)*eye(idx_u);
            for lag = 1:idx_u
                Sig_u = Sig_u + covBasisU(:, :, lag)*corr_u(lag);
            end

            for lag = 0:idx_yp-1
                if lag <= M-1
                    corr_yp(lag+1) = sum(diag(G, lag));
                else
                    corr_yp(lag+1) = 0*G(1, 1);
                end
            end
            Sig_yp = 0*G(1, 1)*eye(idx_yp);
            for lag = 1:idx_yp
                Sig_yp = Sig_yp + covBasisYp(:, :, lag)*corr_yp(lag);
            end

            A = sigma_m + 0*G(1, 1)*eye(nobs);
            A(1:idx_u, 1:idx_u) = A(1:idx_u, 1:idx_u) + Sig_u;
            A(idx_u+1:nobs, idx_u+1:nobs) = ...
                A(idx_u+1:nobs, idx_u+1:nobs) + Sig_yp;

            residual = zetam - H1*g;
            minimize(0.5*trace(A_prev\A) + 0.5*t ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                t >= 0;
                [A residual; residual' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        info.status = cvx_status;
        info.obj(iter) = cvx_optval;
        info.iter = iter;

        if ~contains(cvx_status, 'Solved') || any(~isfinite(g))
            info.solved = false;
            break
        end

        g_new = full(g);
        rel_step = norm(g_new - g_prev)/max(1, norm(g_prev));
        info.finalRelStep = rel_step;
        g_prev = g_new;

        if rel_step < mm.tol
            info.converged = true;
            break
        end
    end

    info.hitMaxIter = info.solved && info.iter == mm.maxIter && ~info.converged;
    g_opt = g_prev;
    sigmag = blkdiag(covar_data(g_opt, u_data_var, idx_u), ...
        covar_data(g_opt, y_data_var, idx_y));
    zeta_opt = pr_zeta_given_g(zetam, H, sigma_m, sigmag, g_opt, idx_yf, mm.jitter);
    yf_est = zeta_opt(idx_u+idx_yp+1:end);
end

function [yf_est, z_opt, g_opt, G_opt, info] = hierarchical_bayes_mm_predict(zetam, H, ...
        sigma_m, g_init, lambda_g, u_data_var, y_data_var, idx_u, idx_yp, ...
        idx_y, covBasisU, covBasisY, mm)
    M = length(g_init);
    nobs = length(zetam);
    nz = idx_u + idx_y;
    g_prev = g_init;
    z_opt = H*g_prev;
    G_opt = g_prev*g_prev';
    inv_sigma_m = sigma_m\eye(nobs);
    R_sigma_m = chol(inv_sigma_m);
    info.solved = true;
    info.status = 'Not started';
    info.iter = 0;
    info.obj = nan(mm.maxIter, 1);
    info.rankGap = nan(mm.maxIter, 1);
    info.finalRankGap = trace(G_opt) - sum(g_prev.^2);
    info.finalRelStep = nan;
    info.converged = false;
    info.hitMaxIter = false;

    for iter = 1:mm.maxIter
        B_prev = blkdiag(covar_data(g_prev, u_data_var, idx_u), ...
            covar_data(g_prev, y_data_var, idx_y));
        B_prev = make_spd(B_prev, mm.jitter);

        cvx_begin quiet sdp
            variable z(nz)
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_u(idx_u, 1)
            expression corr_y(idx_y, 1)
            expression Sig_u(idx_u, idx_u)
            expression Sig_y(idx_y, idx_y)
            expression B(nz, nz)
            expression Breg(nz, nz)

            for lag = 0:idx_u-1
                if lag <= M-1
                    corr_u(lag+1) = sum(diag(G, lag));
                else
                    corr_u(lag+1) = 0*G(1, 1);
                end
            end
            Sig_u = 0*G(1, 1)*eye(idx_u);
            for lag = 1:idx_u
                Sig_u = Sig_u + covBasisU(:, :, lag)*corr_u(lag);
            end

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

            B = 0*G(1, 1)*eye(nz);
            B(1:idx_u, 1:idx_u) = Sig_u;
            B(idx_u+1:nz, idx_u+1:nz) = Sig_y;
            Breg = B + mm.jitter*eye(nz);

            obs_res = zetam - z(1:nobs);
            model_res = z - H*g;
            minimize(trace(B_prev\Breg) + sum_square(R_sigma_m*obs_res) ...
                + t + (1/lambda_g)*sum_square(g) ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                t >= 0;
                [Breg model_res; model_res' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        info.status = cvx_status;
        info.obj(iter) = cvx_optval;
        info.iter = iter;

        if ~contains(cvx_status, 'Solved') || any(~isfinite(g)) || any(~isfinite(z))
            info.solved = false;
            break
        end

        g_new = full(g);
        z_new = full(z);
        G_new = full(G);
        info.rankGap(iter) = trace(G_new) - sum(g_new.^2);
        rel_step = norm(g_new - g_prev)/max(1, norm(g_prev));
        info.finalRelStep = rel_step;
        g_prev = g_new;
        z_opt = z_new;
        G_opt = G_new;

        if rel_step < mm.tol
            info.converged = true;
            break
        end
    end

    g_opt = g_prev;
    info.finalRankGap = trace(G_opt) - sum(g_opt.^2);
    info.hitMaxIter = info.solved && info.iter == mm.maxIter && ~info.converged;
    yf_est = z_opt(idx_u+idx_yp+1:end);
end

%% Laplace posterior covariance for the Gaussian hierarchical MAP
function [cov_z, cov_theta, info] = laplace_posterior_covariance(z_hat, g_hat, H, ...
        sigma_m, lambda_g, u_data_var, y_data_var, idx_u, idx_yp, idx_y, ...
        idx_yf, jitter)
    M = length(g_hat);
    nz = idx_u + idx_y;
    nobs = idx_u + idx_yp;

    S = blkdiag(covar_data(g_hat, u_data_var, idx_u), ...
        covar_data(g_hat, y_data_var, idx_y));
    S = make_spd(S, jitter);
    P = S\eye(nz);

    a = z_hat - H*g_hat;
    b = P*a;
    C = P - b*b';

    R = sigma_m\eye(nobs);
    PhiTRPhi = blkdiag(R, zeros(idx_yf, idx_yf));

    dS = cell(M, 1);
    PS = cell(M, 1);
    dSb = zeros(nz, M);
    PSb = zeros(nz, M);
    for ii = 1:M
        dS{ii} = blkdiag(covar_data_gradient(g_hat, ii, u_data_var, idx_u), ...
            covar_data_gradient(g_hat, ii, y_data_var, idx_y));
        PS{ii} = P*dS{ii};
        dSb(:, ii) = dS{ii}*b;
        PSb(:, ii) = PS{ii}*b;
    end

    PH = P*H;
    HPH = H'*PH;
    HPSb = H'*PSb;
    dSbPdSb = dSb'*P*dSb;

    Jzz = PhiTRPhi + P;
    Jzg = -PH - PSb;
    Jgg = zeros(M, M);
    for ii = 1:M
        for jj = ii:M
            Sij = blkdiag(covar_data_hessian(ii, jj, u_data_var, idx_u), ...
                covar_data_hessian(ii, jj, y_data_var, idx_y));
            value = HPH(ii, jj) ...
                + HPSb(ii, jj) ...
                + HPSb(jj, ii) ...
                + dSbPdSb(jj, ii) ...
                - 0.5*trace(PS{jj}*PS{ii}) ...
                + 0.5*trace(C*Sij);
            if ii == jj
                value = value + 1/lambda_g;
            end
            Jgg(ii, jj) = value;
            Jgg(jj, ii) = value;
        end
    end

    Htheta = [Jzz Jzg; Jzg' Jgg];
    Htheta = (Htheta + Htheta')/2;
    eigHtheta = eig(Htheta);
    eigJzz = eig((Jzz + Jzz')/2);
    eigJgg = eig((Jgg + Jgg')/2);
    SchurG = Jgg - Jzg'*(Jzz\Jzg);
    SchurG = (SchurG + SchurG')/2;
    eigSchurG = eig(SchurG);

    [HthetaDamped, thetaDamping] = make_spd_with_damping(Htheta, jitter);
    cov_theta = HthetaDamped\eye(nz + M);
    cov_theta = (cov_theta + cov_theta')/2;
    cov_z = cov_theta(1:nz, 1:nz);
    cov_z = (cov_z + cov_z')/2;

    info.Htheta = Htheta;
    info.Jzz = Jzz;
    info.Jzg = Jzg;
    info.Jgg = Jgg;
    info.HthetaDamped = HthetaDamped;
    info.damping = thetaDamping;
    info.solved = info.damping == 0;
    info.minEigHtheta = min(eigHtheta);
    info.numNegEigHtheta = sum(eigHtheta < -1e-8);
    info.minEigJzz = min(eigJzz);
    info.minEigJgg = min(eigJgg);
    info.minEigSchurG = min(eigSchurG);
end

%% Original Gaussian hierarchical joint MAP gradient norm
function [gradNorm, gradInfNorm, grad] = original_map_gradient_norm(z_hat, g_hat, ...
        zetam, H, sigma_m, lambda_g, u_data_var, y_data_var, idx_u, idx_yp, ...
        idx_y, jitter)
    M = length(g_hat);
    nz = idx_u + idx_y;
    nobs = idx_u + idx_yp;

    S = blkdiag(covar_data(g_hat, u_data_var, idx_u), ...
        covar_data(g_hat, y_data_var, idx_y));
    S = make_spd(S, jitter);
    P = S\eye(nz);

    a = z_hat - H*g_hat;
    b = P*a;
    C = P - b*b';

    e = zetam - z_hat(1:nobs);
    R = sigma_m\eye(nobs);
    grad_z = zeros(nz, 1);
    grad_z(1:nobs) = -R*e;
    grad_z = grad_z + b;

    grad_g = -H'*b + (1/lambda_g)*g_hat;
    for ii = 1:M
        Si = blkdiag(covar_data_gradient(g_hat, ii, u_data_var, idx_u), ...
            covar_data_gradient(g_hat, ii, y_data_var, idx_y));
        grad_g(ii) = grad_g(ii) + 0.5*trace(C*Si);
    end

    grad = [grad_z; grad_g];
    gradNorm = norm(grad);
    gradInfNorm = norm(grad, inf);
end

%% Empirical Bayes posterior covariance conditioned on an estimated g
function cov_z = empirical_bayes_posterior_covariance(g_hat, sigma_m, ...
        u_data_var, y_data_var, idx_u, idx_yp, idx_y, jitter)
    nobs = idx_u + idx_yp;
    nz = idx_u + idx_y;
    sigmag = blkdiag(covar_data(g_hat, u_data_var, idx_u), ...
        covar_data(g_hat, y_data_var, idx_y));
    sigmag = (sigmag + sigmag')/2;
    obsIdx = 1:nobs;
    Psi = sigmag(obsIdx, obsIdx) + sigma_m;
    Psi = make_spd(Psi, jitter);
    cov_z = sigmag - sigmag(:, obsIdx)*(Psi\sigmag(obsIdx, :));
    cov_z = (cov_z + cov_z')/2;
    cov_z = cov_z + jitter*eye(nz);
end

%% Basic covariance comparison metrics A relative to B
function [traceRatio, medianDiagRatio, relFrobDiff] = covariance_compare(A, B)
    diagB = max(diag(B), eps);
    traceRatio = trace(A)/max(trace(B), eps);
    medianDiagRatio = median(diag(A)./diagB);
    relFrobDiff = norm(A - B, 'fro')/max(norm(B, 'fro'), eps);
end

%% Covariance matrix induced by offline data noise
function sigma_g = covar_data(g, var, cov_size)
    gcorr = xcorr(g, cov_size-1);
    gcorr = gcorr(cov_size:end);
    sigma_g = var*toeplitz(gcorr);
    sigma_g = (sigma_g + sigma_g')/2;
end

%% First derivative of covariance matrix with respect to g(index)
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

%% Second derivative of covariance matrix with respect to g(i), g(j)
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

%% MLE probability density function
function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, u_data_var, ...
        y_data_var, idx_u, idx_y)
    Psi = psi(g, sigma_m, u_data_var, y_data_var, idx_u, idx_y);
    Psi = make_spd(Psi, 1e-10);
    res = zetam - H*g;

    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P) * prod(sign(du));
    logdetPsi = log(c) + sum(log(abs(du)));

    MLE = logdetPsi + res'*(Psi\res);
end

%% Gaussian posterior mean / MAP conditioned on g
function MAP = pr_zeta_given_g(zetam, H, Sigma_m, sigmag, g_opt, idx_yf, jitter)
    sigmag = make_spd(sigmag, jitter);
    iSigmag = sigmag\eye(size(sigmag));
    iSigma_m = Sigma_m\eye(size(Sigma_m));

    res1 = iSigmag + blkdiag(iSigma_m, zeros(idx_yf, idx_yf));
    res2 = [Sigma_m\zetam; zeros(idx_yf, 1)] + sigmag\(H*g_opt);
    MAP = res1\res2;
end

%% Nonconvex marginal covariance
function Psi = psi(g, sigma_m, u_data_var, y_data_var, idx_u, idx_y)
    sigmag = blkdiag(covar_data(g, u_data_var, idx_u), ...
        covar_data(g, y_data_var, idx_y));
    Psi = sigmag + sigma_m;
end

function basis = toeplitz_covar_basis(var, cov_size)
    idx = (1:cov_size)';
    basis = zeros(cov_size, cov_size, cov_size);
    for lag = 0:cov_size-1
        basis(:, :, lag+1) = var*(abs(idx - idx') == lag);
    end
end

function x = solve_regularized_quadratic(Q, c, jitter)
    Q = make_spd(Q, jitter);
    x = Q\c;
end

%% Hankel matrix
function H = GenHankel(X, window)
    [N, d] = size(X);
    numCols = N - window + 1;
    H = zeros(d*window, numCols);

    for i = 1:numCols
        block = X(i:i+window-1, :)';
        H(:, i) = block(:);
    end
end

%% Check whether an error vector is inside a Gaussian confidence ellipsoid
function [inside, stat, threshold] = gaussian_ellipsoid_contains(error, covar, level, jitter)
    covar = make_spd(covar, jitter);
    stat = error'*(covar\error);
    threshold = chi2inv(level, length(error));
    inside = stat <= threshold;
end

%% SPD regularization for numerical linear solves
function S = make_spd(S, jitter)
    S = full((S + S')/2);
    eyeS = eye(size(S));
    delta = jitter;
    [~, p] = chol(S);
    while p ~= 0
        S = S + delta*eyeS;
        delta = 10*delta;
        [~, p] = chol(S);
    end
end

%% SPD damping while reporting the added diagonal value
function [S, damping] = make_spd_with_damping(S, jitter)
    S = full((S + S')/2);
    eyeS = eye(size(S));
    lambdaMin = min(eig(S));
    damping = max(0, -lambdaMin + jitter);
    if damping > 0
        S = S + damping*eyeS;
    end

    [~, p] = chol(S);
    if p == 0
        return
    end

    step = max(jitter, eps(norm(S, 'fro')));
    while p ~= 0
        S = S + step*eyeS;
        damping = damping + step;
        [~, p] = chol(S);
        if p ~= 0
            step = 10*step;
        end
    end
end
