% Compare data-driven optimal control algorithms with MM variants
% Gaussian uncertainties, no input errors, correlated output noise
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all;
experimentSeed = 2;
rng(experimentSeed);

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

Ne = 10;
methodNames = {'EB-fmincon', 'Convex', 'N4SID', 'Proj', ...
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
hbMMSolved = false(Ne, 1);
hbMMConverged = false(Ne, 1);
hbRankGap = nan(Ne, 1);
hbLaplaceCovTheta = cell(Ne, 1);
hbLaplaceCovZ = cell(Ne, 1);
hbLaplaceCovYf = cell(Ne, 1);
hbLaplaceMeanStdYf = nan(Ne, 1);
hbLaplaceDamping = nan(Ne, 1);
hbLaplaceSolved = false(Ne, 1);
hbLaplaceDamped = false(Ne, 1);
hbLaplaceTime = zeros(Ne, 1);
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
hbOriginalGradNorm = nan(Ne, 1);
hbOriginalGradInfNorm = nan(Ne, 1);
hbMinEigHtheta = nan(Ne, 1);
hbNumNegEigHtheta = nan(Ne, 1);
hbMinEigJzz = nan(Ne, 1);
hbMinEigJgg = nan(Ne, 1);
hbMinEigSchurG = nan(Ne, 1);
oracleControlCost = nan(Ne, 1);
hbMapRefineExitflag = nan(Ne, 1);
hbMapRefineIter = nan(Ne, 1);
hbMapRefineFirstOrderOpt = nan(Ne, 1);
hbMapRefineStep = nan(Ne, 1);
hbMapRefineTime = nan(Ne, 1);

%% Parameters
nx = 10;    % States
nu = 1;     % Inputs
ny = 1;     % Outputs
N = 100;    % Data points
L = 40;     % N > (nu+1)(L+nx)-1
M = N-L+1;  % Number of columns
LL = L-nx;

k = 0.95;
var = 1e-2;

mm.maxIter = 100;
mm.tol = 1e-3;
mm.eta = 1e-4;
mm.jitter = 1e-9;

%% Define system
alpha = 0.4;
beta = 0.3;
A = toeplitz([1-2*alpha-beta; alpha; zeros(8, 1)]);
A(1, 1) = 1-alpha-beta;
A(10, 10) = 1-alpha-beta;
B = [1; zeros(9, 1)];
C = [1 zeros(1, 9)];
D = 0;
trueSys = ss(A, B, C, D, -1);
trueSys = trueSys/norm(trueSys);

for ii = 1:Ne
    fprintf('Experiment %d/%d\n', ii, Ne);

    %% Create OFFLINE input/output data
    ud = randn(N, nu);
    ud_dist = ud;

    yd = lsim(trueSys, ud);
    Lchol = chol(toeplitz(var*k.^(0:N-1)'));
    yd_dist = yd + Lchol'*randn(N, ny);

    %% Create ONLINE input/output data
    u = randn(nx, nu);
    u_dist = u(:);

    [y, ~, x] = lsim(trueSys, u);
    x0 = trueSys.A*x(end, :)' + trueSys.B*u_dist(end);
    Lchol = chol(toeplitz(var*k.^(0:nx-1)'));
    y_dist = y + Lchol'*randn(nx, ny);
    y_dist = y_dist(:);

    %% Reference and weights
    yref = [ones(10, 1); -ones(10, 1); ones(10, 1)];
    gain = 1/freqresp(trueSys, 0);
    uref = gain*[ones(10, 1); -ones(10, 1); ones(10, 1)];
    q = 5;
    r = 0.5;

    %% Auxiliary indices
    idx_up = nx*nu;
    idx_uf = LL*nu;
    idx_yp = nx*ny;
    idx_yf = LL*ny;
    idx_y = idx_yp + idx_yf;

    [freeYTrue, TuTrue] = output_prediction_matrices(trueSys.A, trueSys.B, ...
        trueSys.C, x0, idx_yf);
    u_true_opt = solve_regularized_quadratic(r*eye(idx_uf) + q*(TuTrue'*TuTrue), ...
        r*uref + q*TuTrue'*(yref - freeYTrue), mm.jitter);
    y_true_opt = freeYTrue + TuTrue*u_true_opt;
    oracleControlCost(ii) = r*sum((u_true_opt - uref).^2) ...
        + q*sum((y_true_opt - yref).^2);

    %% Hankel matrix of offline trajectories
    Hu = GenHankel(ud_dist, L);
    Hy = GenHankel(yd_dist, L);
    Hup = Hu(1:nx, :);
    Huf = Hu(nx+1:end, :);
    Hyp = Hy(1:nx, :);
    Hyf = Hy(nx+1:end, :);

    %% Common empirical Bayes control model
    zetam = [uref; y_dist; yref];
    H1 = [Huf; Hy];
    sigma_y_m = blkdiag(toeplitz(var*k.^(0:nx-1)'), eye(idx_yf)/q);
    sigma_m = blkdiag(eye(idx_uf)/r, sigma_y_m);
    covBasisY = correlated_covar_basis(k, var, idx_y, M);
    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, ...
        k, var, idx_uf, idx_y);

    %% Empirical Bayes with direct nonconvex MML
    tic
    g0 = constrained_least_squares([Hu; Hy], [u_dist; zetam], Hup, u_dist, mm.jitter);
    g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hup, u_dist, [], [], [], options);

    sigmag = covar_data(g_opt, k, var, idx_y);
    zeta_opt = pr_zeta_given_g([y_dist; yref], Hy, sigma_y_m, sigmag, g_opt, mm.jitter);
    u_ctr1 = Huf*g_opt;
    yy_ctr1 = zeta_opt(idx_yp+1:end);
    y_ctr1 = lsim(trueSys, u_ctr1, [], x0);
    ebCondCovZ{ii} = empirical_bayes_posterior_covariance_control(g_opt, ...
        sigma_y_m, k, var, idx_y, mm.jitter);
    ebCondCovYf{ii} = ebCondCovZ{ii}(idx_yp+1:end, idx_yp+1:end);
    ebCondMeanStdYf(ii) = sqrt(mean(max(diag(ebCondCovYf{ii}), 0)));
    errory(ii, 1) = r*sum((u_ctr1 - uref).^2) + q*sum((y_ctr1 - yref).^2);
    t_calc(ii, 1) = toc;

    %% Convex approximation
    tic
    lambda_1 = 1/(sum(g0.^2)*var + var);
    lambda_2 = 1/(sum(g0.^2)*var + 1/q);
    lambda = nx*var*lambda_1 + LL*var*lambda_2;
    Q_conv = lambda*eye(M) + lambda_1*(Hyp'*Hyp) ...
        + lambda_2*(Hyf'*Hyf) + r*(Huf'*Huf);
    c_conv = lambda_1*(Hyp'*y_dist) + lambda_2*(Hyf'*yref) ...
        + r*(Huf'*uref);
    g_opt2 = equality_constrained_quadratic(Q_conv, c_conv, Hup, u_dist, mm.jitter);

    u_ctr2 = Huf*g_opt2;
    y_ctr2 = lsim(trueSys, u_ctr2, [], x0);
    errory(ii, 2) = r*sum((u_ctr2 - uref).^2) + q*sum((y_ctr2 - yref).^2);
    t_calc(ii, 2) = toc;

    %% N4SID
    tic
    sys = n4sid(ud_dist, yd_dist, nx);
    X_filt = KF_RTS(y_dist', sys.A, sys.C, sys.NoiseVariance*sys.K*sys.K', ...
        sys.NoiseVariance*ones(ny, 1), B=sys.B, u=u_dist);
    x0hat = sys.A*X_filt(:, end) + sys.B*u_dist(end);
    [freeY3, Tu3] = output_prediction_matrices(sys.A, sys.B, sys.C, x0hat, idx_yf);
    u_ctr3 = solve_regularized_quadratic(r*eye(idx_uf) + q*(Tu3'*Tu3), ...
        r*uref + q*Tu3'*(yref - freeY3), mm.jitter);
    y_ctr3 = lsim(trueSys, u_ctr3, [], x0);
    errory(ii, 3) = r*sum((u_ctr3 - uref).^2) + q*sum((y_ctr3 - yref).^2);
    t_calc(ii, 3) = toc;

    %% Projection
    tic
    Hproj = [Hu; Hy(1:nx, :)];
    Hyfhat = Hyf*Hproj'*((Hproj*Hproj')\Hproj);
    Kproj = Hyfhat*pinv(Hproj);
    futureUIdx = idx_up+1:idx_up+idx_uf;
    knownIdx = [1:idx_up, idx_up+idx_uf+1:idx_up+idx_uf+idx_yp];
    freeY4 = Kproj(:, knownIdx)*[u_dist; y_dist];
    Tu4 = Kproj(:, futureUIdx);
    u_ctr4 = solve_regularized_quadratic(r*eye(idx_uf) + q*(Tu4'*Tu4), ...
        r*uref + q*Tu4'*(yref - freeY4), mm.jitter);
    y_ctr4 = lsim(trueSys, u_ctr4, [], x0);
    errory(ii, 4) = r*sum((u_ctr4 - uref).^2) + q*sum((y_ctr4 - yref).^2);
    t_calc(ii, 4) = toc;

    %% Empirical Bayes MM
    tic
    [g_hat5, ebInfo] = empirical_bayes_mm_control(zetam, H1, Hup, u_dist, g0, ...
        sigma_m, k, var, idx_uf, idx_y, covBasisY, mm);
    sigmag5 = covar_data(g_hat5, k, var, idx_y);
    zeta_opt5 = pr_zeta_given_g([y_dist; yref], Hy, sigma_y_m, sigmag5, g_hat5, mm.jitter);
    u_ctr5 = Huf*g_hat5;
    yy_ctr5 = zeta_opt5(idx_yp+1:end);
    y_ctr5 = lsim(trueSys, u_ctr5, [], x0);
    ebMMCondCovZ{ii} = empirical_bayes_posterior_covariance_control(g_hat5, ...
        sigma_y_m, k, var, idx_y, mm.jitter);
    ebMMCondCovYf{ii} = ebMMCondCovZ{ii}(idx_yp+1:end, idx_yp+1:end);
    ebMMCondMeanStdYf(ii) = sqrt(mean(max(diag(ebMMCondCovYf{ii}), 0)));
    ebMMIter(ii) = ebInfo.iter;
    ebMMHitMaxIter(ii) = ebInfo.hitMaxIter;
    ebMMFinalRelStep(ii) = ebInfo.finalRelStep;
    errory(ii, 5) = r*sum((u_ctr5 - uref).^2) + q*sum((y_ctr5 - yref).^2);
    t_calc(ii, 5) = toc;

    %% Hierarchical Bayes MM
    tic
    lambda_g = max(sum(g0.^2), 1e-6);
    [z_hat6, g_hat6, G_hat6, hbInfo] = hierarchical_bayes_mm_control([y_dist; yref], ...
        Hy, Hup, Huf, u_dist, uref, g0, lambda_g, sigma_y_m, r, k, var, ...
        idx_y, covBasisY, mm);
    u_ctr6 = Huf*g_hat6;
    yy_ctr6 = z_hat6(idx_yp+1:end);
    y_ctr6 = lsim(trueSys, u_ctr6, [], x0);
    hbMMIter(ii) = hbInfo.iter;
    hbMMHitMaxIter(ii) = hbInfo.hitMaxIter;
    hbMMFinalRelStep(ii) = hbInfo.finalRelStep;
    hbMMSolved(ii) = hbInfo.solved;
    hbMMConverged(ii) = hbInfo.converged;
    hbRankGap(ii) = trace(G_hat6) - sum(g_hat6.^2);
    errory(ii, 6) = r*sum((u_ctr6 - uref).^2) + q*sum((y_ctr6 - yref).^2);
    t_calc(ii, 6) = toc;

    refineTic = tic;
    [z_lap6, g_lap6, refineInfo] = refine_control_joint_map(g_hat6, ...
        [y_dist; yref], Hy, Hup, Huf, u_dist, uref, lambda_g, sigma_y_m, ...
        r, k, var, idx_y, mm.jitter, options);
    hbMapRefineTime(ii) = toc(refineTic);
    hbMapRefineExitflag(ii) = refineInfo.exitflag;
    hbMapRefineIter(ii) = refineInfo.iterations;
    hbMapRefineFirstOrderOpt(ii) = refineInfo.firstorderopt;
    hbMapRefineStep(ii) = norm([z_lap6; g_lap6] - [z_hat6; g_hat6]) ...
        / max(1, norm([z_hat6; g_hat6]));

    [hbOriginalGradNorm(ii), hbOriginalGradInfNorm(ii)] = ...
        original_map_gradient_norm_control(z_lap6, g_lap6, [y_dist; yref], ...
        Hy, Hup, Huf, uref, lambda_g, sigma_y_m, r, k, var, idx_y, mm.jitter);

    tic
    [hbLaplaceCovZ{ii}, hbLaplaceCovTheta{ii}, laplaceInfo] = ...
        laplace_posterior_covariance_control(z_lap6, g_lap6, [y_dist; yref], ...
        Hy, Hup, Huf, uref, lambda_g, sigma_y_m, r, k, var, idx_y, mm.jitter);
    hbLaplaceCovYf{ii} = hbLaplaceCovZ{ii}(idx_yp+1:end, idx_yp+1:end);
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

    [hbLaplaceYf90Inside(ii), hbLaplaceYf90Stat(ii), hbLaplaceYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(y_true_opt - z_lap6(idx_yp+1:end), hbLaplaceCovYf{ii}, ...
        0.90, mm.jitter);
    [ebCondYf90Inside(ii), ebCondYf90Stat(ii), ebCondYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(y_true_opt - yy_ctr1, ebCondCovYf{ii}, ...
        0.90, mm.jitter);
    [ebMMCondYf90Inside(ii), ebMMCondYf90Stat(ii), ebMMCondYf90Threshold(ii)] = ...
        gaussian_ellipsoid_contains(y_true_opt - yy_ctr5, ebMMCondCovYf{ii}, ...
        0.90, mm.jitter);
    [covCompareTraceRatioYf(ii, 1), covCompareMedianDiagRatioYf(ii, 1), ...
        covCompareRelFrobYf(ii, 1)] = covariance_compare(hbLaplaceCovYf{ii}, ebCondCovYf{ii});
    [covCompareTraceRatioYf(ii, 2), covCompareMedianDiagRatioYf(ii, 2), ...
        covCompareRelFrobYf(ii, 2)] = covariance_compare(hbLaplaceCovYf{ii}, ebMMCondCovYf{ii});
end

%% Results
erroryPlot = sqrt(errory)/q/sqrt(idx_yf);
fprintf('\nControl MM comparison settings\n');
fprintf('Seed: %d, Ne: %d, MM maxIter: %d, tol: %.4g, eta: %.4g\n', ...
    experimentSeed, Ne, mm.maxIter, mm.tol, mm.eta);
fprintf('\nControl cost-derived RMSE summary\n');
fprintf('True-model oracle feasible cost: median %.4g, mean %.4g\n', ...
    median(oracleControlCost), mean(oracleControlCost));
fprintf('%16s %12s %12s %12s\n', 'Method', 'Median', 'Mean', 'Time [s]');
for jj = 1:numMethods
    fprintf('%16s %12.4g %12.4g %12.4g\n', methodNames{jj}, ...
        median(erroryPlot(:, jj)), mean(erroryPlot(:, jj)), mean(t_calc(:, jj)));
end
fprintf('\nMM diagnostics\n');
fprintf('EB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
    mean(ebMMIter), sum(ebMMHitMaxIter), Ne, median(ebMMFinalRelStep, 'omitnan'));
fprintf('HB-MM mean iterations: %.2f, hit maxIter: %d/%d, median final rel step: %.4g\n', ...
    mean(hbMMIter), sum(hbMMHitMaxIter), Ne, median(hbMMFinalRelStep, 'omitnan'));
fprintf('HB-MM solved/converged: %d/%d solved, %d/%d converged\n', ...
    sum(hbMMSolved), Ne, sum(hbMMConverged), Ne);
fprintf('HB-MM mean rank gap trace(G)-||g||^2: %.4g\n', mean(hbRankGap));
fprintf('HB-MM median rank gap trace(G)-||g||^2: %.4g\n', median(hbRankGap));
fprintf('HB original MAP refine success: %d/%d, median iterations: %.4g, median first-order opt: %.4g\n', ...
    sum(hbMapRefineExitflag > 0), Ne, median(hbMapRefineIter, 'omitnan'), ...
    median(hbMapRefineFirstOrderOpt, 'omitnan'));
fprintf('HB original MAP refine median relative step: %.4g, mean time [s]: %.4g\n', ...
    median(hbMapRefineStep, 'omitnan'), mean(hbMapRefineTime, 'omitnan'));
fprintf('\nHierBayes-MM control-objective pseudo-posterior covariance summary\n');
fprintf('Mean future-output posterior std: %.4g\n', mean(hbLaplaceMeanStdYf));
fprintf('Median future-output posterior std: %.4g\n', median(hbLaplaceMeanStdYf));
fprintf('Mean Laplace covariance time [s]: %.4g\n', mean(hbLaplaceTime));
fprintf('Undamped Hessian fraction: %.2f\n', mean(hbLaplaceSolved));
fprintf('Damped Hessian count: %d/%d\n', sum(hbLaplaceDamped), Ne);
fprintf('Mean Hessian damping: %.4g\n', mean(hbLaplaceDamping));
fprintf('Max Hessian damping: %.4g\n', max(hbLaplaceDamping));
fprintf('Mean original MAP projected gradient norm: %.4g\n', mean(hbOriginalGradNorm));
fprintf('Median original MAP projected gradient norm: %.4g\n', median(hbOriginalGradNorm));
fprintf('Median original MAP projected gradient inf-norm: %.4g\n', median(hbOriginalGradInfNorm));
fprintf('Future output oracle-optimal 90%% ellipsoid coverage: %.2f (%d/%d)\n', ...
    mean(hbLaplaceYf90Inside), sum(hbLaplaceYf90Inside), Ne);
fprintf('Median future output Mahalanobis/threshold: %.4g\n', ...
    median(hbLaplaceYf90Stat./hbLaplaceYf90Threshold));

fprintf('\nControl-objective pseudo-posterior covariance comparison on future output y_f\n');
fprintf('Coverage reference: true-model quadratic-cost optimal feasible trajectory\n');
fprintf('%28s %12s %12s\n', 'Covariance', 'MeanStd', 'OracleHit90');
fprintf('%28s %12.4g %9.2f (%d/%d)\n', 'Laplace HB', ...
    mean(hbLaplaceMeanStdYf), mean(hbLaplaceYf90Inside), sum(hbLaplaceYf90Inside), Ne);
fprintf('%28s %12.4g %9.2f (%d/%d)\n', 'EB conditioned on fmincon g', ...
    mean(ebCondMeanStdYf), mean(ebCondYf90Inside), sum(ebCondYf90Inside), Ne);
fprintf('%28s %12.4g %9.2f (%d/%d)\n', 'EB conditioned on EB-MM g', ...
    mean(ebMMCondMeanStdYf), mean(ebMMCondYf90Inside), sum(ebMMCondYf90Inside), Ne);
fprintf('Laplace/EB-fmincon trace ratio: %.4g, median diag ratio: %.4g, rel Frobenius diff: %.4g\n', ...
    mean(covCompareTraceRatioYf(:, 1), 'omitnan'), ...
    median(covCompareMedianDiagRatioYf(:, 1), 'omitnan'), ...
    mean(covCompareRelFrobYf(:, 1), 'omitnan'));
fprintf('Laplace/EB-MM trace ratio: %.4g, median diag ratio: %.4g, rel Frobenius diff: %.4g\n', ...
    mean(covCompareTraceRatioYf(:, 2), 'omitnan'), ...
    median(covCompareMedianDiagRatioYf(:, 2), 'omitnan'), ...
    mean(covCompareRelFrobYf(:, 2), 'omitnan'));

fprintf('\nLaplace Hessian diagnostics by experiment\n');
fprintf('%4s %10s %7s %11s %11s %11s %11s %10s %10s\n', ...
    'Exp', 'Damping', 'NegEig', 'minEig(H)', 'minEig(Jzz)', ...
    'minEig(Jgg)', 'minEig(Sg)', 'ProjGrad', 'RankGap');
for jj = 1:Ne
    fprintf('%4d %10.4g %7d %11.4g %11.4g %11.4g %11.4g %10.4g %10.4g\n', ...
        jj, hbLaplaceDamping(jj), hbNumNegEigHtheta(jj), hbMinEigHtheta(jj), ...
        hbMinEigJzz(jj), hbMinEigJgg(jj), hbMinEigSchurG(jj), ...
        hbOriginalGradNorm(jj), hbRankGap(jj));
end

figure(10)
groupIdx = repelem(1:numMethods, Ne)';
boxplot(erroryPlot(:), groupIdx, 'Labels', methodNames)
ylabel('Control cost-derived RMSE')
grid on

figure(11)
boxplot(t_calc(:), groupIdx, 'Labels', methodNames)
ylabel('Calculation time [s]')
grid on

%% FUNCTIONS
function [g_opt, info] = empirical_bayes_mm_control(zetam, H1, Hup, u_dist, ...
        g_init, sigma_m, k, var, idx_uf, idx_y, covBasisY, mm)
    M = length(g_init);
    nobs = length(zetam);
    g_prev = g_init;
    info = init_mm_info(mm);

    for iter = 1:mm.maxIter
        A_prev = psi(g_prev, sigma_m, k, var, idx_uf, idx_y);
        A_prev = make_spd(A_prev, mm.jitter);

        cvx_begin quiet sdp
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_y(2*M-1, 1)
            expression Sig_y(idx_y, idx_y)
            expression A(nobs, nobs)

            for tau = -(M-1):(M-1)
                corr_y(tau+M) = sum(diag(G, abs(tau)));
            end
            Sig_y = 0*G(1, 1)*eye(idx_y);
            for tt = 1:2*M-1
                Sig_y = Sig_y + covBasisY(:, :, tt)*corr_y(tt);
            end

            A = sigma_m + 0*G(1, 1)*eye(nobs);
            A(idx_uf+1:nobs, idx_uf+1:nobs) = ...
                A(idx_uf+1:nobs, idx_uf+1:nobs) + Sig_y;
            residual = zetam - H1*g;

            minimize(0.5*trace(A_prev\A) + 0.5*t ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                Hup*g == u_dist;
                Hup*G == u_dist*transpose(g);
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
end

function [z_opt, g_opt, G_opt, info] = hierarchical_bayes_mm_control(zetay, Hy, Hup, ...
        Huf, u_dist, uref, g_init, lambda_g, sigma_y_m, r, k, var, idx_y, covBasisY, mm)
    M = length(g_init);
    g_prev = g_init;
    z_opt = Hy*g_prev;
    G_opt = g_prev*g_prev';
    R_sigma_y_m = chol(sigma_y_m\eye(length(zetay)));
    info = init_mm_info(mm);
    info.rankGap = nan(mm.maxIter, 1);

    for iter = 1:mm.maxIter
        z_prev = z_opt;
        B_prev = make_spd(covar_data(g_prev, k, var, idx_y), mm.jitter);

        cvx_begin quiet sdp
            variable z(idx_y)
            variable g(M)
            variable G(M, M) symmetric
            variable t
            expression corr_y(2*M-1, 1)
            expression B(idx_y, idx_y)
            expression Breg(idx_y, idx_y)

            for tau = -(M-1):(M-1)
                corr_y(tau+M) = sum(diag(G, abs(tau)));
            end
            B = 0*G(1, 1)*eye(idx_y);
            for tt = 1:2*M-1
                B = B + covBasisY(:, :, tt)*corr_y(tt);
            end

            Breg = B + mm.jitter*eye(idx_y);
            obs_res = zetay - z;
            model_res = z - Hy*g;

            minimize(trace(B_prev\Breg) + sum_square(R_sigma_y_m*obs_res) ...
                + t + r*sum_square(Huf*g - uref) ...
                + (1/lambda_g)*sum_square(g) ...
                + mm.eta*trace(G) - 2*mm.eta*transpose(g_prev)*g)
            subject to
                Hup*g == u_dist;
                Hup*G == u_dist*transpose(g);
                t >= 0;
                [Breg model_res; model_res' t] >= 0;
                [G g; g' 1] >= 0;
        cvx_end

        if ~contains(cvx_status, 'Solved') || any(~isfinite(g)) || any(~isfinite(z))
            info.solved = false;
            info.status = cvx_status;
            info.iter = iter;
            info.breakReason = 'solver_failed_or_nonfinite';
            break
        end

        g_new = full(g);
        z_new = full(z);
        z_opt = z_new;
        G_opt = full(G);
        info.rankGap(iter) = trace(G_opt) - sum(g_new.^2);
        [g_prev, info, stopNow] = update_mm_info_state(g_new, z_new, ...
            g_prev, z_prev, cvx_status, cvx_optval, iter, mm, info);
        if stopNow
            break
        end
    end

    g_opt = g_prev;
    info.finalRankGap = trace(G_opt) - sum(g_opt.^2);
end

function info = init_mm_info(mm)
    info.solved = true;
    info.status = 'Not started';
    info.iter = 0;
    info.obj = nan(mm.maxIter, 1);
    info.finalRelStep = nan;
    info.converged = false;
    info.hitMaxIter = false;
    info.breakReason = 'not_started';
end

function [g_new, info, stopNow] = update_mm_info(g_value, g_prev, status, optval, iter, mm, info)
    info.status = status;
    info.obj(iter) = optval;
    info.iter = iter;
    stopNow = false;

    if ~contains(status, 'Solved') || any(~isfinite(g_value))
        info.solved = false;
        info.breakReason = 'solver_failed_or_nonfinite';
        stopNow = true;
        g_new = g_prev;
        return
    end

    g_new = full(g_value);
    info.finalRelStep = norm(g_new - g_prev)/max(1, norm(g_prev));
    if info.finalRelStep < mm.tol
        info.converged = true;
        info.breakReason = 'step_tolerance';
        stopNow = true;
    end
    info.hitMaxIter = info.solved && iter == mm.maxIter && ~info.converged;
    if info.hitMaxIter
        info.breakReason = 'max_iter';
    elseif ~stopNow
        info.breakReason = 'continue';
    end
end

function [g_new, info, stopNow] = update_mm_info_state(g_value, z_value, ...
        g_prev, z_prev, status, optval, iter, mm, info)
    info.status = status;
    info.obj(iter) = optval;
    info.iter = iter;
    stopNow = false;

    if ~contains(status, 'Solved') || any(~isfinite(g_value)) || any(~isfinite(z_value))
        info.solved = false;
        info.breakReason = 'solver_failed_or_nonfinite';
        stopNow = true;
        g_new = g_prev;
        return
    end

    g_new = full(g_value);
    z_new = full(z_value);
    thetaStep = norm([z_new; g_new] - [z_prev; g_prev]);
    thetaNorm = max(1, norm([z_prev; g_prev]));
    info.finalRelStep = thetaStep/thetaNorm;
    if info.finalRelStep < mm.tol
        info.converged = true;
        info.breakReason = 'step_tolerance';
        stopNow = true;
    end
    info.hitMaxIter = info.solved && iter == mm.maxIter && ~info.converged;
    if info.hitMaxIter
        info.breakReason = 'max_iter';
    elseif ~stopNow
        info.breakReason = 'continue';
    end
end

function basis = correlated_covar_basis(k, var, cov_size, M)
    basis = zeros(cov_size, cov_size, 2*M-1);
    for tau = -(M-1):(M-1)
        for row = 1:cov_size
            for col = 1:cov_size
                basis(row, col, tau+M) = var*k^abs(tau + row - col);
            end
        end
    end
end

function sigma_g = covar_data(g, k, var, cov_size)
    M = length(g);
    gcorr = xcorr(g, M-1);
    sigma_g = zeros(cov_size, cov_size);
    varseq = var*k.^abs(2-M-cov_size:M+cov_size-2);
    for row = 1:cov_size
        for col = 1:cov_size
            sigma_g(row, col) = varseq(row-col+cov_size:row-col+cov_size+2*M-2)*gcorr;
        end
    end
    sigma_g = (sigma_g + sigma_g')/2;
end

function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, k, var, idx_u, idx_y)
    Psi = make_spd(psi(g, sigma_m, k, var, idx_u, idx_y), 1e-10);
    res = zetam - H*g;
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P)*prod(sign(du));
    logdetPsi = log(c) + sum(log(abs(du)));
    MLE = logdetPsi + res'*(Psi\res);
end

function MAP = pr_zeta_given_g(zetam, H, Sigma_m, sigmag, g_opt, jitter)
    sigmag = make_spd(sigmag, jitter);
    iSigmag = sigmag\eye(size(sigmag));
    iSigma_m = Sigma_m\eye(size(Sigma_m));
    res1 = iSigmag + iSigma_m;
    res2 = Sigma_m\zetam + sigmag\(H*g_opt);
    MAP = res1\res2;
end

function Psi = psi(g, sigma_m, k, var, idx_u, idx_y)
    sigmag = blkdiag(zeros(idx_u, idx_u), covar_data(g, k, var, idx_y));
    Psi = sigmag + sigma_m;
end

function [cov_z, cov_theta, info] = laplace_posterior_covariance_control(z_hat, ...
        g_hat, zetay, H, Aeq, Huf, uref, lambda_g, sigma_y_m, r, k, var, idx_y, jitter)
    [Jzz, Jzg, Jgg] = gaussian_hierarchical_hessian_control(z_hat, g_hat, ...
        H, Huf, lambda_g, sigma_y_m, r, k, var, idx_y, jitter);

    Htheta = [Jzz Jzg; Jzg' Jgg];
    Htheta = (Htheta + Htheta')/2;
    eigJzz = eig((Jzz + Jzz')/2);
    Ng = null(Aeq);
    if isempty(Ng)
        eigJgg = NaN;
        eigSchurG = NaN;
    else
        JggRed = Ng'*Jgg*Ng;
        eigJgg = eig((JggRed + JggRed')/2);
        SchurG = Jgg - Jzg'*(Jzz\Jzg);
        SchurG = (SchurG + SchurG')/2;
        SchurGRed = Ng'*SchurG*Ng;
        eigSchurG = eig((SchurGRed + SchurGRed')/2);
    end

    [cov_z, cov_theta, Hred, HredDamped, thetaDamping] = ...
        constrained_laplace_covariance(Htheta, idx_y, Aeq, jitter);
    eigHred = eig(Hred);

    info.Htheta = Htheta;
    info.Hred = Hred;
    info.Jzz = Jzz;
    info.Jzg = Jzg;
    info.Jgg = Jgg;
    info.HredDamped = HredDamped;
    info.damping = thetaDamping;
    info.solved = info.damping == 0;
    info.minEigHtheta = min(eigHred);
    info.numNegEigHtheta = sum(eigHred < -1e-8);
    info.minEigJzz = min(eigJzz);
    info.minEigJgg = min(eigJgg);
    info.minEigSchurG = min(eigSchurG);
end

function [gradNorm, gradInfNorm, grad] = original_map_gradient_norm_control(z_hat, ...
        g_hat, zetay, H, Aeq, Huf, uref, lambda_g, sigma_y_m, r, k, var, idx_y, jitter)
    M = length(g_hat);
    S = make_spd(covar_data(g_hat, k, var, idx_y), jitter);
    P = S\eye(idx_y);
    a = z_hat - H*g_hat;
    b = P*a;
    C = P - b*b';
    R = sigma_y_m\eye(idx_y);
    obs_res = z_hat - zetay;

    grad_z = R*obs_res + b;
    grad_g = -H'*b + r*Huf'*(Huf*g_hat - uref) + (1/lambda_g)*g_hat;
    for ii = 1:M
        Si = covar_data_gradient_corr(g_hat, ii, k, var, idx_y);
        grad_g(ii) = grad_g(ii) + 0.5*trace(C*Si);
    end

    grad = projected_gradient(grad_z, grad_g, Aeq);
    gradNorm = norm(grad);
    gradInfNorm = norm(grad, inf);
end

function [z_ref, g_ref, info] = refine_control_joint_map(g_init, zetay, ...
        H, Aeq, Huf, u_dist, uref, lambda_g, sigma_y_m, r, k, var, idx_y, ...
        jitter, options)
    obj = @(g) control_profiled_map_objective(g, zetay, H, Huf, uref, ...
        lambda_g, sigma_y_m, r, k, var, idx_y, jitter);
    mapOptions = optimoptions(options, 'SpecifyObjectiveGradient', true, ...
        'Display', 'off');

    info.exitflag = -999;
    info.iterations = NaN;
    info.firstorderopt = NaN;
    g_ref = g_init;
    f0 = obj(g_init);
    try
        [g_try, f_try, exitflag, output] = fmincon(obj, g_init, [], [], ...
            Aeq, u_dist, [], [], [], mapOptions);
        if all(isfinite(g_try)) && isfinite(f_try) && f_try <= f0*(1 + 1e-8)
            g_ref = g_try;
        end
        info.exitflag = exitflag;
        if isfield(output, 'iterations')
            info.iterations = output.iterations;
        end
        if isfield(output, 'firstorderopt')
            info.firstorderopt = output.firstorderopt;
        end
    catch
        g_ref = g_init;
    end

    z_ref = control_conditional_map_z(g_ref, zetay, H, sigma_y_m, k, var, ...
        idx_y, jitter);
end

function [f, grad] = control_profiled_map_objective(g, zetay, H, Huf, uref, ...
        lambda_g, sigma_y_m, r, k, var, idx_y, jitter)
    M = length(g);
    z = control_conditional_map_z(g, zetay, H, sigma_y_m, k, var, idx_y, jitter);
    S = make_spd(covar_data(g, k, var, idx_y), jitter);
    P = S\eye(idx_y);
    a = z - H*g;
    b = P*a;
    obs_res = z - zetay;
    R = sigma_y_m\eye(idx_y);
    u_res = Huf*g - uref;

    f = 0.5*logdet_spd(S) + 0.5*obs_res'*(R*obs_res) ...
        + 0.5*a'*b + 0.5*r*sum(u_res.^2) ...
        + 0.5*(1/lambda_g)*sum(g.^2);

    if nargout > 1
        C = P - b*b';
        grad_g = -H'*b + r*Huf'*u_res + (1/lambda_g)*g;
        for ii = 1:M
            Si = covar_data_gradient_corr(g, ii, k, var, idx_y);
            grad_g(ii) = grad_g(ii) + 0.5*trace(C*Si);
        end
        grad = grad_g;
    end
end

function z = control_conditional_map_z(g, zetay, H, sigma_y_m, k, var, idx_y, jitter)
    S = make_spd(covar_data(g, k, var, idx_y), jitter);
    P = S\eye(idx_y);
    R = sigma_y_m\eye(idx_y);
    z = make_spd(R + P, jitter)\(R*zetay + P*(H*g));
end

function cov_z = empirical_bayes_posterior_covariance_control(g_hat, sigma_y_m, ...
        k, var, idx_y, jitter)
    sigmag = covar_data(g_hat, k, var, idx_y);
    sigmag = make_spd(sigmag, jitter);
    Psi = make_spd(sigmag + sigma_y_m, jitter);
    cov_z = sigmag - sigmag*(Psi\sigmag);
    cov_z = (cov_z + cov_z')/2;
    cov_z = cov_z + jitter*eye(idx_y);
end

function [Jzz, Jzg, Jgg] = gaussian_hierarchical_hessian_control(z_hat, g_hat, ...
        H, Huf, lambda_g, sigma_y_m, r, k, var, idx_y, jitter)
    M = length(g_hat);
    S = make_spd(covar_data(g_hat, k, var, idx_y), jitter);
    P = S\eye(idx_y);
    a = z_hat - H*g_hat;
    b = P*a;
    C = P - b*b';
    R = sigma_y_m\eye(idx_y);

    dS = cell(M, 1);
    PS = cell(M, 1);
    dSb = zeros(idx_y, M);
    PSb = zeros(idx_y, M);
    for ii = 1:M
        dS{ii} = covar_data_gradient_corr(g_hat, ii, k, var, idx_y);
        PS{ii} = P*dS{ii};
        dSb(:, ii) = dS{ii}*b;
        PSb(:, ii) = PS{ii}*b;
    end

    PH = P*H;
    HPH = H'*PH;
    HPSb = H'*PSb;
    dSbPdSb = dSb'*P*dSb;

    Jzz = R + P;
    Jzg = -PH - PSb;
    Jgg = r*(Huf'*Huf) + (1/lambda_g)*eye(M);
    for ii = 1:M
        for jj = ii:M
            Sij = covar_data_hessian_corr(ii, jj, k, var, idx_y);
            value = HPH(ii, jj) ...
                + HPSb(ii, jj) ...
                + HPSb(jj, ii) ...
                + dSbPdSb(jj, ii) ...
                - 0.5*trace(PS{jj}*PS{ii}) ...
                + 0.5*trace(C*Sij);
            Jgg(ii, jj) = Jgg(ii, jj) + value;
            Jgg(jj, ii) = Jgg(ii, jj);
        end
    end

    Jzz = (Jzz + Jzz')/2;
    Jgg = (Jgg + Jgg')/2;
end

function sigma_i = covar_data_gradient_corr(g, index, k, var, cov_size)
    M = length(g);
    sigma_i = zeros(cov_size, cov_size);
    for row = 1:cov_size
        for col = 1:cov_size
            offset = row - col;
            value = 0;
            for jj = 1:M
                value = value + g(jj)*var*k^abs(index - jj + offset);
                value = value + g(jj)*var*k^abs(jj - index + offset);
            end
            sigma_i(row, col) = value;
        end
    end
    sigma_i = (sigma_i + sigma_i')/2;
end

function sigma_ij = covar_data_hessian_corr(ii, jj, k, var, cov_size)
    sigma_ij = zeros(cov_size, cov_size);
    for row = 1:cov_size
        for col = 1:cov_size
            offset = row - col;
            sigma_ij(row, col) = var*k^abs(ii - jj + offset) ...
                + var*k^abs(jj - ii + offset);
        end
    end
    sigma_ij = (sigma_ij + sigma_ij')/2;
end

function [inside, stat, threshold] = gaussian_ellipsoid_contains(error, covar, level, jitter)
    covar = make_spd(covar, jitter);
    stat = error'*(covar\error);
    threshold = chi2inv(level, length(error));
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
    cov_red = spd_inverse(HredDamped, jitter);
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

function g = constrained_least_squares(A, b, C, d, jitter)
    M = size(A, 2);
    KKT = [A'*A + jitter*eye(M), C'; C, zeros(size(C, 1))];
    rhs = [A'*b; d];
    sol = KKT\rhs;
    g = sol(1:M);
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

function Sinv = spd_inverse(S, jitter)
    S = full((S + S')/2);
    n = size(S, 1);
    [R, p] = chol(S);
    if p == 0
        Sinv = R\(R'\eye(n));
    else
        [V, D] = eig(S);
        lambda = max(real(diag(D)), jitter);
        Sinv = V*diag(1./lambda)*V';
    end
    Sinv = (Sinv + Sinv')/2;
end

function value = logdet_spd(S)
    S = full((S + S')/2);
    [R, p] = chol(S);
    if p == 0
        value = 2*sum(log(diag(R)));
    else
        [~, U, Pm] = lu(S);
        du = diag(U);
        value = log(det(Pm)*prod(sign(du))) + sum(log(abs(du)));
    end
end

function [freeY, Tu] = output_prediction_matrices(A, B, C, x0, horizon)
    freeY = zeros(horizon, 1);
    Tu = zeros(horizon, horizon);
    for row = 1:horizon
        freeY(row) = C*(A^(row-1))*x0;
        for col = 1:row-1
            Tu(row, col) = C*(A^(row-1-col))*B;
        end
    end
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
