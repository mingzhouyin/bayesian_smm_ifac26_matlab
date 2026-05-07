% Implement algorithms of data-driven optimal control
% Gaussian uncertainties, no input errors, correlated noise
%
% Copyright 2025 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all; rng(2);

options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'StepTolerance', 1e-14, ...
    'MaxFunctionEvaluations', 1e5, 'MaxIterations',1e4);

Ne = 1;

nl = 10;
lambda_grid = logspace(-1,0,nl);

errory1 = zeros(Ne,nl);
errory2 = zeros(Ne,1);
errory3 = zeros(Ne,1);
errory4 = zeros(Ne,1);

%% Parameters
nx = 10;    % States
nu = 1;     % Inputs
ny = 1;     % Outputs
N = 100;    % Data points
L = 40;     % N > (nu+1)(L+nx)-1
M = N-L+1;  % Number of columns
LL = L-nx;

k = 0.95; var = 1e-2;

%% Define system
alpha = 0.4; beta = 0.3;
A = toeplitz([1-2*alpha-beta;alpha;zeros(8,1)]);
A(1,1) = 1-alpha-beta; A(10,10) = 1-alpha-beta;
B = [1;zeros(9,1)];
C = [1 zeros(1,9)];
D = 0;
trueSys = ss(A,B,C,D,-1);
trueSys = trueSys/norm(trueSys);

for ii = 1:Ne
    %% Create OFFLINE input/output data
    % Data ud
    ud = randn(N, nu);
    ud_dist = ud;
    
    % Data yd
    yd = lsim(trueSys, ud);
    Lchol = chol(toeplitz(var*k.^(0:N-1)'));
    yd_dist = yd + Lchol' * randn(N,ny);
    
    %% Create ONLINE input/output data
    % Data u
    u = randn(nx, nu);
    u_dist = u(:);
    
    % Data y
    [y,~,x] = lsim(trueSys, u);
    x0 = trueSys.A*x(end,:)' + trueSys.B*u_dist(end);
    Lchol = chol(toeplitz(var*k.^(0:nx-1)'));
    y_dist = y + Lchol' * randn(nx,ny);
    y_dist = y_dist(:);

    % Reference
    yref = [ones(10,1);-ones(10,1);ones(10,1)];
    gain = 1/freqresp(trueSys,0);
    uref = gain*[ones(10,1);-ones(10,1);ones(10,1)];
    % uref = zeros(LL,1);
    q = 5;
    r = 0.5;
    
    % Auxiliary indicies
    idx_up = nx * nu;
    idx_uf = LL * nu;
    idx_yp = nx * ny;
    idx_yf = LL * ny;
    idx_y = idx_yp + idx_yf;
    
    %% Hankel matrix of offline trajectories
    % Compute input Hankel matrix
    Hu = GenHankel(ud_dist, L);
    % Compute output Hankel matrix
    Hy = GenHankel(yd_dist, L);
    Hup = Hu(1:nx,:);
    Huf = Hu(nx+1:end,:);
    Hyp = Hy(1:nx,:);
    Hyf = Hy(nx+1:end,:);

    for jj = 1:nl
        lambda = lambda_grid(jj);
        %% MLE Problem
        % Set up optimization problem to estimate g*
        % Cost function
        zetam = [uref;y_dist;yref];
        H1 = [Huf;Hy];
        sigma_m = blkdiag(eye(idx_uf)/r,toeplitz(var*k.^(0:nx-1)'),eye(idx_yf)/q);
        Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, k, var, idx_uf, idx_y, lambda);
        
        % Non-convex optimiztion problem
        g0 = pinv([Hu;Hy])*[u_dist;zetam];
        g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hup, u_dist, [], [], [], options);
        
        %% MAP Optimization Problem
        sigmag = covar_data(g_opt, k, var, idx_y);
        zeta_opt = pr_zeta_given_g([y_dist;yref], Hy, blkdiag(toeplitz(var*k.^(0:nx-1)'),eye(idx_yf)/q), sigmag, g_opt);
    
        u_ctr1 = Huf*g_opt;
        yy_ctr1 = zeta_opt(idx_yp+1:end);
        y_ctr1 = lsim(trueSys, u_ctr1, [], x0);
        errory1(ii,jj) = r*sum_square(u_ctr1-uref)+q*sum_square(y_ctr1-yref);
    end
end

%% FUNCTIONS 
%% Covariance matrix
function sigma_g = covar_data(g, k, var, cov_size)
    M = length(g);
    gcorr = xcorr(g,M-1);
    sigma_g = zeros(cov_size, cov_size);
    varseq = var * k.^abs(2-M-cov_size:M+cov_size-2);
    for i = 1:cov_size
        for j = 1:cov_size
            sigma_g(i,j) = varseq(i-j+cov_size:i-j+cov_size+2*M-2) * gcorr;
        end
    end
    
    % gcorr = xcorr(g,cov_size-1);
    % gcorr = gcorr(cov_size:end);
    % sigma_g2 = var*toeplitz(gcorr);
end

%% MLE probability density function
function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, k, var, idx_u, idx_y, lambda)
    % Compute psi matrix
    Psi = psi(g, sigma_m, k, var, idx_u, idx_y);

    % Compute residual
    res = zetam - H * g;

    % Logdet Psi
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));

    % Compute cost
    MLE = res' * (Psi \ res) + lambda*norm(g,1);
end

%%
function MAP = pr_zeta_given_g(zetam, H, Sigma_m, sigmag, g_opt)
    res1 = inv(sigmag) + inv(Sigma_m);
    res2 = Sigma_m \ zetam + sigmag \ H * g_opt;
    MAP = res1 \ res2;
end

%% Nonconvex function
function Psi = psi(g, sigma_m, k, var, idx_u, idx_y)
    sigmag = blkdiag(zeros(idx_u,idx_u), covar_data(g, k, var, idx_y));
    Psi = sigmag + sigma_m;
end

%%
function H = GenHankel(X, window)
    [N, d] = size(X); 

    numCols = N - window + 1;
    H = zeros(d * window, numCols);

    for i = 1:numCols
        block = X(i:i+window-1, :)';
        H(:, i) = block(:);
    end
end
