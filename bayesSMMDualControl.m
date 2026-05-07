% Implement algorithms of data-driven optimal control
% Gaussian uncertainties, no input errors, correlated noise
%
% Copyright 2025 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all; rng(2);

options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'StepTolerance', 1e-14, ...
    'MaxFunctionEvaluations', 1e5, 'MaxIterations',1e4);

Ne = 1;

%% Parameters
nx = 10;    % States
L = 40;     % N >= (nu+1)(L+nx)-1
N = 2*(L+nx)-1;    % Data points
LL = L-nx;

Nctr = 100;

k = 0.95; var = 1e-6;

%% Define system & control task
alpha = 0.4; beta = 0.3;
A = toeplitz([1-2*alpha-beta;alpha;zeros(8,1)]);
A(1,1) = 1-alpha-beta; A(10,10) = 1-alpha-beta;
B = [1;zeros(9,1)];
C = [1 zeros(1,9)];
D = 0;
trueSys = ss(A,B,C,D,-1);
trueSys = trueSys/norm(trueSys);

% Reference
yref = ones(Nctr+LL-1,1);
gain = freqresp(trueSys,0);
uref = yref / gain;
q = 5;
r = 0.5;

for ii = 1:Ne
    %% Create OFFLINE input/output data
    % Data ud
    ud_0 = randn(N,1);
    ud = ud_0;
    
    % Data yd
    yd_0 = lsim(trueSys, ud);
    Lchol = chol(toeplitz(var*k.^(0:N-1)'));
    yd = yd_0 + Lchol' * randn(N,1);
    
    %% Hankel matrix of offline trajectories
    Hu = GenHankel(ud, L);
    Hy = GenHankel(yd, L);
    
    %% Create ONLINE input/output data
    % Data u
    u_ini_0 = randn(nx,1);
    u_ini = u_ini_0;

    % Online noise sequence
    Lchol = chol(toeplitz(var*k.^(0:nx+Nctr-1)'));
    v = Lchol' * randn(nx+Nctr,1);
    
    % Data y
    [y_ini_0,~,x] = lsim(trueSys, u_ini_0);
    x0 = trueSys.A*x(end,:)' + trueSys.B*u_ini(end);
    y_ini = y_ini_0 + v(1:nx,:);

    %% Control
    uu_ctr = zeros(Nctr,LL);
    yy_ctr = zeros(Nctr,LL);
    y0_ctr = zeros(Nctr,1);
    u_ctr = zeros(Nctr+nx,1);
    y_ctr = zeros(Nctr+nx,1);
    error = zeros(Nctr,1);

    u_ctr(1:nx) = u_ini;
    y_ctr(1:nx) = y_ini;

    for i = 1:Nctr
        %% MLE Problem
        % Set up optimization problem to estimate g*
        % Cost function
        zetam = [uref(i:i+LL-1);y_ini;yref(i:i+LL-1)];
        H1 = [Hu(nx+1:end,:);Hy];
        sigma_m = blkdiag(eye(LL)/r,toeplitz(var*k.^(0:nx-1)'),eye(LL)/q);
        Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, k, var, LL, L);
        
        % Non-convex optimiztion problem
        g0 = pinv([Hu;Hy])*[u_ini;zetam];
        g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hu(1:nx,:), u_ini, [], [], [], options);
        
        %% MAP Optimization Problem
        sigmag = covar_data(g_opt, k, var, L);
        zeta_opt = pr_zeta_given_g([y_ini;yref(i:i+LL-1)], Hy, blkdiag(toeplitz(var*k.^(0:nx-1)'),eye(LL)/q), sigmag, g_opt);
    
        uu_ctr(i,:) = Hu(nx+1:end,:)*g_opt;
        yy_ctr(i,:) = zeta_opt(nx+1:end);

        y_ini_hat = zeta_opt(1:nx);
        
        u_ctr(i+nx) = uu_ctr(i,1);
        y0_ctr(i) = trueSys.C*x0 + trueSys.D*u_ctr(i+nx);
        y_ctr(i+nx) = y0_ctr(i) + v(i+nx);
        x0 = trueSys.A*x0 + trueSys.B*u_ctr(i+nx);

        error(i) = r*(u_ctr(i+nx)-uref(i))^2+q*(y0_ctr(i)-yref(i))^2;
        % if i-LL >= 0
        %     Hu = [Hu u_ctr(i-LL+1:i+nx)];
        %     Hy = [Hy y_ctr(i-LL+1:i+nx)];
        % end
        u_ini = [u_ini(2:end);u_ctr(i+nx)];
        y_ini = [y_ini_hat(2:end);y_ctr(i+nx)];

    end
end

plot(yref(1:Nctr))
hold on
plot(y0_ctr)
plot(y_ctr(nx+1:end))

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
end

%% MLE probability density function
function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, k, var, LL, L)
    % Compute psi matrix
    Psi = psi(g, sigma_m, k, var, LL, L);

    % Compute residual
    res = zetam - H * g;

    % Logdet Psi
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));

    % Compute cost
    MLE = v + res' * (Psi \ res) + 0.1 * norm(g,1);
end

%%
function MAP = pr_zeta_given_g(zetam, H, Sigma_m, sigmag, g_opt)
    res1 = inv(sigmag) + inv(Sigma_m);
    res2 = Sigma_m \ zetam + sigmag \ H * g_opt;
    MAP = res1 \ res2;
end

%% Nonconvex function
function Psi = psi(g, sigma_m, k, var, LL, L)
    sigmag = blkdiag(zeros(LL,LL), covar_data(g, k, var, L));
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
