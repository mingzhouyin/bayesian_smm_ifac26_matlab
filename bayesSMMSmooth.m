% Implement algorithms of data-driven optimal smoothing
% t-distribution uncertainties, no input errors, i.i.d. noise
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all; rng(1);

options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'StepTolerance', 1e-14, ...
    'MaxFunctionEvaluations', 1e5, 'MaxIterations',1e4);

Ne = 100;
errory1 = zeros(Ne,1);
errory2 = zeros(Ne,1);
errory3 = zeros(Ne,1);
errory4 = zeros(Ne,1);

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

t_calc = zeros(Ne,4);

for ii = 1:Ne
    %% Define system
    trueSys = drss(nx, ny, nu);
    trueSys.D = 0;
    while max(abs(pole(trueSys))) > 0.95
        trueSys = drss(nx, ny, nu);
    end
    trueSys = trueSys/norm(trueSys);
    
    %% Create OFFLINE input/output data
    % Data ud
    ud = randn(N, nu);
    ud_dist = ud;
    
    % Data yd
    yd = lsim(trueSys, ud);
    yd_dist = yd + sqrt(y_data_var) * trnd(dof,N,ny);
    
    %% Create ONLINE input/output data
    % Data u
    u = randn(L, nu);
    u_dist = u(:);
    
    % Data y
    y = lsim(trueSys, u);
    y_dist = y + sqrt(y_var) * trnd(dof,L,ny);
    y_dist = y_dist(:);
    
    % Auxiliary indicies
    idx_u = L * nu;
    idx_y = L * ny;
    
    %% Hankel matrix of offline trajectories
    % Compute input Hankel matrix
    Hu = GenHankel(ud_dist, L);
    % Compute output Hankel matrix
    Hy = GenHankel(yd_dist, L);
    
    %% MLE Problem
    % Set up optimization problem to estimate g*
    % Cost function
    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, y_dist, Hy, y_var, ...
        y_data_var, idx_y, dof);

    %% Non-convex optimiztion problem
    tic
    feval = realmax;
    for i = 1:5
        g0 = randn(M,1);
        [g_opt1,feval1] = fmincon(Pr_zetam_given_g, g0, [], [], Hu, u_dist, [], [], [], options);
        if feval1<feval
            g_opt = g_opt1;
            feval = feval1;
        end
    end

    % g0 = pinv(Hu)*u_dist;
    % g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hu, u_dist, [], [], [], options);
    
    % MAP Optimization Problem
    sigmag = covar_data(g_opt, y_data_var, idx_y);
    Pr_zeta_given_g = @(y) pr_zeta_given_g(y, y_dist, Hy, y_var, sigmag, g_opt, dof);
    
    y_est1 = fmincon(Pr_zeta_given_g, y_dist, [], [], [], [], [], [], [], options);
    t_calc(ii,1) = toc;

    %% Convex optimization
    tic
    lambda = L*y_data_var;
    cvx_begin quiet
        variable g_opt2(M)
        minimize lambda*sum_square(g_opt2) + sum_square(Hy*g_opt2-y_dist)
        subject to
            Hu*g_opt2 == u_dist
    cvx_end

    % MAP Optimization Problem
    sigmag2 = covar_data(g_opt2, y_data_var, idx_y);
    Pr_zeta_given_g = @(y) pr_zeta_given_g(y, y_dist, Hy, y_var, sigmag2, g_opt2, dof);
    
    y_est2 = fmincon(Pr_zeta_given_g, y_dist, [], [], [], [], [], [], [], options);
    t_calc(ii,2) = toc;
    
    %% N4SID + KF
    tic
    sys = n4sid(ud_dist,yd_dist,nx);
    X_filt = KF_RTS(y_dist', sys.A, sys.C, y_var*sys.K*sys.K', ...
        y_var*ones(ny,1), B=sys.B, u=u_dist);
    y_est3 = (sys.C*X_filt)';
    t_calc(ii,3) = toc;
    
    %% Proj
    % lambda_u = 1;
    % lambda_d = 0;
    % lambda = 1e-3;
    % for i = 1:20
    %     cvx_begin quiet
    %         variable g(M)
    %         minimize sum_square(Hy*g-y_dist) + lambda*norm(g,1)
    %         subject to
    %             Hu*g == u_dist
    %     cvx_end
    %     card = sum(abs(g)>1e-4);
    %     if card == L+nx
    %         break
    %     else
    %         if card > L+nx
    %             lambda_d = lambda;
    %             lambda = (lambda+lambda_u)/2;
    %         else
    %             lambda_u = lambda;
    %             lambda = (lambda+lambda_d)/2;
    %         end
    %     end
    % end
    % y_est4 = Hy*g;
    
    tic
    H1 = [Hu;Hy(1:nx,:)];
    Hyf = Hy(nx+1:end,:);
    Hyfhat = Hyf*H1'*((H1*H1')\H1);
    Hyhat = [Hy(1:nx,:); Hyfhat];
    cvx_begin quiet
        variable g(M)
        minimize sum_square(Hyhat*g-y_dist)
        subject to
            Hu*g == u_dist
    cvx_end
    y_est4 = Hyhat*g;
    t_calc(ii,4) = toc;
    
    %% Error
    errory1(ii) = norm(y_est1 - y)/sqrt(idx_y);
    errory2(ii) = norm(y_est2 - y)/sqrt(idx_y);
    errory3(ii) = norm(y_est3 - y)/sqrt(idx_y);
    errory4(ii) = norm(y_est4 - y)/sqrt(idx_y);
end

figure(10)
boxplot([errory1 errory2 errory3 errory4])

%% FUNCTIONS 
%% Covariance matrix
function sigma_g = covar_data(g, var, cov_size)
    % sigma_g = zeros(cov_size, cov_size);
    % for i = 1:cov_size
    %     for j = 1:cov_size
    %         cov_ij = 0;
    %         for k = 1:M-abs(i-j)
    %             cov_ij = cov_ij + g(k)*g(k+abs(i-j));
    %         end
    %         sigma_g(i,j) = var * cov_ij;
    %     end
    % end

    gcorr = xcorr(g,cov_size-1);
    gcorr = gcorr(cov_size:end);
    sigma_g = var*toeplitz(gcorr);
end

%% MLE probability density function
function MLE = pr_zetam_given_g(g, y_dist, Hy, y_var, y_data_var, idx_y, dof)
    % Compute psi matrix
    Psi = psi(g, y_var, y_data_var, idx_y);

    % Compute residual
    res = y_dist - Hy * g;

    % Logdet Psi
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));

    % Compute cost
    MLE = v + (dof+idx_y)*log(1+res'*(Psi\res)/dof);
    % MLE = v + res'*(Psi\res);
end

%%
function MAP = pr_zeta_given_g(y, y_dist, Hy, y_var, sigmag, g_opt, dof)
    delta1 = y - y_dist;
    delta2 = y - Hy*g_opt;
    MAP = log(1+delta1'*delta1/y_var/dof) + log(1+delta2'*(sigmag\delta2)/dof);
    % MAP = delta1'*delta1/y_var + delta2'*(sigmag\delta2);
end

%% Nonconvex function
function Psi = psi(g, y_var, y_data_var, idx_y)
    sigmag = covar_data(g, y_data_var, idx_y);
    Psi = sigmag + y_var*eye(idx_y); 
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
