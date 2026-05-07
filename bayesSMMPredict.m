% Implement algorithms of data-driven optimal prediction
% Gaussian uncertainties, input errors, i.i.d. noise
%
% Copyright 2026 Leibniz University Hannover, Mingzhou Yin & Seyed Ali Nazari

clc; clear; close all; rng(2);

options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'StepTolerance', 1e-14, ...
    'MaxFunctionEvaluations', 1e5, 'MaxIterations',1e4);

Ne = 100;
errory1 = zeros(Ne,1);
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
u_data_var = 1e-4;
y_data_var = 1e-4;
u_var = 1e-2;
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
    ud_dist = ud + sqrt(u_data_var) * randn(N,nu);
    
    % Data yd
    yd = lsim(trueSys, ud);
    yd_dist = yd + sqrt(y_data_var) * randn(N,ny);
    
    %% Create ONLINE input/output data
    % Data u
    u = randn(L, nu);
    u_dist = u + sqrt(u_var) * randn(L,nu);
    u_dist = u_dist(:);
    
    % Data y
    y = lsim(trueSys, u);
    y_dist = y + sqrt(y_var) * randn(L,ny);
    y_dist = y_dist(:);
    
    % Auxiliary indicies
    idx_u = L * nu;
    idx_y = L * ny;
    idx_yp = nx * ny;
    idx_yf = LL * ny;

    yp_dist = y_dist(1:idx_yp);
    yf = y(idx_yp+1:end);
    
    %% Hankel matrix of offline trajectories
    % Compute input Hankel matrix
    Hu = GenHankel(ud_dist, L);
    % Compute output Hankel matrix
    Hy = GenHankel(yd_dist, L);
    Hyp = Hy(1:idx_yp,:);
    Hyf = Hy(idx_yp+1:end,:);
    
    %% MLE Problem
    % Set up optimization problem to estimate g*
    % Cost function
    zetam = [u_dist;yp_dist];
    H1 = [Hu;Hyp];
    sigma_m = diag([u_var*ones(idx_u,1);y_var*ones(idx_yp,1)]);

    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, u_data_var, ...
        y_data_var, idx_u, idx_yp);

    %% Non-convex optimiztion problem
    tic
    % feval = realmax;
    % for i = 1:5
    %     g0 = randn(M,1);
    %     [g_opt1,feval1] = fmincon(Pr_zetam_given_g, g0, [], [], [], [], [], [], [], options);
    %     if feval1<feval
    %         g_opt = g_opt1;
    %         feval = feval1;
    %     end
    % end

    g0 = pinv(H1)*zetam;
    g_opt = fmincon(Pr_zetam_given_g, g0, [], [], [], [], [], [], [], options);

    % MAP Optimization Problem
    sigmag = blkdiag(covar_data(g_opt, u_data_var, idx_u), covar_data(g_opt, y_data_var, idx_y));
    zeta_opt = pr_zeta_given_g(zetam, [Hu;Hy], sigma_m, sigmag, g_opt, idx_yf);
    t_calc(ii,1) = toc;

    %% Convex optimization
    tic
    lambda_1 = 1/(sum_square(g0)*y_data_var+y_var);
    lambda_2 = 1/(sum_square(g0)*u_data_var+u_var);
    lambda = nx*y_data_var*lambda_1 + L*u_data_var*lambda_2;
    cvx_begin quiet
        variable g_opt2(M)
        minimize lambda*sum_square(g_opt2) + lambda_1*sum_square(Hyp*g_opt2-yp_dist) + lambda_2*sum_square(Hu*g_opt2-u_dist)
    cvx_end

    % MAP Optimization Problem
    yf_est1 = zeta_opt(idx_u+idx_yp+1:end);

    sigmag2 = blkdiag(covar_data(g_opt2, u_data_var, idx_u), covar_data(g_opt2, y_data_var, idx_y));
    zeta_opt2 = pr_zeta_given_g(zetam, [Hu;Hy], sigma_m, sigmag2, g_opt2, idx_yf);

    yf_est2 = zeta_opt2(idx_u+idx_yp+1:end);
    t_calc(ii,2) = toc;

    %% N4SID
    % sys = arx(ud_dist,yd_dist,[nx,nx,1]);
    % y_est = zeros(L,1);
    % y_est(1:nx) = yp_dist;
    % a = -fliplr(sys.A(2:end));
    % b = fliplr(sys.B(2:end));
    % for i = 1:LL
    %     y_est(nx+i) = a*y_est(i:i+nx-1) + b*u_dist(i:i+nx-1);
    % end
    % yf_est3 = y_est(idx_yp+1:end);

    tic
    sys = n4sid(ud_dist,yd_dist,nx);
    y_est = compare(iddata(y_dist,u_dist),sys);
    yf_est3 = y_est(idx_yp+1:end).OutputData;
    t_calc(ii,3) = toc;

    %% Proj
    tic
    Hyfhat = Hyf*H1'*((H1*H1')\H1);
    cvx_begin quiet
        variable g(M)
        minimize sum_square(H1*g-zetam)
    cvx_end
    yf_est4 = Hyfhat*g;
    % yf_est4 = Hyfhat*pinv(H1)*zetam;
    t_calc(ii,4) = toc;
    
    %% Error
    errory1(ii) = norm(yf_est1 - yf)/sqrt(idx_yf);
    errory2(ii) = norm(yf_est2 - yf)/sqrt(idx_yf);
    errory3(ii) = norm(yf_est3 - yf)/sqrt(idx_yf);
    errory4(ii) = norm(yf_est4 - yf)/sqrt(idx_yf);
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
function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, u_data_var, y_data_var, idx_u, idx_y)
    % Compute psi matrix
    Psi = psi(g, sigma_m, u_data_var, y_data_var, idx_u, idx_y);

    % Compute residual
    res = zetam - H * g;

    % Logdet Psi
    [~, U, P] = lu(Psi);
    du = diag(U);
    c = det(P) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));

    % Compute cost
    MLE = v + res' * (Psi \ res);
end

%%
function MAP = pr_zeta_given_g(zetam, H, Sigma_m, sigmag, g_opt, idx_yf)
    res1 = inv(sigmag) + blkdiag(inv(Sigma_m),zeros(idx_yf,idx_yf));
    res2 = [Sigma_m \ zetam; zeros(idx_yf,1)] + sigmag \ H * g_opt;
    MAP = res1 \ res2;
end

%% Nonconvex function
function Psi = psi(g, sigma_m, u_data_var, y_data_var, idx_u, idx_y)
    sigmag = blkdiag(covar_data(g, u_data_var, idx_u), covar_data(g, y_data_var, idx_y));
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
