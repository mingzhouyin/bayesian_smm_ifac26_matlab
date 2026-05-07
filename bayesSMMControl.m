% Implement algorithms of data-driven optimal control
% Gaussian uncertainties, no input errors, correlated noise
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

k = 0.95; var = 1e-2;

t_calc = zeros(Ne,4);

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
    
    %% MLE Problem
    % Set up optimization problem to estimate g*
    % Cost function
    zetam = [uref;y_dist;yref];
    H1 = [Huf;Hy];
    sigma_m = blkdiag(eye(idx_uf)/r,toeplitz(var*k.^(0:nx-1)'),eye(idx_yf)/q);
    Pr_zetam_given_g = @(g) pr_zetam_given_g(g, zetam, H1, sigma_m, k, var, idx_uf, idx_y);
    
    %% Non-convex optimiztion problem
    tic
    g0 = pinv([Hu;Hy])*[u_dist;zetam];
    g_opt = fmincon(Pr_zetam_given_g, g0, [], [], Hup, u_dist, [], [], [], options);

    % MAP Optimization Problem
    sigmag = covar_data(g_opt, k, var, idx_y);
    zeta_opt = pr_zeta_given_g([y_dist;yref], Hy, blkdiag(toeplitz(var*k.^(0:nx-1)'),eye(idx_yf)/q), sigmag, g_opt);

    u_ctr1 = Huf*g_opt;
    yy_ctr1 = zeta_opt(idx_yp+1:end);
    y_ctr1 = lsim(trueSys, u_ctr1, [], x0);
    errory1(ii) = r*sum_square(u_ctr1-uref)+q*sum_square(y_ctr1-yref);
    t_calc(ii,1) = toc;

    %% Convex optimization
    tic
    lambda_1 = 1/(sum_square(g0)*var+var);
    lambda_2 = 1/(sum_square(g0)*var+1/q);
    lambda = nx*var*lambda_1 + LL*var*lambda_2;
    cvx_begin quiet
        variable g_opt2(M)
        minimize lambda*sum_square(g_opt2) + lambda_1*sum_square(Hyp*g_opt2-y_dist) + lambda_2*sum_square(Hyf*g_opt2-yref) + r*sum_square(Huf*g_opt2-uref)
        subject to
            Hup*g_opt2 == u_dist
    cvx_end
    
    % MAP Optimization Problem
    sigmag2 = covar_data(g_opt2, k, var, idx_y);
    zeta_opt2 = pr_zeta_given_g([y_dist;yref], Hy, blkdiag(toeplitz(var*k.^(0:nx-1)'),eye(idx_yf)/q), sigmag2, g_opt2);

    u_ctr2 = Huf*g_opt2;
    yy_ctr2 = zeta_opt2(idx_yp+1:end);
    y_ctr2 = lsim(trueSys, u_ctr2, [], x0);
    errory2(ii) = r*sum_square(u_ctr2-uref)+q*sum_square(y_ctr2-yref);
    t_calc(ii,2) = toc;

    %% N4SID
    tic
    sys = n4sid(ud_dist,yd_dist,nx);
    X_filt = KF_RTS(y_dist', sys.A, sys.C, sys.NoiseVariance*sys.K*sys.K', ...
        sys.NoiseVariance*ones(ny,1), B=sys.B, u=u_dist);
    x0hat = sys.A*x(:,end) + sys.B*u_dist(end);
    cvx_begin quiet
        variables u_ctr3(idx_uf) yy_ctr3(idx_yf)
        minimize r*sum_square(u_ctr3-uref)+q*sum_square(yy_ctr3-yref)
        subject to
            for i = 1:idx_yf
                yy_ctr3(i) == sys.C*x0hat;
                x0hat = sys.A*x0hat + sys.B*u_ctr3(i);
            end
    cvx_end
    y_ctr3 = lsim(trueSys, u_ctr3, [], x0);
    errory3(ii) = r*sum_square(u_ctr3-uref)+q*sum_square(y_ctr3-yref);
    t_calc(ii,3) = toc;

    %% Proj
    tic
    H1 = [Hu;Hy(1:nx,:)];
    Hyf = Hy(nx+1:end,:);
    Hyfhat = Hyf*H1'*((H1*H1')\H1);
    cvx_begin quiet
        variables u_ctr4(idx_uf) yy_ctr4(idx_yf)
        minimize r*sum_square(u_ctr4-uref)+q*sum_square(yy_ctr4-yref)
        subject to
            yy_ctr4 == Hyfhat*pinv(H1)*[u_dist;u_ctr4;y_dist]
    cvx_end
    y_ctr4 = lsim(trueSys, u_ctr4, [], x0);
    errory4(ii) = r*sum_square(u_ctr4-uref)+q*sum_square(y_ctr4-yref);
    t_calc(ii,4) = toc;
    
    %% Plot results
    % figure(1);
    % plot(yy_ctr3,'Marker','o')
    % hold on
    % plot(y_ctr3,'Marker','*')
    % legend('Pred y', 'y')
end

figure(10)
boxplot(sqrt([errory1 errory2 errory3 errory4])/q/sqrt(idx_yf))

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
function MLE = pr_zetam_given_g(g, zetam, H, sigma_m, k, var, idx_u, idx_y)
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
    MLE = v + res' * (Psi \ res);
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
