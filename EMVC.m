function [U] = EMVC(Data, X, lamb, betaa)

%addpath('MIC/');
addpath('LibADMM-master/proximal_operator');
addpath('code_coregspectral/');

num_views = size(X,2);
N = size(X{1,1},2);
obj = zeros(1, 1000);

maxIter = 100;
mu=10^(-6);
max_mu = 10^(10);
rho = 1.9;

beta = betaa;
lambda = lamb;

for i=1:num_views
    data_tmp = Data{i};
    Data{i} = NormalizeFea(data_tmp);
end


%%initialization of p hat
fprintf('Initializing p hat \n');
P_hat = zeros(N, N);
Q = zeros(N, N);
Z = zeros(N, N);

%%initialization of p^{(i)}s (transition probability matrices)
fprintf('Initializing p^{(i)}s \n');
P_i   = zeros(N, N, num_views);
K=[];
sigma(1)=optSigma(Data{1});
sigma(2)=optSigma(Data{2});
%sigma(3)=optSigma(Data{3});
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=100;%same setting as co-regspectral multiview spectral
    %options.t = sigma(j);
    K(:,:,j) = constructKernel(Data{j},Data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    P_i(:,:,j)=L_rw;
end


%%initialization of Lagrange multipliers
fprintf('Initializing y^{(i)}s \n');
Y_i = zeros(N, N, num_views);

%%initialization of e^{(i)}s (representation errors)
fprintf('Initializing E^{(i)}s \n');
E_i = randn(N, N, num_views);


thresh = 1e-8;

for iter = 1:maxIter

    %Solving P_hat
    fprintf('Updating P hat\n');
    temp = (sum(P_i - E_i - Y_i/mu, 3) + Q - Z /mu)/(num_views + 1);
    P_hat = project_simplex(temp);
    disp(sum(P_hat(1,:)));
    fprintf('Updating or Solving E \n');
    
    %d = K * N
    Vec = 1:num_views * N;
    row = 0;
    depth = 1;
    for ii = 1:num_views * N
        row = row + 1;

        Vec(ii) = 1/(2 * norm(E_i(row, :, depth), 2));   
            
        if(row == N)
            row = 0;
            depth = depth + 1;
        end
    end
    D_hat = diag(Vec);
    

    Bl = 1:N * num_views;
    B  = zeros(num_views * N, 1); 
    for i = 1:N
        for j = 1:num_views
            Bl(1 + (j-1) * N: j*N) = repelem(1/(2 * norm(E_i(:, i, j), 2)), N);
        end
        D_i = diag(Bl);
        A = (beta/mu) * D_hat + (lambda/mu) * D_i + eye(num_views * N, num_views * N);
             
        for j = 1:num_views
            B(1+(j-1)*N:j*N, 1) = (P_i(:, i, j) - P_hat(:, i) - (1/mu) * Y_i(:, i, j));
        end
        
        Res = A\B; 
        for j=1:num_views
            E_i(:, i, j) =  Res((j - 1) * N + 1: j * N, 1);
        end
    end
    
    fprintf('Done with updating E\n');
        
    %solving Q using SVD
    fprintf('Updating Q\n');
    [Q,nuclearnormQ] = prox_nuclear(P_hat+Z/mu,1/mu);
    
    fprintf('Updating Z\n');
    Z = Z + mu * (P_hat - Q);
    
    
    %do we need to define a new mu for Y_i
    fprintf('Updating Y_i\n');
    for ii=1:num_views
        Y_i(:, :, ii) = Y_i(:, :, ii) + mu * (P_hat + E_i(:, :, ii) - P_i(:, :, ii));
    end
    
    mu = min(rho * mu, max_mu);
    
    %calculate the objective value
    E = E_i(:, :, 1);
    for ii=2:num_views
        E = vertcat(E, E_i(:, :, ii));
    end
    Obj = norm_nuclear(P_hat) + beta * norm_l21(E) + lambda * norm_g1(E_i, N, num_views);
    
    obj(iter) = Obj;
    disp(Obj);

    if iter>2
        Obj_diff = (obj(iter-1)-obj(iter)) /obj(iter-1);
        
        if Obj_diff < thresh
            break;
        end
    end
    
end


[pi,~]=eigs(P_hat',1);
Dist=pi/sum(pi);
pi=diag(Dist);
% P=(pi*P+P'*pi)/2;
U=(pi^0.5*P_hat*pi^-0.5+pi^-0.5*P_hat'*pi^0.5)/2;
