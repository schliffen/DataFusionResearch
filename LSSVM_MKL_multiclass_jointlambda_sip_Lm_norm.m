function [theta,beta,E,sis,t]=LSSVM_MKL_multiclass_jointlambda_sip_Lm_norm(Kmix,L,m_norm)

time0 = cputime;

% SIP for multiple class Lm-norm LSSVM MKL solver
% Kmix  a cell object of multiple centered kernels
% L is a Nxk matrix of labels, where n is the number of data samples, k is the number of classes, notice that this version requires that A(:,j)^-2 = ones, for all j
% m_norm is the rank of the m-norm on the primal problem (kernel coefficients) 
%
% output 
% theta:  kernel coefficients
% beta:      the dual variables 
% E:      the dummy variable checking the covergence
% sis:    the dummy variable of f(alpha)
% t:      the costed CPU time
%

% Notice:
% The program is for L_2 LS-SVM MKL
% the single LSSVM is solved by the included function linsolve_LSSVM_multiclass_beta
% the Lm-norm is optimized by the  solve_theta_relax_cvx_lssvm function based on cvx


% Coded by Shi Yu shee.yu@gmail.com  March 2010


N = length(Kmix{1});   % N is the number of samples
k = size(L,2);  % size of classes
beta = rands(N,k)/2;  % the dual variables for LS-SVM MKL

% add the identity matrix
Kmix = [Kmix {eye(size(N,N))}];

sis = compute_sis(Kmix, L, beta);

E=[];

for n=1:1:100
    disp(n)
    [theta,gamma] =  solve_theta_relax_cvx_lssvm(beta,sis,L,m_norm);
    [Km] = compose_kernels(Kmix,theta);
    Kmerge =[{Km}];
    
    [trash1, alpha, beta, b] = linsolve_LSSVM(Kmix, L, theta);
    S = compute_sis(Kmix, L, beta);  % compute_sis only compute the quadratic term invovling the kernels
    sis = [sis; S];
    temp = compute_sis(Kmerge, L, beta)  - trace(beta'*L);  % this is the full objective function
    eps = 1-temp/gamma
    E = [E eps];
    if length(E)>=3 && (E(end)-E(end-1))<5e-6
	t=cputime-time0;
        break
    end
    t=cputime-time0;
end

time = cputime - time0;

function sis=compute_sis(kernels, L, B)
% this function doesn't need lambda, just store the quadratic term containing the kernels

N = length(kernels);
p = size(L,2);  % size of classes
sis = zeros(1,N);

for n=1:N
   for j = 1:1:p
	sis(n) = sis(n) + 0.5 * B(:,j)' * kernels{n} * B(:,j);
   end
end


function  [obj,alpha,beta,b] =linsolve_LSSVM(K,L,theta)
% solve LS-SVM as linear problem, the formulation adapted here is different from the formulation given by Suykens et al.
G = zeros(size(K{1}));
for n=1:length(K)
    G = G + theta(n) * K{n};
end
n = size(G,1);  % number of data
p = size(L,2);  % number of classes

for loop = 1:1:p
	Y{loop} = diag(L(:,loop));
end
onevec = ones(size(L(:,1)));
H =[0 onevec';onevec G];


cond(H)
J = [zeros(1,p); 1./L];
sol = linsolve(H,J);
beta = sol(2:end,:);
b = sol(1,:);

alpha = zeros(size(beta));
for loop=1:1:p
	alpha(:,loop) = inv(Y{loop})*beta(:,loop);
end
obj=0;


function [cK] = compose_kernels(kernels,theta)

cK = zeros(size(kernels{1}));
for n=1:1:length(kernels)
	cK = cK + theta(n)*kernels{n};
end

function [theta,obj] =  solve_theta_relax_cvx_lssvm(beta,sis,L,m_norm)


% beta is the dual variable 
% sis is the matrix of active constraints
%            s1_1 s2_1 ... sp_1      -- iteration 1
%            s1_2 s2_2 ... sp_2      -- iteration 2
%            ...
%            s1_n s2_n ... sp_n      -- iteration n
%a is the regression or label vector


[g,p] = size(sis); 
% g is the number of constraints
% p is the number of kernels

n = size(beta,1);


cvx_begin
   variables u theta(p)
   maximize(u)
   subject to
     for kk=1:g

        size(beta)
	size(L)
	
        0.5*sis(kk,:) * theta - -trace(beta'*L).*ones(n,1) >=u; 
     end
     for kk=1:p
	  theta(kk) >= 0
     end
     norm(theta,m_norm) <= 1
cvx_end

obj = min(sis*theta)-trace(beta'*L).*ones(n,1);
obj = obj(1);

