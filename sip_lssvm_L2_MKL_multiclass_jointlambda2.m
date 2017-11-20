function [theta,A,E,sis,t]=sip_lssvm_L2_MKL_multiclass_jointlambda2(kernels,L)

time0 = cputime;

% SLP for multiple class L2-norm LSSVM MKL solver
% kernels  a cell object of multiple centered kernels
% A is a nxk matrix of labels, where n is the number of data samples, k is the number of classes, notice that this version requires that A(:,j)^-2 = ones, for all j
% 
%
% output 
% theta:  kernel coefficients
% A:      the dual variables 
% E:      the dummy variable checking the covergence
% sis:    the dummy variable of f(alpha)
% t:      the costed CPU time
%

% Notice:
% The program is for L_2 LS-SVM MKL
% the single LSSVM is solved by the included function linsolve_LSSVM_multiclass_beta
% the L2-norm is optimized by the  solve_theta_relax function based on Sedumi


% note that the dual variables are still based on alpha, not the transformed beta

% Coded by Shi Yu shi.yu@esat.kuleuven.be and Tillmann Falck tillmann.falck@esat.kuleuven.be, 2009 


N = length(kernels{1});   % N is the number of samples
p = size(L,2);  % size of classes

B = rands(N,p)/2;  % the dual variables for LS-SVM MKL

% add the identity matrix
kernels = [kernels {eye(size(N,N))}];

sis = compute_sis(kernels, L, B);

E=[];

for n=1:1:100
    disp(n)
    [dummy,theta,gamma] = solve_theta_relax(B,sis,L);
    dummy 
    theta 
    gamma
    [Km] = compose_kernels(kernels,theta);
    Kmerge =[{Km}];
    
    [trash1, A, B, b] = linsolve_LSSVM(kernels, L, theta);
    S = compute_sis(kernels, L, B);  % compute_sis only compute the quadratic term invovling the kernels
    sis = [sis; S];
    B
    temp = compute_sis(Kmerge, L, B)  - trace(B'*L);  % this is the full objective function
    temp
    eps = 1-temp/gamma
    E = [E eps]
    %if abs(eps)<5e-3 
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

J = [zeros(1,p); 1./L];
sol = linsolve(H,J);
beta = sol(2:end,:);
b = sol(1,:);

alpha = zeros(size(beta));
for loop=1:1:p
	alpha(:,loop) = inv(Y{loop})*beta(:,loop);
end
obj=0;


function [dummy, theta, obj] =  solve_theta_relax(beta,sis,L)

% alpha is the dual variable 
% sis is the matrix of active constraints
%            s1_1 s2_1 ... sp_1      -- iteration 1
%            s1_2 s2_2 ... sp_2      -- iteration 2
%            ...
%            s1_n s2_n ... sp_n      -- iteration n
%a is the regression or label vector


p = size(sis,2); % p is the number of kernels



% max b'y
% s.t. c-A'y >= 0

% variable y: dummy  fixvariable theta_1 theta_2 ... theta_p
% object function max: dummy 


b = [1; sparse(p+1,1)];


%if size(sis,1)==1
%	else
%		sis = sis (2:end,:);
%end


n = size(sis,1); % n is the number of iteration already done
	
% linear constraints 
mAt = [ones(n,1) sparse(n,1) -sis;   
       0 1  sparse(1,p);
       0 -1 sparse(1,p);
       sparse(p,2) -speye(p,p);
       sparse(p+1,1) -speye(p+1,p+1)];


%size(mAt)
mAt = sparse(mAt);

c= [ -trace(beta'*L).*ones(n,1);
	1;
	-1;
	sparse(p+p+1,1)];

%size(c)

c = sparse(c);

K.l = [n+p+2];
K.q = 1+p;


pars.fid=1;

%tic

[primalsol, dualsol, info] = sedumi(mAt',b,c,K,pars);

%st = toc
dummy = dualsol(1);
theta = dualsol(3:end);

obj = b'*dualsol;
%obj = 0.5*theta'*sis - sum(sum(alpha));



function [cK] = compose_kernels(kernels,theta)

cK = zeros(size(kernels{1}));
for n=1:1:length(kernels)
	cK = cK + theta(n)*kernels{n};
end