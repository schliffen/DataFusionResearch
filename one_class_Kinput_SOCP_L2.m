function [t,theta,alpha,spendtime] = one_class_Kinput_SOCP_L2(K,nu)

%function [t,theta,alpha,spendtime] = one_class_Kinput_SOCP_L2(K,nu)
%
% Learn the optimal kernel by maximising the distance between the data
% points and the origin. Here a positive lower bound on the kernel weights
% mu can be specified as well.
% this is different from Tijl De Bie's method which is based on L1 norm
% 
%
%INPUTS:
% K = a struct array containing the kernels
% nu      = the fraction of outliers tolerated
%
%OUTPUTS:
% t     = the objective (margin or so...)
% theta   = the weights of the kernels
% alpha = the dual variables for the weight vector
% spendtime = the CPU time used for optimization
%
%Author: Tijl De Bie, March 2006.
%	 Shi Yu,      June 2009  shee.yu@gmail.com.


% add the path of SeDuMi package
%addpath '~/no_backup/fullgenomic/SeDuMi_1_21/';


j=length(K); % the number of kernels
n=size(K{1},2); % the number of samples

sK = [];

for loop=1:j
  temp= K{loop} + 1E-6*eye(size(K{loop}));
  sK{loop} = chol(temp);
  clear temp;
end

clear K;

%y = [t u(1-k) alpha], dimension 1+n
%
% max b'y
% s.t. c-A'y >= 0

% object function: -t
b = [-1 ; sparse(n+j+1,1)];
%disp 'b';
%size(b)

c = sparse([-1+1e-5;... the sum of alphas constraint: alpha'*e -1 >= 0
        1+1e-5;... the sum of alphas constraint: -alpha'*e +1.00... >= 0
        sparse(n,1);... the lower bounds on alpha: alpha >= 0
        ones(n,1)/(nu*n);... the upper bounds on alpha: 1/(nu*n)-alpha >= 0
	0.5;
	-0.5;
	sparse(1+j,1);... the q cone quadratic constraints 
        sparse((n+2)*j,1)]); % the r cone quadratic constraints
%disp 'c';
%size(c)
% Inequality constraints


mAt = [];

mAt = sparse([sparse(1,j+2) ones(1,n);  % alpha'*e -1 >= 0
        sparse(1,j+2) -ones(1,n); % -alpha'*e +1.00... >= 0
        sparse(n,j+2) speye(n); % alpha >= 0
        sparse(n,j+2) -speye(n);  % 1/(nu*n)-alpha >= 0
	sparse(1,j+1) -1 sparse(1,n);  % 0.5- dummy >=0
	sparse(1,j+1) +1 sparse(1,n);  % -0.5+dummy >=0
        1 sparse(1,j+n+1); % the q cone constraint on t>norm(u)
	sparse(j,1)  speye(j) sparse(j,n+1); % the q cone constraint
	spalloc(j*(n+2),n+j+2,j*n*n+2*j)]);


mAt=full(mAt);



%quadratic cone constraints

for i=1:j
    mAt(j+6+2*n+(i-1)*(n+2),i+1)=1;
    mAt(j+7+2*n+(i-1)*(n+2),j+2)=1;
    mAt(j+8+2*n+(i-1)*(n+2):j+5+2*n+i*(n+2),3+j:end)=sK{i};
end


%mAt
mAt = sparse(mAt);



%size(mAt)


A.l = [4+2*n];
A.q = 1+j;
A.r = [ones(1,j)*(n+2)];

pars.fid=0;


tic
[primalsol, dualsol, info] = sedumi(-mAt',b,c,A,pars);

spendtime = toc;


dualsol
t=dualsol(1:j+1);
alpha=dualsol(3+j:end);
%sum(alpha)

theta=[];

for loop=1:j
theta = [theta;primalsol(2*n+6+j+(loop-1)*(n+2))];
end

