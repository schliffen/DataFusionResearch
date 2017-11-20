function [t,theta,alpha,spendtime] = one_class_Kinput_SOCP_Linf(K,nu,lb)

%function [t,theta,alpha,spendtime] = (K,nu,lb)
%
% Learn the optimal kernel by maximising the distance between the data
% points and the origin. Here a positive lower bound on the kernel weights
% mu can be specified as well.
%
%INPUTS:
% K = a struct array containing the kernels
% nu      = the fraction of outliers tolerated
% lb      = a lower bound on the weights mu, multiplied by the number of
%           kernels. Should be in (0,1): 0 means the SDP method, 1 means
%           uniform weighting. It is the fraction of the total weight that
%           must be spread equally over all kernels.
%
%OUTPUTS:
% t     = the objective (margin or so...)
% theta    = the coefficients of kernels
% alpha = the dual variables of 1-SVM
%
%Author: Tijl De Bie, March 2006.
%	 Shi Yu,      June 2009   shi.yu@hotmail.com

% add the path of Sedumi package

addpath '~/no_backup/fullgenomic/SeDuMi_1_21/';


j=length(K); % the number of kernels
n=size(K{1},2); % the number of samples
d=size(K{1},1);  % the number of dimensions

meanKernel=zeros(n,n);

for i=1:j
    K{i}=(K{i}+K{i}')/2;
    meanKernel=K{i}+meanKernel;
end
meanKernel=meanKernel/j;



sK = [];

for loop=1:j
  temp= K{loop}*(1-lb) + meanKernel*lb + 1E-6*eye(size(K{loop}));
  sK{loop} = chol(temp);
  clear temp;
end

clear K;

%y = [t alpha], dimension 1+n
% 
% max b'y
% s.t. c-A'y >= 0

% object function: -t
b = [-1+lb ; sparse(n+1,1)];

%size(b)

c = sparse([-1+1e-5;... the sum of alphas constraint: alpha'*e -1 >= 0
        1+1e-5;... the sum of alphas constraint: -alpha'*e +1.00... >= 0
        sparse(n,1);... the lower bounds on alpha: alpha >= 0
        ones(n,1)/(nu*n);... the upper bounds on alpha: 1/(nu*n)-alpha >= 0
	0.5;
	-0.5]);

for loop=1:j
  c = [c;
      0;
      0;
      sparse(d,1)];  % the quadratic R cone constraints

end
	
% Inequality constraints

mAt = sparse([0 0 ones(1,n);... alpha'*e -1 >= 0
        0 0 -ones(1,n);... -alpha'*e +1.00... >= 0
        sparse(n,2) speye(n);... alpha >= 0
        sparse(n,2) -speye(n);... 1/(nu*n)-alpha >= 0
	0 -1 sparse(1,n);
	0 1  sparse(1,n);
        spalloc(j*(d+2),n+2,j*n*d+j*2)]);

mAt=full(mAt);

%quadratic cone constraints

for i=1:j
    mAt(4+2*n+(i-1)*(d+2)+1,1)=1;
    mAt(4+2*n+(i-1)*(d+2)+2,2)=1;
    mAt(4+2*n+(i-1)*(d+2)+3:4+2*n+i*(d+2),3:n+2)=sK{i};
end

c = full(c);

mAt = sparse(mAt);

K.l = [4+2*n];
K.r = ones(1,j)*(d+2);
%cd 'C:\matlab\toolbox\SDPT3-3.02'; A=-mAt'; save sedumiformat.mat A c b K; startup; [blk,AA,CC,b] = read_sedumi('sedumiformat.mat'); [obj,primalsol,dualsol,Z] = sqlp(blk,AA,CC,b);
pars.fid=1;

tic

[primalsol, dualsol, info] = sedumi(-mAt',b,c,K,pars);

spendtime = toc;

primalsol;

dualsol;

t=dualsol(1);
alpha=dualsol(3:end);
%sum(alpha)
%theta=primalsol{2}([1:n+1:(j-1)*(n+1)+1]);
theta=primalsol(4+2*n+[1:d+2:(j-1)*(d+2)+1]);
theta=theta+lb/j;

