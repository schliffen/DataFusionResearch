function [t,alpha,theta,spendtime] =  qclp_oneclass_Linf_regularized(K,nu,lb)

% one SVM fusion using mosek as solver, formulated as QCLP problem
%Author: Shi Yu,      June 2009.

% add the path of MOSEK toolbox
addpath 'C:\Program Files\Mosek\5\toolbox\r2007a\';
%addpath '~/no_backup/fullgenomic/mosek/mosek/5/toolbox/r2007a/';


j=length(K); % the number of kernels
n=size(K{1},1); % the number of samples

meanKernel=zeros(n,n);

for i=1:j
    K{i}=(K{i}+K{i}')/2;
    K{i}=K{i}+eye(size(K{i},1))*1e-6;
    meanKernel=K{i}+meanKernel;
end
meanKernel=meanKernel/j;


sK=[];

for i=1:j
    temp = chol(K{i}*(1-lb)+meanKernel*lb);
    sK{i}=temp(1:n,:);
    clear temp;
    K{i}=[];
end
clear Kernels

clear prob;




%the size of x is 1+ n + d*k
r = 1+n+n*j;


c = sparse(r,1);
c(1) = 1;  % minimize t

prob.c =c;

clear c;

% Next quadratic terms in the constraints .
% constraint of 1st kernel

prob.a = [];
prob.buc = [];
prob.blc = [];


opri=[];
% j number of constraints   of   -beta'*beta + t >=0
for loop = 1:1:j
    prob.a = [prob.a; -1 sparse([zeros(1,r-1)])];%   t
    prob.buc = [prob.buc; 0];
    prob.blc = [prob.blc; -inf];
    opri = [opri; loop*ones(n,1)];  % decide which constraints the qudractic functions are
end



oprf= +ones(j*n,1);   % all these functions are - x^2 +t >=0 
oprj = [n+1+1: n+1+j*n]';   % the index of variables    t  a1  an  --> beta1(1)...betam(1) beta1(2)...betam(2) 
oprg = 2 * ones(j*n,1);   % all power 2
charcode = [112*ones(j*n,1) 111*ones(j*n,1) 119*ones(j*n,1)];
opr = char(charcode);  % all in ['pow']
clear charcode;

% separable parameters done!


for loop =1:1:j
 vert = zeros(n,r);
 vert (:,2:n+1) = sK{loop};
 vert(:,1+n+(loop-1)*n+1:1+n+loop*n) = -eye(n,n);
 vert = sparse(vert);
 prob.a = [prob.a; vert];
 prob.buc = [prob.buc; sparse(n,1)];
 prob.blc = [prob.blc; sparse(n,1)];

 clear vert;
end


prob.a = [prob.a; 0 ones(1,n) sparse(1,r-n-1)];
prob.buc = [prob.buc; 1];
prob.blc = [prob.blc; 1];




prob.blx = [sparse(1,n+1)  -inf*ones(1,r-n-1)]';
prob.bux = [inf (1/(nu*n))*ones(1,n) +inf*ones(1,r-n-1)]';

prob.bux

prob.a = sparse(prob.a);
prob.buc = sparse(prob.buc);
prob.blc = sparse(prob.blc);


tic

%[r,res] = mosekopt ('minimize', prob);

[res]= mskscopt(opr,opri,oprj,oprf,oprg,prob.c,prob.a,prob.blc,prob.buc,prob.blx,prob.bux); 


spendtime = toc;

theta = res.sol.itr.suc(1:j);

alpha = res.sol.itr.xx(2:n+1);
t = res.sol.itr.xx(1);