function  [obj,alpha,b] =linsolve_weighted_LSSVM_multiclass_alpha(K,L,lambda)

% solve LS-SVM as linear problem, the formulation adapted here strictly follows the formulation given by Suykens et al.

n = size(K,1);
Y = diag(L);

% count how many positive class L+ and L-
lpos = length(find(L==1));
lneg = length(find(L==-1));

wpos = 2*lpos/n;
wneg = 2*lneg/n;

w = ones(n,1);
w(find(L==1))=wpos;
w(find(L==-1))=wneg;

clear wpos wneg lpos lneg;

W = diag(w);


H =[0 L';L Y*K*Y+1/lambda*W];

J = [0;ones(n,1)];

sol = linsolve(H,J);
alpha = sol(2:end);
b = sol(1);

obj=0;
