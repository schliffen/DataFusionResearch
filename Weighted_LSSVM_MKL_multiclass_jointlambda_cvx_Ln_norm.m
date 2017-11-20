function [beta, theta, obj, lambda, w] =  Weighted_LSSVM_MKL_multiclass_jointlambda_cvx_Ln_norm(Kmix,L,n_norm)

% an yet inefficient solution to solve the L2-norm LSSVM MKL problem using the CVX tool (binary class only...), the regularization term lambda is estimated jointly as a coefficient to the identity matrix
% 
% input variables
% Kmix is the cell array of multiple kernels
% a is the matrix of labels, for multi-class case, the class number is equal to the column number L
% kappa is the norm in the primal problem
%
% output variables:
% obj is the optimum of the objective function
% beta is the matrix of dual variables
% lambda is the estimated regularization term
% theta are the kernel coefficients
% Note: this code only applies for binary classification problem

% the bias term can be solved independently after beta is obtained

% code written by Shi Yu shee.yu@gmail.com, March 2010 
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


%p is the number of kernels
%N is the number of samples
%L is the regression or label vector

p = length(Kmix);
N = size(Kmix{1},1);

Y = diag(L);

for i=1:1:p
 Kmixnew{i} = Y*Kmix{i}*Y + 0.00001*eye(N,N);
end


% count how many positive class L+ and L-
lpos = length(find(L==1));
lneg = length(find(L==-1));

wpos = 2*lpos/N;
wneg = 2*lneg/N;

w = ones(N,1);
w(find(L==1))=wpos;
w(find(L==-1))=wneg;

clear wpos wneg lpos lneg;

W = diag(w);

Kmixnew{p+1} = W;

cvx_begin
   variables t beta(N) s(p+1)
   dual variable thetatemp{p+1}
   minimize(.5*t  - sum(beta))
   subject to
     beta'*L == 0
     for kk=1:p+1
       thetatemp{kk}: beta' * Kmixnew{kk} * beta <= s(kk)
     end
     norm(s,n_norm) <= t
cvx_end

obj = cvx_optval;
thetatemp = cell2mat(thetatemp);
theta = thetatemp(1:p+1).*2;
lambda = 1/(theta(p+1).*n_norm);

