function [beta, theta, obj, lambda, w] =  socp_weighted_LSSVM_MKL_cvx_L2_binary_jointlambda(Kmix,L)

% an yet inefficient solution to solve the L2-norm LSSVM MKL problem using the CVX tool (binary class only...), the regularization term lambda is estimated jointly as a coefficient to the identity matrix
% 
% input variables
% Kmix is the cell array of multiple kernels
% L is the matrix of labels, for multi-class case, the class number is equal to the column number L
% Note: this code only accept binary classification, for multi-class classification, you need to change the code to adopt different w vectors
% 
%
% output variables:
% obj is the optimum of the objective function
% beta is the matrix of dual variables
% lambda is the estimated regularization term
% theta are the kernel coefficients


% the bias term can be solved independently after beta is obtained

% code written by Shi Yu shi.yu@esat.kuleuven.be  and Tillmann Falck tillmann.falck@esat.kuleuven.be, 2009 
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
   dual variable theta{p+1}
   minimize(.5*t  - sum(beta))
   subject to
     beta'*L == 0
     for kk=1:p+1
       theta{kk}: beta' * Kmixnew{kk} * beta <= s(kk)
     end
     norm(s) <= t
cvx_end

obj = cvx_optval;
theta = cell2mat(theta);
theta = theta(1:p+1).*2;
lambda = 1/(theta(p+1).*2);

