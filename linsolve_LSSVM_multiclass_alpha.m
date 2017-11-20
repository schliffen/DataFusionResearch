function  [alpha,b] =linsolve_LSSVM_multiclass_alpha(K,L,lambda)

% solve LS-SVM as linear problem, the formulation adapted here strictly follows the formulation given by Suykens et al.'s book
% 
%
% input:
% K is the kernel matrix
% L is the matrix with labels, for multi-class problem, the number of columns of L is the class number
% lambda is the regularization term
%
%
% output:
% alpha is the matrix of dual variables 
% b is the solved bias term
% 
% code written by Shi Yu shi.yu@esat.kuleuven.be
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


n = size(K,1);
Y = diag(L);

H =[0 L';L Y*K*Y+1/lambda*eye(n,n)];

J = [0;ones(n,1)];

sol = linsolve(H,J);
alpha = sol(2:end);
b = sol(1);


