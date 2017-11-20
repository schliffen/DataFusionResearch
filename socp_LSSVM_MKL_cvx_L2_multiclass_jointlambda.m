function [beta, theta, obj, lambda] =  socp_LSSVM_MKL_cvx_L2_multiclass_jointlambda(Kmix,L)

% an yet inefficient solution to solve the L2-norm LSSVM MKL problem using the CVX tool (binary class only...), the regularization term lambda is estimated jointly as a coefficient to the identity matrix
% 
% input variables
% Kmix is the cell array of multiple kernels
% L is the matrix of labels, for multi-class case, the class number is equal to the column number L
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


%k is the number of kernels
%n is the number of samples
%a is the regression or label vector

p = length(Kmix);
N = size(Kmix{1},1);
k = size(L,2);

% a set of labels
Ymix =[];
for loop=1:1:size(L,2)
	Y = diag(L(:,loop));
	Ymix=[Ymix {Y}];
end

for i=1:1:p
 Kmix{i} = Kmix{i} + 0.00001*eye(N,N);
end

Kmix{p+1} = eye(N,N);



cvx_begin
   variables t beta(N,k) s(p+1)
   dual variable theta{p+1}
   % beta is a n x m matrix
   minimize(.5*t  - sum(sum(beta)))
   subject to
     for mm=1:k
      beta(:,mm)'*L(:,mm) == 0
     end
     for kk=1:p
       sumk = 0;
       for mm=1:k
	  sumk = sumk + beta(:,mm)'*Ymix{mm}*Kmix{kk}*Ymix{mm}*beta(:,mm);
       end
        theta{kk}: sumk <= s(kk)
     end
     sumk = 0;
     for mm=1:k
	sumk = sumk + beta(:,mm)'*eye(N,N)*beta(:,mm);
     end
     theta{p+1}: sumk<=s(p+1)

     norm(s) <= t
cvx_end

obj = cvx_optval;
theta = cell2mat(theta);
mu = theta(1:p+1).*2;
lambda = 1/(theta(k+1).*2);
theta = mu;

