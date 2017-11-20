function [t,theta,alpha] = one_class_Kinput_cvx_Ln_norm(Kmix,nu,n_norm)

%function [t,theta,alpha] = one_class_Kinput_cvx_Ln_norm(Kmix,nu,n_norm)
%
% Learn the optimal kernel by maximising the distance between the data
% points and the origin. Here a positive lower bound on the kernel weights
% mu can be specified as well.
% this is different from Tijl De Bie's method which is based on L1 norm
% 
%
%INPUTS:
% Kmix    = a struct array containing the kernels
% nu      = the fraction of outliers tolerated
% lb      = a lower bound on the weights mu, multiplied by the number of
%           kernels. Should be in (0,1): 0 means the SDP method, 1 means
%           uniform weighting. It is the fraction of the total weight that
%           must be spread equally over all kernels.
% n_norm... the norm rank of the dual problem
%
%OUTPUTS:
% t     = the objective (margin or so...)
% theta = the weights of the kernels
% alpha = the dual variables for the 1-SVM
%
%Author: Shi Yu,     shee.yu@gmail.com March 2010.




p=length(Kmix); % the number of kernels
N=size(Kmix{1},2); % the number of samples


for i=1:1:p
 Kmix{i} = Kmix{i} + 0.00001*eye(N,N);
end


cvx_begin
   variables t s(p) alpha(N);
   dual variable theta{p};
   minimize(t)
   subject to
     for kk=1:p
	theta{kk}: s(kk) >= alpha'*Kmix{kk}*alpha
     end
     t>=norm(s,n_norm)
     for kk=1:N
	alpha(kk)>=0;
	alpha(kk)<=1/(nu*N);
     end
     sum(alpha) == 1
cvx_end

theta = cell2mat(theta);

