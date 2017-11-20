function [theta,alpha,E,Sis,time]=sip_classicalSVM_multiclass_cvx_Lm_norm(Kmix,L,C,m_norm)

% SIP for multiple class standard MKL solver
% Kmix  a cell object of multiple centered kernels
% L is a nxk matrix, where n is the number of data samples, k is the number of classes
%
% Notice:
% The program is for SVM MKL
% 
% Coded by Shi Yu shee.yu@gmail.com March 2010 


time0 = cputime;

N = length(Kmix{1});
k = size(L,2);  % size of classes

alpha = randn(N,k)/2;
%alpha = zeros(N,k)/2;
Sis = compute_sis(Kmix, L, alpha);

E=[];

for n=1:1:100
    disp(n)
    [theta,gamma] = solve_theta_relax_cvx(alpha,Sis,m_norm);
    [Km] = compose_kernels(Kmix,theta);
    Kmerge =[{Km}];
    [obj,alpha,thetatemp,b] =qoqc_classicalSVM_Linf_multiclass(Kmerge,L,C)
    S = compute_sis(Kmix, L, alpha);
    Sis = [Sis; S]
    temp = compute_sis(Kmerge, L, alpha) - sum(sum(alpha));
    eps = 1-temp/gamma;
    E= [E eps];
    if length(E)>=3 && abs(E(end)-E(end-1))<5e-6
    	break
    end
end

disp(n)

time = cputime - time0;


function sis=compute_sis(kernels, L, alpha)
p = length(kernels);
k = size(L,2);  % size of classes
sis = zeros(1,p);

%c = - sum(B);

for n=1:p
   for j = 1:1:k
	sis(n) = 0.5 * alpha(:,j)' *diag(L(:,j))* kernels{n} * diag(L(:,j))*alpha(:,j);
   end
end



function [cK] = compose_kernels(kernels,theta)

cK = zeros(size(kernels{1}));
for n=1:1:length(kernels)
	cK = cK + theta(n)*kernels{n};
end
