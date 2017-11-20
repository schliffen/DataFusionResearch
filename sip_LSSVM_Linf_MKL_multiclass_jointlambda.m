function [theta,beta,E,sis,time]=sip_LSSVM_Linf_MKL_multiclass_jointlambda(Kmix,L)

time0 = cputime;

% SLP for multiple class Linf-norm LSSVM MKL solver
% kernels  a cell object of multiple centered kernels
% A is a nxk matrix of labels, where n is the number of data samples, 
% k is the number of classes, notice that this version requires that A(:,j)^-2 = ones, for all j
%
% Output variables 
% theta:  kernel coefficients
% beta:      the dual variables 
% E:      the dummy variable checking the covergence
% sis:    the dummy variable of f(alpha)
% time:      the costed CPU time
%

% Notice:
% The program is for L_inf LS-SVM MKL with joint estimated lambda (regularization parameter) 
% the single LSSVM is solved by the included function linsolve_LSSVM
% the Linf-norm is optimized by the  solve_lp function based on Matlab "linprog" function

% Coded by Shi Yu shi.yu@esat.kuleuven.be and Tillmann Falck tillmann.falck@esat.kuleuven.be, 2009 

time0 = cputime;

N = length(Kmix{1});
k = size(L,2);  % size of classes


% add an identity matrix in the kernel fusion
Kmix = [Kmix {eye(N,N)}];

beta = rands(N,k)/2;
sis = compute_sis(Kmix, L, beta);

E=[];

for n=1:1:100
    disp(n)
    [theta,gamma] = solve_lp(sis);
    [trash1, B, trash2, trash3] = linsolve_LSSVM(Kmix, L, theta);
    S = compute_sis(Kmix, L, B);
    sis = [sis; S]
    eps = 1+S*theta/gamma;

    E = [E eps];
    if abs(eps) < 5e-4
        time = cputime - time0;
        break
    end
end

disp(n)

time = cputime - time0;

function sis=compute_sis(Kmix, L, beta)

p = length(Kmix);
k = size(L,2);  % size of classes
sis = zeros(1,p);
c = - sum(sum(beta));

for n=1:p-1
	sis(n)= c;
   for j = 1:1:k
	sis(n) = sis(n) + 0.5 * beta(:,j)' *diag(L(:,j))* Kmix{n} * diag(L(:,j))*beta(:,j);
   end
end

sis(p) = c;
for j=1:1:k
  sis(p) = sis(p) + 0.5 * beta(:,j)'*Kmix{p}*beta(:,j);
end

function [theta,gamma]=solve_lp(Sis)
S=size(Sis);

[theta,gamma] = linprog([-1, zeros(1,S(2))], [ones(S(1), 1), -Sis], zeros(S(1),1), [0, ones(1,S(2))], 1, [-inf, zeros(1,S(2))]);
theta(1) = [];

function  [obj,alpha,beta,b] =linsolve_LSSVM(K,L,theta)
% solve LS-SVM as linear problem, the formulation adapted here is different from the formulation given by Suykens et al.
G = zeros(size(K{1}));
for n=1:length(K)
    G = G + theta(n) * K{n};
end
n = size(G,1);  % number of data
p = size(L,2);  % number of classes

for loop = 1:1:p
	Y{loop} = diag(L(:,loop));
end
onevec = ones(size(L(:,1)));
H =[0 onevec';onevec G];

J = [zeros(1,p); 1./L];
sol = linsolve(H,J);
beta = sol(2:end,:);
b = sol(1,:);

alpha = zeros(size(beta));
for loop=1:1:p
	alpha(:,loop) = inv(Y{loop})*beta(:,loop);
end
obj=0;
