function [theta,alpha,S,stq]=sip_classicalSVM_Linf_multiclass(K,L,C)

% SIP for the Linf-norm multiple class standard MKL solver
% kernels  a cell object of multiple centered kernels
% A is a nxk matrix, where n is the number of data samples, k is the number of classes
% lambda is the regularization parameter

% output: theta  as  the kernel coefficients
%          beta      as  the multiple dual variables stored in the matrix
%	  S      dummy variable to check the convergence
%	  stq      the CPU time costed 
%
%
% Notice:
% The program is for multi-class Vapnik's SVM MKL Linf-norm fusion
% it calls the solve_qcqp_mosek function (self-included) to solve the single kernel SVM, which solves the problem using MOSEK as QCQP
% to solve the Linf-norm optimization problem, it uses the Matlab function "linprog" and solves it as LP, ("solve_lp" function)
% 
% Coded by Shi Yu shee.yu@gmail.com and Tillmann Falck tillmann.falck@esat.kuleuven.be, 2009 
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


N = length(K{1});
j = size(K,2);
k = size(L,2);

time0 = cputime;
alpha = zeros(N,k)/2;
%alpha = rands(N,1)/2;
sis = compute_sis(K, L, alpha);

E=[];

for n=1:40
    [theta,gamma] = solve_lp(sis);
    theta
    alpha = solve_qcqp_mosek(K, L, C, theta);
    
    S = compute_sis(K, L, alpha);
    sis = [sis; S]
    temp = 0;

    for loop = 1:j
	temp = temp+ theta(loop)*S(loop);
    end

    
    eps = 1+temp/gamma;
    E= [E eps];
    if length(E)>=3 && abs(E(end)-E(end-1))<5e-6
	break
    end
end

stq =cputime-time0;
disp(n)


function sis=compute_sis(K, L, alpha)
j = length(K);
k = size(L,2);  % size of classes
sis = zeros(1,j);

%c = - sum(B);

for loop1=1:j
   for loop2 = 1:1:k
	sis(loop1) = 0.5 * alpha(:,loop2)' *diag(L(:,loop2))* K{loop1} * diag(L(:,loop2))*alpha(:,loop2) - sum(alpha(:,loop2));
	
   end
end


function [theta,gamma]=solve_lp(Sis)
S=size(Sis);

[theta,gamma] = linprog([-1, zeros(1,S(2))], [ones(S(1), 1), -Sis], zeros(S(1),1), [0, ones(1,S(2))], 1, [-inf, zeros(1,S(2))]);
theta(1) = [];



% this is a corrected version, instead of solving ls, we solve the qs by MOSEK!
function  [alpha] =solve_qcqp_mosek(K,L,C,theta)

% solves the qp problem on single kernel (given the optimal kernel) to obtain the beta
% the beta is then iterated in the SILP procedure

% K is the given optimal kernel
% L is the class label matrix, the column number k corrresponds to the number of classes
% C is the box constraint

% always one kernel
j = 1;

% read how many clusters
k = size(L,2);

% read how many data
N = size(K{1},1);


% recompose the optimal kernel
G = zeros(size(K{1}));
for loop=1:length(K)
    G = G + theta(loop) * K{loop};
end


% from now on we only need G
clear K theta;

% call the QCQP solver
[obj,alpha,t,dummy,b] =qoqc_classicalSVM_multiclass([{G}],L,C);
