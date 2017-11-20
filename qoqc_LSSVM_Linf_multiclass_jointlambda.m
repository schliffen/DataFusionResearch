function  [obj,beta,t,theta,lambda] =qoqc_LSSVM_Linf_multiclass_jointlambda(Kmix,L)

% an yet inefficient solution to solve the Linf-norm LSSVM MKL problem as QCQP in MOSEK, the regularization term lambda is estimated jointly as a coefficient to the identity matrix
% 
% input variables
% Kmix is the cell array of multiple kernels
% L is the matrix of labels, for multi-class case, the class number is equal to the column number L
% 
%
% output variables:
% obj is the optimum of the objective function
% beta is the dual variables of LSSVM
% t is the dummy variable in optimization
% theta are the kernel coefficients
% stq is the costed CPU time
% the automatically estimated lambda value is the last coefficents in theta

% the bias term can be solved independently after B is obtained


% code written by Shi Yu shi.yu@esat.kuleuven.be shee.yu@gmail.com
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


% read how many data
numD = size(Kmix{1},1);

dummK = eye(numD,numD);

Kmix = [Kmix dummK];

% read how many kernels
numP = size(Kmix,2);

% read how many clusters
numK = size(L,2);


% creat cluster indicator matrix L from partition matrix P
%L = P*(P'*P)^-0.5;


% init the mosek problem
clear prob;


% c vector .
c = -ones(numD*numK+1,1);
c(end)=1;
prob.c =c;
clear c;




% Next quadratic terms in the constraints .
% constraint of 1st kernel
prob.qcsubk =[];
prob.qcsubi =[];
prob.qcsubj = [];
prob.qcval =[];
prob.a = [];
prob.buc = [];
prob.blc=[];
prob.blx = -inf*ones(numD*numK+1,1);  % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)
prob.bux = +inf*ones(numD*numK,1);     % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)
prob.bux = [prob.bux; +inf];  % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)

yacc =[];
for clusterLoop = 1:1:numK
	    yacc= [yacc; L(:,clusterLoop)];
end
yacc =[yacc; 0];


for kernelLoop = 1:1:numP
    Qk=[];

    for clusterLoop = 1:1:numK
	    clusterLoop
	    Y = diag(L(:,clusterLoop));
	    Qk=[Qk;sparse(numD,(clusterLoop-1)*numD) 2*Y*Kmix{kernelLoop}*Y sparse(numD,(numK-clusterLoop)*numD+1)];
    end

    Qk=[Qk; sparse(1,numD*numK+1)];

	trQk = tril(Qk);
	trQk = sparse(trQk);
	[i,j,s] = find(trQk);	
	
	clear Qk;

	prob.qcsubk = [prob.qcsubk; kernelLoop*ones(size(i))];
	prob.qcsubi = [prob.qcsubi i'];
	prob.qcsubj = [prob.qcsubj j'];
	prob.qcval =  [prob.qcval s'];

	prob.a = [prob.a; sparse(1,numK*numD) -1];   %  a'*Y*K*Y*a - 2*gamma <=0
	prob.buc = [prob.buc; 0];
        prob.blc = [prob.blc; -inf];


end


% linear constraints  (Ya)'*\vec 1 = 0

size(yacc')
size(prob.a)
prob.a = [prob.a; yacc'];

prob.buc = [prob.buc; 0];
prob.blc = [prob.blc; 0];

prob.qcsubi = prob.qcsubi';
prob.qcsubj = prob.qcsubj';
prob.qcval = prob.qcval';

size(prob.c)
size(prob.qcsubi)

tic;
[r,res] = mosekopt ('minimize', prob);
stq =toc;

% get the optimal dual variable solution

size(res.sol.itr.xx)

betatemp = res.sol.itr.xx(1:numK*numD);


beta = [];

for loop=1:1:numK
 beta = [beta betatemp((loop-1)*numD+1:loop*numD)];
end

t = res.sol.itr.xx(end);

theta = res.sol.itr.suc(1:numP);

beta = beta.*2;

obj = 0.5*t - sum(beta);

lambda = 1/theta(end);