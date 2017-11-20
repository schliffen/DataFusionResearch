function  [obj,beta,t,theta,stq] =qoqc_LSSVM_multiclass_Linf_regularized(Kmix,L,lambda,lb)

% an yet inefficient solution to solve the Linf-norm LSSVM MKL problem as QCQP in MOSEK
% 
% input variables
% Kmix is the cell array of multiple kernels
% L is the matrix of labels, for multi-class case, the class number is equal to the column number L
% lambda is the reguralization parameter
%
% output variables:
% obj is the optimum of the objective function
% B is the matrix of dual variables
% t is the dummy variable in optimization
% mu are the kernel coefficients
% stq is the costed CPU time



% the bias term can be solved independently after B is obtained

% code written by Shi Yu shi.yu@esat.kuleuven.be  shee.yu@gmail.com
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


% read how many kernels
numP = size(Kmix,2);

% read how many clusters
numK = size(L,2);

% read how many data
numD = size(Kmix{1},1);

% regularize the kernel with lowerbound
meanKernel=zeros(numD,numD);
for i=1:1:numP
    Kmix{i}=(Kmix{i}+Kmix{i}')/2;
    Kmix{i}=Kmix{i}+eye(size(Kmix{i},1))*1e-6;
    meanKernel=Kmix{i}+meanKernel;
end
meanKernel=meanKernel/numP;

for i=1:1:numP
    Kmix{i}=Kmix{i}*(1-lb)+meanKernel*lb;
end


% creat cluster indicator matrix L from partition matrix P
%L = P*(P'*P)^-0.5;



%variables
% B_1 B_2 ,..., B_K, t


% init the mosek problem
clear prob;


% c vector .
c = -ones(numD*numK+1,1);
c(end)=0.5;
prob.c =c;
clear c;


% the quadratic term in objective function

prob.qosubi = [];
prob.qosubj = [];
prob.qoval = [];


for kernelLoop = 1:1:numK
	
	prob.qosubi = [prob.qosubi numD*(kernelLoop-1)+1:numD*(kernelLoop-1)+numD];
	prob.qosubj = [prob.qosubj numD*(kernelLoop-1)+1:numD*(kernelLoop-1)+numD];
	prob.qoval =  [prob.qoval 1/lambda*ones(1,numD)];
end
prob.qosubi = [prob.qosubi numK*numD+1]';
prob.qosubj = [prob.qosubj numK*numD+1]';
prob.qoval = [prob.qoval 0]';


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

mu = res.sol.itr.suc(1:numP)*2;

for loop=1:1:size(mu,1)
   theta(loop) = (1-lb)*mu(loop)+lb/size(mu,1);
end

obj = 0.5 * t + 0.5/lambda* (trace(beta'*beta))- sum(sum(beta));