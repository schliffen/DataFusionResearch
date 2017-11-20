function  [obj,B,t,munew,b] =qoqc_classicalSVM_multiclass_regularized(Kmix,L,sC,lb)

% multiple class MKL for Vapnik's SVM based on QCQP formulation
% Kmix, a cell array object Kernel matrices
% L,    a matrix of labels, the rows are samples, the columns are classes
% sC, the box constraint parameter, normally sC = 1
% lb, the lowerbound parameter (theta_min), lb should be positive values from 0 to 1, when lb = 1, the kernels in Kmix are averagely combined
% the program requires mosek, and it solves the problem as QCQP
%
%
% programmed by Shi Yu 2009  shi.yu@esat.kuleuven.be

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
prob.blx = sparse(numD*numK+1,1);
prob.bux = sC*ones(numD*numK,1);
prob.bux = [prob.bux; +inf];

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

beta = res.sol.itr.xx(1:numK*numD);


B = [];

for loop=1:1:numK
 B = [B beta((loop-1)*numD+1:loop*numD)];
end

t = res.sol.itr.xx(end);

mu = res.sol.itr.suc(1:numP);

for loop=1:1:size(mu,1)
   munew(loop) = (1-lb)*mu(loop)+lb/size(mu,1);
end


obj = 0.5*t - sum(beta);

%combining kernels
Kcombine = zeros(numD,numD);
for loop=1:1:numP
	Kcombine = Kcombine + Kmix{loop}.*mu(loop);
end

% cacluating b
b=[];

for loop=1:1:numK
    L0 = L(:,loop);
    beta0= B(:,loop);
    H = diag(L0)*Kcombine*diag(L0);
    
    svi = find( beta0 > 5E-8);
	svii = find(beta0 > 5E-8 & beta0 < (sC - 5E-8));
	if length(svii) > 0
		b0 =  (1/length(svii))*sum(L0(svii) - H(svii,svi)*beta0(svi).*L0(svii));
          else 
		fprintf('No support vectors on margin - cannot compute bias.\n');
		b0 = 0;
        end

 b=[b b0];
end



