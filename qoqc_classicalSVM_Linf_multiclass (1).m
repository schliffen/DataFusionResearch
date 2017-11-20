function  [obj,alpha,theta,b] =qoqc_classicalSVM_Linf_multiclass(Kmix,L,C)

% multiple class MKL for Vapnik's SVM
% Kmix, a cell array object Kernel matrices
% L,    a matrix of labels, the rows are samples, the columns are classes
% C, the box constraint parameter, normally C = 1
% the program requires mosek, and it solves the problem as QCQP
%
%
% programmed by Shi Yu shee.yu@gmail.com  June 2009


% read how many kernels
j = size(Kmix,2);

% read how many clusters
k = size(L,2);

% read how many data samples
N = size(Kmix{1},1);

% creat cluster indicator matrix L from partition matrix P
%L = P*(P'*P)^-0.5;


% init the mosek problem
clear prob;


% c vector .
c = -ones(N*k+1,1);
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
prob.blx = sparse(N*k+1,1);
prob.bux = C*ones(N*k,1);
prob.bux = [prob.bux; +inf];

yacc =[];
for clusterLoop = 1:1:k
	    yacc= [yacc; L(:,clusterLoop)];
end
yacc =[yacc; 0];


for kernelLoop = 1:1:j
    Qk=[];

    for clusterLoop = 1:1:k
	    clusterLoop
	    Y = diag(L(:,clusterLoop));
	    Qk=[Qk;sparse(N,(clusterLoop-1)*N) 2*Y*Kmix{kernelLoop}*Y sparse(N,(k-clusterLoop)*N+1)];
    end

    Qk=[Qk; sparse(1,N*k+1)];

	trQk = tril(Qk);
	trQk = sparse(trQk);
	[id,jd,sd] = find(trQk);	
	
	clear Qk;

	prob.qcsubk = [prob.qcsubk; kernelLoop*ones(size(id))];
	prob.qcsubi = [prob.qcsubi id'];
	prob.qcsubj = [prob.qcsubj jd'];
	prob.qcval =  [prob.qcval sd'];

	prob.a = [prob.a; sparse(1,k*N) -1];   %  a'*Y*K*Y*a - 2*gamma <=0
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

beta = res.sol.itr.xx(1:k*N);


alpha = [];

for loop=1:1:k
 alpha = [alpha  beta((loop-1)*N+1:loop*N)];
end

t = res.sol.itr.xx(end);

theta = res.sol.itr.suc(1:j);

obj = 0.5*t - sum(beta);

%combining kernels
Kcombine = zeros(N,N);
for loop=1:1:j
	Kcombine = Kcombine + Kmix{loop}.*theta(loop);
end

% cacluating b
b=[];

for loop=1:1:k
    L0 = L(:,loop);
    beta0= alpha(:,loop);
   
    svi = find(beta0 > 5E-8);
	svii = find(beta0 > 5E-8 & beta0 < (C - 5E-8));
	if length(svii) > 0
		b0 =  (1/length(svii))*sum(L0(svii) - Kcombine(svii,svi)*beta0(svi).*L0(svii));
          else 
		fprintf('No support vectors on margin - cannot compute bias.\n');
		b0 = 0;
        end

 b=[b b0];
end



