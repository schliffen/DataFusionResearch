function  [obj,alpha,t,theta,b] =qoqc_weighted_classicalSVM_binary(Kmix,L,C)

% multiple class MKL for Vapnik's SVM
% Kmix, a cell array object Kernel matrices
% L,    a matrix of labels, the rows are samples, the columns are classes
% C, the box constraint parameter, normally C = 1
% the program requires mosek, and it solves the problem as QCQP
% Note: this code only accept binary classification, for multi-class classification, you need to change the code to adopt different w vectors
%
%
% programmed by Shi Yu 2009, shee.yu@gmail.com


% read how many kernels
p = size(Kmix,2);

% read how many clusters
k = size(L,2);

% read how many data
N = size(Kmix{1},1);

% count how many positive class L+ and L-
lpos = length(find(L==1));
lneg = length(find(L==-1));

wpos = lpos/N;
wneg = lneg/N;

% w is the weight vector
w = ones(N,1);
w(find(L==1))=wpos;
w(find(L==-1))=wneg;


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

% upperbound of Support Vectors
prob.bux = repmat(w.*C,k,1);


prob.bux = [prob.bux; +inf];

yacc =[];
for clusterLoop = 1:1:k
	yacc= [yacc; L(:,clusterLoop)];
end
yacc =[yacc; 0];


for kernelLoop = 1:1:p
    Qk=[];

    for clusterLoop = 1:1:k
	clusterLoop
	Y = diag(L(:,clusterLoop));
	Qk=[Qk;sparse(N,(clusterLoop-1)*N) 2*Y*Kmix{kernelLoop}*Y sparse(N,(k-clusterLoop)*N+1)];
    end

    Qk=[Qk; sparse(1,N*k+1)];

	trQk = tril(Qk);
	trQk = sparse(trQk);
	[i,j,s] = find(trQk);	
	
	clear Qk;

	prob.qcsubk = [prob.qcsubk; kernelLoop*ones(size(i))];
	prob.qcsubi = [prob.qcsubi i'];
	prob.qcsubj = [prob.qcsubj j'];
	prob.qcval =  [prob.qcval s'];

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
 alpha = [alpha beta((loop-1)*N+1:loop*N)];
end

t = res.sol.itr.xx(end);

theta = res.sol.itr.suc(1:p);


obj = 0.5*t - sum(beta);

%combining kernels
Kcombine = zeros(N,N);
for loop=1:1:p
	Kcombine = Kcombine + Kmix{loop}.*theta(loop);
end

% cacluating b
b=[];

for loop=1:1:k
    L0 = L(:,loop);
    beta0= alpha(:,loop);

    H = diag(L0)*Kcombine*diag(L0);
    svi = find( beta0 > 5E-8);
	svii = find(beta0 > 5E-8 & beta0 < (w.*C - 5E-8));
	if length(svii) > 0
		b0 =  (1/length(svii))*sum(L0(svii) - H(svii,svi)*beta0(svi).*L0(svii));
          else 
		fprintf('No support vectors on margin - cannot compute bias.\n');
		b0 = 0;
        end

 b=[b b0];
end



