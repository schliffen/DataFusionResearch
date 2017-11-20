function [alpha, theta, obj,b] =  socp_weighted_classicalSVM_sedumi_L2_binary(Kmix,L,C)

% k is the number of kernels
% n is the number of samples
% L is the regression or label vector
% p is the number of classes
% the program solves L2-norm SVM MKL as socp, it requires sedumi package to excute
% Note: this code only accept binary classification, for multi-class classification, you need to change the code to adopt different w vectors

% programmed by Shi Yu  shee.yu@gmail.com


p = length(Kmix);
N = size(Kmix{1},1);
k = size(L,2);

for loop=1:p
  Chol{loop} = chol(Kmix{loop}+0.00001.*eye(N,N))';
end

yarr = zeros(k,k*N);


for loop=1:1:k
	yarr(loop,(loop-1)*N+1:loop*N) = L(:,loop)';
end

% count how many positive class L+ and L-
lpos = length(find(L==1));
lneg = length(find(L==-1));

wpos = lpos/N;
wneg = lneg/N;

% w is the weight vector
w = ones(N,1);
w(find(L==1))=wpos;
w(find(L==-1))=wneg;


% max b'y
% s.t. c-A'y >= 0

% object function max: -t/2 + sum(beta)


% variable:  [t s1 s2 ... sk 0.5 beta_class1(1...n)  beta_classp(1...n)]
b = [-0.5; sparse(p+1,1); ones(N*k,1)];


% second order cone constraints 
mAt = [sparse(k*N,2+p)  speye(k*N,k*N);   %beta_i <= sC
       sparse(k*N,2+p)  -speye(k*N,k*N);  %beta_i >= 0
       sparse(k,2+p)    yarr;              %-a'*beta >=0
       sparse(k,2+p)    -yarr;             % a'*beta <=0
       sparse(1,p+1) 1  sparse(1,N*k);
       sparse(1,p+1) -1 sparse(1,N*k);
       -speye(p+1,p+1) sparse(p+1,k*N+1);];

%mAt = sparse(mAt);

for kernelnum = 1:1:p
	mAt = [mAt;
	 0 sparse(1,kernelnum-1) -1 sparse(1,p-kernelnum)  sparse(1,1+N*k);
	 sparse(1,p+1)      -1 sparse(1,N*k);];

	for loop=1:1:k
		 mAt = [mAt;
		 sparse(N,p+2) sparse(N,(loop-1)*N) -Chol{kernelnum}'*diag(L(:,loop)) sparse(N,(k-loop)*N);
		 ];
	end
end

%full(mAt)

mAt = sparse(mAt);




c= [repmat(w.*C,k,1);
	sparse(N*k,1);
	sparse(2*k,1);
	0.5;
	-0.5;
	sparse(p+1,1);
	sparse(p*(k*N+2),1)
	];

c = sparse(c);
%full(c)

K.l = [2*(N*k)+2*k+2];
K.q = p+1;
K.r = [(N*k+2)*ones(1,p)];

pars.fid=1;

%tic

[primalsol, dualsol, info] = sedumi(mAt',b,c,K,pars);

%st = toc

st = dualsol;

obj = b'*dualsol;


alpha=[];

for loop=1:1:k
	alpha = [alpha dualsol(p+2+(loop-1)*N+1:p+2+loop*N)];
end

theta1 = primalsol(2*N+8);
theta2 = primalsol(3*N+10);

theta =primalsol(2*k*N+2*k+2+p+1+1:k*N+2:end);
theta = theta.*2;


%combining kernels
Kcombine = zeros(N,N);
for loop=1:1:p
	Kcombine = Kcombine + Kmix{loop}.*theta(loop);
end

b=[];
for loop=1:1:k
    L0 = L(:,loop);
    beta0= alpha(:,loop);

    H = diag(L0)*Kcombine*diag(L0);
    svi = find( beta0 > 5E-8);
	svii = find(beta0 > 5E-8 & beta0 < (C - 5E-8));
	if length(svii) > 0
		b0 =  (1/length(svii))*sum(L0(svii) - H(svii,svi)*beta0(svi).*L0(svii));
          else 
		fprintf('No support vectors on margin - cannot compute bias.\n');
		b0 = 0;
        end

 b=[b b0];
end