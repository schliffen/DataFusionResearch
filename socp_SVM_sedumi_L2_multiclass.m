function [beta, theta, obj,b] =  socp_SVM_sedumi_L2_multiclass(Kmix,L,sC)

% k is the number of kernels
% n is the number of samples
% L is the regression or label vector
% p is the number of classes
% the program solves L2-norm SVM MKL as socp, it requires sedumi package to excute
% 
% programmed by Shi Yu    shee.yu@gmail.com


k = length(Kmix);
n = size(Kmix{1},1);
p = size(L,2);

for loop=1:k
  C{loop} = chol(Kmix{loop}+0.00001.*eye(n,n))';
end

yarr = zeros(p,p*n);


for loop=1:1:p
	yarr(loop,(loop-1)*n+1:loop*n) = L(:,loop)';
end



% max b'y
% s.t. c-A'y >= 0

% object function max: -t/2 + sum(beta)


% variable:  [t s1 s2 ... sk 0.5 beta_class1(1...n)  beta_classp(1...n)]
b = [-0.5; sparse(k+1,1); ones(n*p,1)];


% second order cone constraints 
mAt = [sparse(p*n,2+k)  speye(p*n,p*n);   %beta_i <= \lambda*n
       sparse(p*n,2+k)  -speye(p*n,p*n);  %beta_i >= 0
       sparse(p,2+k)    yarr;              %-a'*beta >=0
       sparse(p,2+k)    -yarr;             % a'*beta <=0
       sparse(1,k+1) 1  sparse(1,n*p);
       sparse(1,k+1) -1 sparse(1,n*p);
       -speye(k+1,k+1) sparse(k+1,p*n+1);];

%mAt = sparse(mAt);

for kernelnum = 1:1:k
	mAt = [mAt;
	 0 sparse(1,kernelnum-1) -1 sparse(1,k-kernelnum)  sparse(1,1+n*p);
	 sparse(1,k+1)      -1 sparse(1,n*p);];

	for loop=1:1:p
		 mAt = [mAt;
		 sparse(n,k+2) sparse(n,(loop-1)*n) -C{kernelnum}'*diag(L(:,loop)) sparse(n,(p-loop)*n);
		 ];
	end
end

%full(mAt)

mAt = sparse(mAt);

c= [sC*ones(n*p,1);
	sparse(n*p,1);
	sparse(2*p,1);
	0.5;
	-0.5;
	sparse(k+1,1);
	sparse(k*(p*n+2),1)
	];

c = sparse(c);
%full(c)

K.l = [2*(n*p)+2*p+2];
K.q = k+1;
K.r = [(n*p+2)*ones(1,k)];

pars.fid=1;

%tic

[primalsol, dualsol, info] = sedumi(mAt',b,c,K,pars);

%st = toc

st = dualsol;

obj = b'*dualsol;


beta=[];

for loop=1:1:p
	beta = [beta dualsol(k+2+(loop-1)*n+1:k+2+loop*n)];
end

%theta1 = primalsol(2*n+8);
%theta2 = primalsol(3*n+10);

theta =primalsol(2*p*n+2*p+2+k+1+1:p*n+2:end);
theta = theta.*2;


%combining kernels
Kcombine = zeros(n,n);
for loop=1:1:k
	Kcombine = Kcombine + Kmix{loop}.*theta(loop);
end

b=[];
for loop=1:1:p
    L0 = L(:,loop);
    beta0= beta(:,loop);
    
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