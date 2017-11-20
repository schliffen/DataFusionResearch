function  [alpha,delta,theta,b] =weighted_classicalSVM_multiclass_cvx_Ln_norm(Kmix,L,C,n_norm)

% multiple class MKL for Vapnik's SVM
% Kmix, a cell array object Kernel matrices
% L,    a matrix of labels, the rows are samples, the columns are classes
% C, the box constraint parameter, normally C = 1
% the program requires cvx to solve the n_norm problem 
% n_norm, the rank of norm posed on the dual problem
%
%
% programmed by Shi Yu 2010
% shee.yu@gmail.com


% read how many kernels
numP = size(Kmix,2);

% read how many classes
numK = size(L,2);

% read how many data
numD = size(Kmix{1},1);


% regularize the kernel matrices
for loop=1:1:numP
 Kmix{loop} = Kmix{loop} + 0.00001*eye(numD,numD);
end

% a set of labels
Ymix =[];
for loop=1:1:size(L,2)
	Y = diag(L(:,loop));
	Ymix=[Ymix {Y}];
end

Ymix

cvx_begin
   variables delta alpha(numD,numK) rho(numP)
   dual variable theta{numP}
   % beta is a n x m matrix
   minimize(.5*delta  - sum(sum(alpha)))
   subject to
     for loop=1:numK
        alpha(:,loop)'*L(:,loop) == 0;
     end

     for loop=1:numD  % loop through number of samples 
	for loop2 = 1:numK  % loop through number of classes

	% check the label of the current sample / classs
	lpos = length(find(L(:,loop2)==1));
	lneg = length(find(L(:,loop2)==-1));

	wpos = lpos/numD;
	wneg = lneg/numD;

	   0<=alpha(loop,loop2)
		if L(loop,loop2)==1
		   alpha(loop,loop2)<=wpos*C
		else
		   alpha(loop,loop2)<=wneg*C
		end
	end
     end

     for loop=1:numP
       sumD = 0;
	for loop2=1:numK
	  loop2
	  loop
		sumD = sumD + alpha(:,loop2)'*Ymix{loop2}*Kmix{loop}*Ymix{loop2}*alpha(:,loop2);
	end
	theta{loop}: sumD <= rho(loop)
     end
     norm(rho,n_norm) <= delta
cvx_end

theta = cell2mat(theta)*2;


%combining kernels
Kcombine = zeros(numD,numD);
for loop=1:1:numP
	Kcombine = Kcombine + Kmix{loop}.*theta(loop);
end



% cacluating b
b=[];


tolb = 5E-8;

for loop=1:1:numK
    L0 = L(:,loop);
    beta0= alpha(:,loop);

    H = diag(L0)*Kcombine*diag(L0);
    svi = find( beta0 > tolb);
	svii = find(beta0 > tolb & beta0 < (1 - tolb));
	if length(svii) > 0
		b0 =  (1/length(svii))*sum(L0(svii) - H(svii,svi)*beta0(svi).*L0(svii));
          else 
		fprintf('No support vectors on margin - cannot compute bias.\n');
		b0 = 0;
        end

 b=[b b0];
end



