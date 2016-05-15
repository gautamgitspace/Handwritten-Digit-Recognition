function main()
save proj3.mat;
load data.mat;

%-----------------------LOGISTIC REGRESSION-------------------------%
N = size(training_set,1);
features = size(training_set,2);
classes = 10;

T = zeros(N, classes);
for i = 1:classes
    T(:, i) = (training_label==i-1);  
end

Weight = train_LR(training_set, T, test_set, test_label);
Wlr = Weight(1:features,:);
blr = ones(1,classes);

save('proj3.mat','Wlr','-append');
save('proj3.mat','blr','-append');

%------------------------NEURAL NETWORKS---------------------------%

wji = rand(785,50)/100;
wkj = rand(51, 10)/100;
Enew=1;
Eold = 2;
EE = [];
eta = 0.001;
while(abs(Enew-Eold) > 0.01),
   for i=1:60000,
       Eold = Enew;
       X = trainX(i,:);
       z = tanh(X*wji);
       z = [1 z];
       a = z*wkj;
       y = zeros(1,10);
       for j = 1:10,
           y(j) = exp(a(j))./sum(exp(a));
       end
       Ecrosstropy = -sum(trainT(i,:).*log(y+0.001));
       EE = [EE Ecrosstropy];
       plot(EE)
       drawnow;
       delk = y - trainT(i,:);
       zz = z(1,2:size(z,2));
       delj = (1-tanh(zz).^2)*sum(wkj*delk');
       delEkj = z'*delk;
       delEji = X'*delj;
       wji = wji - eta*delEji;
       wkj = wkj - eta*delEkj;
   end
end
Wnn1 = wji(2:size(wji,1),:);
bnn1 = wji(1,:);
Wnn2 = wkj(2:size(wkj,1),:);
bnn2 = wkj(1,:);
h='tanh';

save('proj3.mat','Wnn1','-append');
save('proj3.mat','Wnn2','-append');
save('proj3.mat','bnn1','-append');
save('proj3.mat','bnn2','-append');
save('proj3.mat','h','-append');

disp('done');
end

function ws = train_LR( x, T, xv, Lv)
batch = 100;
numpass = 400;
eta.a = 0.01; 
eta.b = 0.01; 
 
M = size(x,2)+1; 
K = size(T,2);  
 
 
phi = [x, ones(size(x,1),1)];
phiV = [xv, ones(size(xv,1),1)];
 
%setting random initial weights
w0 = randn((M-1),K)*0.1;  
w0 = [w0;zeros(1,10)]; %adding bias
ws = StochasticGD(phi, T, phiV, Lv, w0, batch, numpass, eta);
end
 
 
function w = StochasticGD(X, T, validationset ,validationlabels, w0, batchsize, numpass, eta)
    N = size(X,1);
    w = w0;
    epoch = 0;           
    jumpsize = eta.a / (eta.b+epoch);
    batchstartwith = 1;
    
    while(epoch < numpass)
        batchendwith   = min(N,batchstartwith+batchsize-1);
        dw = ErrorGradient(w,X(batchstartwith:batchendwith,:),T(batchstartwith:batchendwith,:));
        w = w - jumpsize*dw;
        batchstartwith = batchstartwith+batchsize;
        if(batchstartwith>N)
            errorrate = TestLogistic(w, validationset, validationlabels);
            %print iterations to Commandline
            fprintf('P:%d,Validation error rate = %.1f%%, Norm dw = %f, stepsize = %f\n',epoch, errorrate*100, norm(dw,2), jumpsize);
            batchstartwith = 1;      
            epoch = epoch+1;
            jumpsize = eta.a / (eta.b+epoch);
        end
    end
end
 
 
function gradientError = ErrorGradient(ws,phi,T)
    N = size(phi,1); 
    K = size(T,2);   
 
    gradientError = zeros(size(ws));
    for n = 1:N
        pn = exp(ws'*phi(n,:)'); 
        yn = pn/sum(pn);            
        for i=1:K
            gradientError(:,i) = gradientError(:,i) + (yn(i) - T(n,i))*phi(n,:)'; 
        end
    end
    gradientError = (1/N)*gradientError;  
end
 
function err_rate = TestLogistic( ws, testphi_matrice, testlabels )
Nwrong =0;
Nv = size(testphi_matrice,1);
    for i=1:Nv
        as = ws'*testphi_matrice(i,:)'; 
        [~, label(i)] = max(as);
        label(i) = label(i) - 1;
        if(label (i) ~= testlabels (i))
            Nwrong = Nwrong+1;
        end
    end
    err_rate = Nwrong / size(testphi_matrice,1);
end



