trainX = loadMNISTImages('train-images-idx3-ubyte')';
traint = loadMNISTLabels('train-labels-idx1-ubyte');
save data.mat;
trainT = bsxfun(@eq, traint, 0:9);
training_set=trainX;
training_label=traint;

testX = loadMNISTImages('t10k-images-idx3-ubyte')';
testt = loadMNISTLabels('t10k-labels-idx1-ubyte');
testT = bsxfun(@eq, testt, 0:9);
test_set=testX;
test_label=testt;

trainX = [ones(size(trainX, 1), 1) trainX];
testX = [ones(size(testX, 1), 1) testX];
save('data.mat','training_set','-append');
save('data.mat','training_label','-append');
save('data.mat','test_set','-append');
save('data.mat','test_label','-append');
save('data.mat','trainX','-append');
save('data.mat','trainT','-append');
save('data.mat','testX','-append');
save('data.mat','testT','-append');



