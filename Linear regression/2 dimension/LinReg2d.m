clear all
load test_2d_reg_data.csv
load train_2d_reg_data.csv

reg_test =  test_2d_reg_data;
reg_train =  train_2d_reg_data;

dimension = size(reg_train,2)-1;
trainSize = size(reg_train,1);
testSize = size(reg_test,1);

X_train = reg_train(:, 1:dimension);
X_train = [ones(trainSize,1), X_train]; % add ones for the bias term to get it on general form
y_train = reg_train(:, dimension+1);

X_test = reg_test(:, 1:dimension);
X_test = [ones(testSize,1), X_test]; % add ones for the bias term to get it on general form
y_test = reg_test(:, dimension+1);

W = pinv(X_train'*X_train)*X_train'*y_train;  %pinv = pseudoinvers to circumvent the non-singular requirement on X^T*X

MSE_train = 1/trainSize*(X_train*W-y_train)'*(X_train*W-y_train); % Mean Square Error for the training set

MSE_test = 1/testSize*(X_test*W-y_test)'*(X_test*W-y_test); % Mean Square Error for the test set



