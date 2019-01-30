clear all
load test_1d_reg_data.csv
load train_1d_reg_data.csv

reg_test =  test_1d_reg_data;
reg_train =  train_1d_reg_data;

dimension = size(reg_train,2)-1;
trainSize = size(reg_train,1);
testSize = size(reg_test,1);

X_test = reg_test(:, 1:dimension);
X_test = [ones(testSize,1), X_test];
y_test = reg_test(:, dimension+1);

X_train = reg_train(:, 1:dimension);
X_train = [ones(trainSize,1), X_train];
y_train = reg_train(:, dimension+1);

W = pinv(X_train'*X_train)*X_train'*y_train;  %pinv = pseudoinvers to circumvent the non-singular requirement on X^TX

%% plot
x1 = 0:0.01:1.1;
x2 = W(1) + W(2).*x1;

% training set
figure(1)
hold on
for j = 1:length(y_train)
    plot(X_train(j,2), y_train(j), 'ob')
end
plot(x1,x2)
title('Linear regression - Training set')
hold off

% test set
figure(2)
hold on
for j = 1:length(y_test)
    plot(X_test(j,2), y_test(j), 'or')
end
plot(x1,x2)
title('Linear regression - test set')
hold off


