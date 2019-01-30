clear all
load cl_train_1.csv
load cl_test_1.csv

cl_test = cl_test_1;
cl_train = cl_train_1;

x1_test = cl_test(:,1);
x2_test = cl_test(:,2);
y_test = cl_test(:,3);


x1_train = cl_train(:,1);
x2_train = cl_train(:,2);

y_train = cl_train(:,3);
w = zeros(1,size(cl_train,2))';

iterations = 1000;
alpha = 0.1; %learning rate
X_train = [ones(length(cl_train),1) cl_train(:,1:2)];
X_test = [ones(length(cl_test),1) cl_test(:,1:2)];

cross_entropy_train = zeros(1,iterations);
cross_entropy_test = zeros(1,iterations);
for k = 1:iterations
    %Calculating Cross Entropy 
    sum1 = [0;0;0];
    element_tr = (sigmf(X_train*w, [1,0]).^y_train).*((1 - sigmf(X_train*w, [1,0])).^(1-y_train));
    element_tr = log(element_tr);
    neg_likelihood_train = - sum(element_tr);
    cross_entropy_train(k) = neg_likelihood_train;
    
    element_te = (sigmf(X_test*w, [1,0]).^y_test).*((1 - sigmf(X_test*w, [1,0])).^(1-y_test));
    element_te = log(element_te);
    neg_likelihood_test = -sum(element_te);
    cross_entropy_test(k) = neg_likelihood_test;  
    
    % Gradient descent
    for i = 1:length(y_train)
        x_curr = [1 cl_train(i,1:2)]';
        y_curr = y_train(i);
        der_CE = sigmf(w'*x_curr,[1 0])-y_curr ;  %CE = cross entropy
        der_h = x_curr;  
        sum1 = sum1 + der_CE*der_h;   
    end
    w = w-alpha*sum1;    
end


%% plot 
x1 = 0:0.01:1.5;
x2 = -w(1)/w(3) - (w(2)/(w(3)).*x1);

figure(1)
hold on
% training set
for j = 1:length(y_train)
    if y_train(j) == 1
        plot(x1_train(j), x2_train(j), 'ob')
    else
        plot(x1_train(j), x2_train(j), 'or')
    end
end
plot(x1,x2)
title('Logistic regression  - training set')
hold off

% test set
figure(2)
hold on
for j = 1:length(y_test)
    if y_test(j) == 1
        plot(x1_test(j), x2_test(j), 'ob')
        hold on
    else
        plot(x1_test(j), x2_test(j), 'or')
        hold on
    end
end
plot(x1,x2)
title('Logistic regression  - test set')
hold off

% cross entropies

k = 1:iterations;

figure(3)
plot(k,cross_entropy_test);
title('Cross entropy for test set')
xlabel('Iteration')

figure(4)
plot(k,cross_entropy_train);
title('Cross entropy for train set')
xlabel('Iteration')




