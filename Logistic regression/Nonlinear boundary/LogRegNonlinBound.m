clear all
load cl_train_2.csv
load cl_test_2.csv

cl_test = cl_test_2;
cl_train = cl_train_2;

x1_test = cl_test(:,1);
x2_test = cl_test(:,2);
y_test = cl_test(:,3);


x1_train = cl_train(:,1);
x2_train = cl_train(:,2);

y_train = cl_train(:,3);
w = zeros(1,size(cl_train,2)+2)';

iterations = 1000;
alpha = 0.1; %learning rate

for k = 1:iterations
    sum = [0;0;0;0;0];
    for i = 1:length(y_train)
        x_curr = [1 cl_train(i,1:2)]';
        y_curr = y_train(i);
        z=w(1)*x_curr(1) + w(2)*x_curr(2) + w(3)*x_curr(3) + w(4)*x_curr(2).^2 + w(5)*x_curr(3).^2;
        der_CE = sigmf(z,[1 0])-y_curr ;
        der_h = [x_curr(1);x_curr(2);x_curr(3);x_curr(2).^2;x_curr(3).^2];
        sum = sum + der_CE*der_h;   
    end
    w = w-alpha*sum;
end

%% plot 
f = @(x,y) w(1)+w(2)*x + w(3)*y + w(4)*x.^2 + w(5)*y.^2;

% Training set

figure(1)
hold on
for j = 1:length(y_train)
    if y_train(j) == 1
        plot(x1_train(j), x2_train(j), 'ob')
    else
        plot(x1_train(j), x2_train(j), 'or')
    end
end
fimplicit(f);
title('Logisitc regression 2 - training set')
hold off

figure (2)
hold on

% Test set
for j = 1:length(y_test)
    if y_test(j) == 1
        plot(x1_test(j), x2_test(j), 'ob')
    else
        plot(x1_test(j), x2_test(j), 'or')
    end
end
fimplicit(f);
title('Logisitc regression 2 - test set')
hold off
