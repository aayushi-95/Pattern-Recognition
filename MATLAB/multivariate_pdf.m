clc;
clear all;

mu = [1.72 3.77];
sigma = [0.4027 -0.0069; -0.0069 1.283];
% x1 =[1;2;2;3;1.5;2;1;1;2]
% x2=[2;3;4;4;5;5;5;4;2]
 x1 = -3:0.2:3;
 x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));

surf(x1,x2,y)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
axis([-3 3 -3 3 0 0.4])
xlabel('x1')
ylabel('x2')
zlabel('Probability Density')