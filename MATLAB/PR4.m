clear all;
mu = [0,0];
sigma = [9 0;0 1];
x = -8:1:8;
y = -8:1:8;
[X,Y] = meshgrid(x,y);

Z = mvnpdf([X(:) Y(:)],mu,sigma);
Z = reshape(Z,size(X));
contour(X,Y,Z), axis equal
% Z = mvnrnd(mu,sigma,100);