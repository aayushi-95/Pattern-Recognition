clc;
clear all;

x = [2; 4; 0; 7; 1; 2; 0; 3; 2; 1; 5; 4; 3];
pd = fitdist(x,'Normal')
% x_val = [-6:1:6];
% y = normpdf(x,2.615,2.022)
% y = normpdf(pd,x_val);
% plot(x_val,y)

% histogram(x,25)
histfit(x)