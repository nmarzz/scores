clear;clc;

dx = 0.1;
dt = 0.1;

x = -3:dx:3;
t = dt:dt:1;

[x,t] = meshgrid(x,t);

% z = conv(x,t);
% 
% surf(x,t,z)
% 
% 

alpha = 1;
D = 1;
z = gradlogp_exp(x,t,alpha,D);

surf(x,t,z)
xlabel('x')
ylabel('t')
