clear;clc;

D = 0.1;
alpha = 1;
dx = 0.01;
space = -3:dx:3;
sigma = sqrt(2*D);

f = @(x,t) -alpha * x;
g = @(x,t) sigma;

n = 10000;
x0 = randn(1,n);


t0 =0;
te = 2;


[x,t] = sde_solve(f,g,x0,t0,te,dx);

plot(t,x)


idx_to_check = 190;
tv = t(idx_to_check);
empiric = x(idx_to_check,:);
analytic = conv(space,tv,alpha,D);


histogram(empiric,'Normalization','pdf')
hold on;
plot(space,analytic)









