function[y] = conv_exp(x,t,alpha,D)
x0 = 10000;
y = -exp(-exp(alpha.*t).*x).*(exp(exp(alpha.*t).*x).*exp((-2.*alpha.*exp(alpha.*t).*x - D + 2.*alpha.^2.*t + D.*exp(2.*alpha.*t))./(2.*alpha)).*erf((exp(-2.*alpha.*t).*alpha.*x0 - alpha.*exp(-alpha.*t).*x - D.*exp(-2.*alpha.*t) + D).*exp(alpha.*t)./(D.*(-1 + exp(-2.*alpha.*t)).*sqrt(-2.*alpha./(D.*(-1 + exp(-2.*alpha.*t)))))) + exp((2.*alpha.^2.*t + D.*exp(2.*alpha.*t) - D)./(2.*alpha)).*erf(exp(alpha.*t).*sqrt(2).*(D.*exp(2.*alpha.*t) - alpha.*exp(alpha.*t).*x - D)./(2.*D.*(exp(2.*alpha.*t) - 1).*sqrt(alpha.*exp(2.*alpha.*t)./(D.*(exp(2.*alpha.*t) - 1))))))./2;
end