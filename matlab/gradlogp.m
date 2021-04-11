function[grad] = gradlogp(x,t,alpha,D)
upper = -x.*exp(alpha.*t).^2.*alpha.*exp(-x.^2.*exp(alpha.*t).^2.*alpha./(2.*(exp(alpha.*t).^2.*D + alpha - D))).*sqrt(-alpha.*pi./(D.*(-1 + exp(-2.*alpha.*t)))).*sqrt(2)./(2.*(exp(alpha.*t).^2.*D + alpha - D).*pi.*sqrt((exp(alpha.*t).^2.*D + alpha - D)./(D.*(exp(alpha.*t).^2 - 1))));
lower = exp(-x.^2.*exp(alpha.*t).^2.*alpha./(2.*(exp(alpha.*t).^2.*D + alpha - D))).*sqrt(-alpha.*pi./(D.*(-1 + exp(-2.*alpha.*t)))).*sqrt(2)./(2.*pi.*sqrt((exp(alpha.*t).^2.*D + alpha - D)./(D.*(exp(alpha.*t).^2 - 1))));

grad = upper./lower;
end