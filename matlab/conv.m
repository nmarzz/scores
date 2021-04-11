function[p] = conv(x,t,alpha,D)
p = exp(-x.^2.*exp(alpha.*t).^2.*alpha./(2.*(exp(alpha.*t).^2.*D + alpha - D))).*sqrt(-alpha.*pi./(D.*(-1 + exp(-2.*alpha.*t)))).*sqrt(2)./(2.*pi.*sqrt((exp(alpha.*t).^2.*D + alpha - D)./(D.*(exp(alpha.*t).^2 - 1))));   
end