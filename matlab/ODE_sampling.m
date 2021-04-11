clear; clc;


%% Plot exponential
% The function gradlogp is the score for the SDE starting at p0 = N(0,1) and evolving according to Ornst.Uhlenbeck
% We expect to be able to sample from N(0,var = D/alpha) for large T

% Define OU process parameters
n = 100000;
D = 0.1;
alpha = 1;

% Create samples
samples = randn(n,1) * sqrt((D/alpha));


% Solve ODE
T = 3;
tstart = 1;
x = -1:0.01:6;
tspan = [T,tstart];
[t,y] = ode45(@(t,y) OU_ODE(t,y,alpha,D),tspan,samples);


histogram(y(end,:),'Normalization','pdf')
hold on
x = -3:0.01:3;

plot(x,conv(x,tstart,alpha,D),'LineWidth',3)
title(sprintf('Histogram of end samples: T = %d',tstart))
legend('Hist approximation','True pdf')
xlabel('x')
ylabel('Density')


%% Plot Normal

% Define OU process parameters
n = 100000;
D = 1;
alpha = 1;

% Create samples
samples = randn(n,1) * sqrt((D/alpha));


% Solve ODE
T = 2;
tstart = 1e-3;
x = -1:0.01:6;
tspan = [T,tstart];
[t,y] = ode45(@(t,y) OU_ODE_EXP(t,y,alpha,D),tspan,samples);


histogram(y(end,:),'Normalization','pdf')
hold on
% plot(0.01:3,exp(-(0.01:3)))
plot(x,conv_exp(x,tstart,alpha,D),'LineWidth',3)
hold on 
e = exp(-x);
e(x < 0) = 0; 
plot(x,e,'LineWidth',3)
legend('Hist approximation','True conv.','True exp. pdf')
title(sprintf('Histogram of end samples: T = %d',tstart))
% legend('Hist approximation','True conv.')
xlim([-1,6])
xlabel('x')
ylabel('Density')
% 
% 
% std(y(end,:))
% mean(y(end,:))




%% Plot ODE paths

% Only plot 30 paths for visibility
plot(-t,y(:,1:30))
title('Backwards sample paths')
xlabel('T')
ylabel('Samples')




function[dy] = OU_ODE(t,y,alpha,D)
% Claim: SDE of form dx = f(x,t)dt + g(x,t) dW 
% has ODE dx = [f(x,t) - g(x,t)^2 score(x,t)/2]dt 
f = -alpha*y;
g = sqrt(2*D);
score = real(gradlogp(y,t,alpha,D));

dy = f - g.^2 * score/2;
end




function[dy] = OU_ODE_EXP(t,y,alpha,D)
% Claim: SDE of form dx = f(x,t)dt + g(x,t) dW 
% has ODE dx = [f(x,t) - g(x,t)^2 score(x,t)/2]dt 
f = -alpha*y;
g = sqrt(2*D);
score = real(gradlogp_exp(y,t,alpha,D));

dy = f - g.^2 * score/2;
end