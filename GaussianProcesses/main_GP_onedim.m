close all
clear

load('mdp.mat')
load('policy.mat')

params.g = 9.81;       % gravity constant
params.m = 1;          % mass of pendulum
params.l = 1;          % length of pendulum
params.b = 0.2;        % damping coefficient
params.h = 0.001;%01*params.period;

X = [];

x = [pi-0.2, 0.0];
actions = [];
for i = 1:1000
    X = [X; x];
    state_index = discretize(x, mdp.S);
    action = mdp.A(policy(state_index));
    actions = [actions;action];
    params.noise = 5*sin(10*x(2))+1*randn();
    x = rungekutta(x, action, params);
end

dotX = diff([X; X(end,:)])/params.h;
d1 = dotX(:,1) - X(:,2);
d2 = dotX(:,2) - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));

y = d2;
x = X(:,2);
x_test = linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, 100)';
 

meanfunc = @meanZero;                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([x_test; flipdim(x_test,1)], f, [7 7 7]/8)
hold on; plot(x_test, mu); plot(x, y, '+')
title('GP posterior')
xlabel('x2')
ylabel('$$\hat{d}$$','Interpreter','Latex')