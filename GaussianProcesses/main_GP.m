close all
clear

load('mdp')
load('policy')

params.g = 9.81;       % gravity constant
params.m = 1;          % mass of pendulum
params.l = 1;          % length of pendulum
params.b = 0.2;        % damping coefficient
% params.period = 2*pi*sqrt(params.l/params.g);
params.h = 0.001;%01*params.period;
% params.std = 1;      % standard deviation input disturbance



trainingsY = [];

x = [pi+0.2, -0.2];
actions = [];
for i = 1:1000
    state_index = discretize(x, mdp.S);
    action = mdp.A(policy(state_index));
    actions = [actions;action];
    params.noise = 5*sin(10*x(2));%+1*randn();
    x = rungekutta(x, action, params);
    trainingsY = [trainingsY; x];
end

diffY = diff(trainingsY)/0.001;

d1_prime = [diffY(:,1); diffY(end,1)] - trainingsY(:,2);
d2_prime = [diffY(:,2); diffY(end,2)] - (1/(params.m*params.l)*actions+params.g/params.l*sin(trainingsY(:,1)));

plot(trainingsY(:,2),d2_prime, 'go')


% 
x = trainingsY(:,2);%gpml_randn(0.8, 20, 1);                 % 20 training inputs
y = d2_prime;%sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
xs = linspace(-1, 2, 100)';                  % 61 test inputs

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
close all
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')