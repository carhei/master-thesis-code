clear
close all
clc

pendulum.nstates = 2;
pendulum.params.g = 9.81;       % gravity constant
pendulum.params.m = 1;          % mass of pendulum
pendulum.params.l = 1;          % length of pendulum
pendulum.params.b = 0.2;        % damping coefficient
% pendulum.params.period = 2*pi*sqrt(pendulum.params.l/pendulum.params.g);
pendulum.params.h = 0.2;%01*pendulum.params.period;
pendulum.params.std = 0.0;      % standard deviation input disturbance



% MDP grid params
x1_bounds = [-pi/4, pi/4];
x1_steps = 3;
x2_bounds = [-5,5];
x2_steps = 3;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];
pendulum.grid.input_bounds = [-10; 10];
pendulum.grid.input_steps =  4;
pendulum.grid.varyspacing = 0;
pendulum.purpose = 'forValueIteration';
pendulum.reward.type = 'exponential';  %exponential reward

mdp = BuildMDP(pendulum);
xmean = mdp.S;%[mdp.S(2,:); mdp.S(4,:);mdp.S(5,:);mdp.S(6,:);mdp.S(8,:)];

theta = zeros(size(xmean,1),size(mdp.A,1));%zeros(size(mdp.S,1),size(mdp.A,1));
theta_prev = abs(theta)+10;

alpha       = 1;   % learning rate
gamma       = 0.6; % discount factor
epsilon     = 0.1; % probability of a random action 



stepsize1 = diff(pendulum.grid.state_bounds(1,:))/(pendulum.grid.state_steps(1)-1);
stepsize2 = diff(pendulum.grid.state_bounds(2,:))/(pendulum.grid.state_steps(2)-1);
xsigma = 0.5*[stepsize1; stepsize2];
n = 1;
x = [0, 0.1];
R = 0;
visited = ones(size(mdp.S,1),size(mdp.A,1));

while n<5e4% norm(abs(theta-theta_prev))>= 0.001
    n = n+1;
    phi = rbf(x, xmean, xsigma);
    state = discr(x, mdp.S);
    if rand > epsilon
        [v, action] = max(theta'*phi);
    else
        action = randi(size(mdp.A,1));
    end
    u = mdp.A(action);
    %     disp(u)
    x_prime = rungekutta(x, u, pendulum.params);
    phi_prime = rbf(x_prime,xmean,xsigma);
    R = mdp.R(state,action);
    %     R = 1-(1-exp(-sum(abs(x).^2,2)/(sigma^2)));
    Qmax = max(theta'*phi_prime);
    delta = R + gamma*Qmax - theta(:,action)'*phi;
    theta_prev = theta;
    theta(:,action) =  theta(:,action) + alpha/visited(state,action)*delta*phi;
    if abs(x(2)) <= 1.5* pendulum.grid.state_bounds(2,2)
        x = x_prime;
    else
        x = [0, 0.1];
    end
    visited(state,action) = visited(state,action) +1;
    for i = 1:size(mdp.S,1)
        V(i,n) = max(theta'*rbf(mdp.S(i,:),xmean,xsigma));
    end
end



figure()

for j = 1:size(mdp.S,1)
    plot(V(j,:))
    hold all
end



opt_params.gamma = gamma;
opt_params.epsilon = 0.0001;

[policy,~, v0_vec] = valueiteration(mdp, opt_params);



figure()

for j = 1:size(mdp.S,1)
    plot(v0_vec(j,:))
    hold all
end
