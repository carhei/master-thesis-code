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
pendulum.params.noise = 0.0;      % standard deviation input disturbance



% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 3;
x2_bounds = [-5,5];
x2_steps = 3;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];

pendulum.grid.input_bounds = [-10; 10];
pendulum.grid.input_steps =  2;

pendulum.grid.varyspacing = 0;


pendulum.purpose = 'forValueIteration';


pendulum.reward.type = 'exponential';  %exponential reward

mdp = BuildMDP(pendulum);

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection

Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));
Q_prev = Q;



episodes = 1000;
steps = 1000;
cum_reward= 0;
vecV = [];
rewardperepisode = [];
visited = zeros(size(mdp.S,1), size(mdp.A,1));


for i = 1:episodes
    x1 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
    x2 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
    x = [x1,x2];
    R = 0;
    cum_reward= 0;
    state = discr(x, mdp.S);
    
    for j = 1:steps
        
        %choose action
        action = egreedy(state,epsilon,Q);
        u = mdp.A(action);
        visited(state,action) = visited(state,action)+1;
        %apply control
        x_prime = rungekutta(x, u, pendulum.params);
        s_prime = discr(x_prime, mdp.S);
        
        R = mdp.R(state,action);
        cum_reward = cum_reward + R;
        
        temp = Q;
        
        Q(state,action) =  Q(state,action) + alpha/visited(state,action) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state,action) )+ ...
            (1-alpha/visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        
        state = s_prime;
        x = x_prime;
        Q_prev = temp;
        
        
    end
    V = max(Q,[],2);
    vecV = [vecV V];
    rewardperepisode = [rewardperepisode, cum_reward/steps];
end




for i = 1:length(V)
    plot(vecV(i,:))
    hold all
end
title('Speedy Q Learning')


opt_params.gamma = gamma;
opt_params.epsilon = 0.0001;

[policy, n, v0_vec] = valueiteration(mdp, opt_params);

figure()

for i = 1:size(v0_vec,1)
    plot(v0_vec(i,:))
    hold all
end
title('Value Iteration')

figure()
for i = 1:size(v0_vec,1)
    semilogy(abs(v0_vec(i,end)-vecV(i,:)))
    hold all
end
title('Error Speedy Q Learning')
