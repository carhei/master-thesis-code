

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
x1_bounds = [-pi/2, pi/2];
x1_steps = 15;
x2_bounds = [-5,5];
x2_steps = 15;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];

pendulum.grid.input_bounds = [-15; 15];
pendulum.grid.input_steps =  9;

pendulum.grid.varyspacing = 0;


pendulum.purpose = 'forLearning';




mdp = BuildMDP(pendulum);


Q = zeros(size(mdp.S,1), size(mdp.A,1));


alpha       = 0.1;   % learning rate
gamma       = 0.99;   % discount factor
epsilon     = 0.1;  % probability of a random action selection


episodes = 2300;
steps = 500;
cum_reward= 0;

steps_index_plot = [];

% xplot = [];steps_plot = [];total_reward=zeros(episodes,1);

for i = 1:episodes
    
    steps_index = 0;
    x = [0, 0.1];
    R = 0;
    cum_reward= 0;
    goal = 0;
    
    state = discretize(x, mdp.S);
    
    
    
    
    for j = 1:steps
        
        
        %choose action
        action = egreedy(state,epsilon,Q);
        u = mdp.A(action);
        
        %apply control
        x_prime = rungekutta(x, u, pendulum.params);
        sigma = 1;
        
        if x_prime(1)>pi/2 || x_prime(1)<-pi/2
            goal = 1;
            R = -100;
        else
            R = -10*x(1).^2 -3*x(2).^2;
            %             R = -(1-exp(-sum(abs(x_prime).^2,2)/(sigma^2)));
        end
        
        
        cum_reward = cum_reward + R;
        
        s_prime = discretize(x_prime, mdp.S);
        
        Q(state,action) =  Q(state,action) + alpha * ( R + gamma*max(Q(s_prime,:),[],2) - Q(state,action) );
        
        state = s_prime;
        x = x_prime;
        steps_index = steps_index +1;
        
        
        %         plot_Pole(x,u,steps_index);
        if goal
            break
        end
    end
    steps_index_plot = [steps_index_plot steps_index];
    
    
    disp(['Episode: ',int2str(i),'  Steps:',int2str(steps_index),'  Reward:',num2str(cum_reward),' epsilon: ',num2str(epsilon)])
end