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

S = mdp.S;
A = mdp.A;
R = mdp.R;



s = size(S,1);
a = size(A,1);


gamma       = 0.9;
epsilon     = 0.1;
delta = 0.8;
kappa = 1/((1-epsilon)*gamma);
m = 150;%log(3*s*a*(1+s*a*kappa)/delta)/(s*epsilon^2*(1-gamma)^2);



Q = max(max(R))/(1-gamma)*ones(s,a); %
U = zeros(s,a);

l       = zeros(s,a);
t       = zeros(s,a);
LEARN   = ones(s,a);
vecV = [];

t_prime = 0;

cum_reward = 0;

% while max(abs(V-V_prev)) >= epsilon &&  i~=415
% for episode = 1:10000
    x = [0, 0.1];
    state = discr(x, S);
    
    for steps = 1:1e5
        
        [v, action] = max(Q(state,:));
        u = A(action);
        
        r = R(state,action);
        cum_reward = cum_reward+r;
        x_prime = rungekutta(x, u, pendulum.params);
        s_prime = discr(x_prime, mdp.S);
        
        if LEARN(state,action)
            U(state,action) = U(state,action) + r + gamma * max(Q(s_prime, :), [], 2);
            l(state,action) = l(state,action)+1;
            if l(state,action) == m
                if Q(state,action)-U(state,action)/m >= 2*epsilon
                    Q(state,action) = U(state,action)/m + epsilon;
                    t_prime = steps;
                else if t(state,action) >= t_prime
                        LEARN(state,action) = false;
                    end
                end
                t(state,action) = steps;
                U(state,action) = 0;
                l(state,action) = 0;
                
            end
        else if t(state,action) < t_prime
                LEARN(state,action) = true;
            end
        end
        
        state = s_prime;
        if abs(x(2)) <= 1.5* pendulum.grid.state_bounds(2,2)
            x = x_prime;
        else
            x = [0, 0.1];
        end
        
    
    
    V = max(Q,[],2);
    vecV = [vecV V];
    
    end
    %     i = i+1;
    
%     l       = zeros(s,a);
%     t       = zeros(s,a);
%     LEARN   = ones(s,a);
%     t_prime = 0;
%     disp(['Episode: ',int2str(episode),'  Reward:',num2str(cum_reward)])
%     cum_reward = 0;
% end



for i = 1:length(V)
    plot(vecV(i,:))
    hold all
end


opt_params.gamma = gamma;
opt_params.epsilon = 0.0001;

[policy, n, v0_vec] = valueiteration(mdp, opt_params);

figure()
for i = 1:size(v0_vec,1)
    plot(v0_vec(i,:))
    hold all
end
