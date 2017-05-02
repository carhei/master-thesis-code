clear
close all
clearvars -global

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILDING MDP FOR THE INVERTED PENDULUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Physical pendulum parameters
pendulum.nstates = 2;
pendulum.params.g = 9.81;               % gravity constant
pendulum.params.m = 1;                  % mass of pendulum
pendulum.params.l = 1;                  % length of pendulum
pendulum.params.b = 0.2;                  % length of pendulum
pendulum.params.h_safe = 0.005;                % sample time
pendulum.params.h_learn = 0.2;
pendulum.params.h= pendulum.params.h_learn;
pendulum.params.noise = 2.0;            % standard deviation input disturbance


% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 25;
x2_bounds = [-5,5];
x2_steps = 25;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];
pendulum.grid.input_bounds = [-20; 20];
pendulum.grid.input_steps =  5;
pendulum.grid.varyspacing = 0;

pendulum.purpose = 'forValueIteration';
pendulum.reward.type = 'exponential';   %exponential reward

mdp = BuildMDP(pendulum);

% Initialize Speedy Q Learning

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection
steps = 100000;
vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps);
visited = zeros(size(mdp.S,1), size(mdp.A,1));



epstarget = 0.01;
epsilon_1 = 0.0001;
m_0 = 5;
% Optimistic initialization
Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));%+ .01*rand(size(mdp.S,1), size(mdp.A,1));


U = zeros(size(mdp.S,1), size(mdp.A,1));
B = zeros(size(mdp.S,1), size(mdp.A,1));
C = zeros(size(mdp.S,1), size(mdp.A,1));
L = ones(size(mdp.S,1), size(mdp.A,1));
m = m_0*ones(size(mdp.S,1), size(mdp.A,1));
t_star = 0;

violation = 0;
violation_times = [];

%% MAIN LOOP


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAFE SPEEDY Q-LEARNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('================Safe Speedy Q-learning====================\n')


x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
x = [x01,x02];
state = discr(x, mdp.S);

for n=1:steps-1
    
    action = egreedy(state,epsilon,Q);
    u = mdp.A(action);
    
    visited(state,action) = visited(state,action)+1;
    %         apply control
    x_prime = rungekutta(x, u, pendulum.params);
    s_prime = discr(x_prime,mdp.S);
    
    if B(state, action) <= t_star
        L(state, action) = 1;
    end
    
    if L(state, action) == 1
        if C(state, action) == 0
            B(state, action) = n;
        end
        
        R = mdp.R(state,action);%+ 1/(1+visited(state_t,action));
        
        C(state, action) = C(state, action) + 1;
        U(state, action) = U(state, action) + R + gamma*max(Q(s_prime, :));
        
        if C(state, action) == m(state, action)
            q = U(state, action)./m(state, action);
            if abs(Q(state, action) - q) >= epsilon_1
                Q(state, action) = q;
                t_star = n;
            elseif B(state, action) > t_star
                fprintf('Discarding update.\n');
                L(state, action) = 0;
            end
            
            U(state, action) = 0;
            C(state, action) = 0;
            m(state, action) = ceil(min(1.02*m(state, action) + 1, 500));
            epsilon_1 = min(epsilon_1*1.1, epstarget);
        end
    end
    state = s_prime;
    
    if abs(x') <= 1.05* pendulum.grid.state_bounds(:,2)
        x = x_prime;
    else
        unsafe = 1;
        violation = violation +1;
        violation_times = [violation_times; n];
        x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
        x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
        x = [x01,x02];
        state = discr(x, mdp.S);
        
    end
    V = max(Q,[],2);
    vecV(:,n) = V;
    if mod(n,1000)==0
        disp(['Episode ' num2str(n)])
    end
end

fprintf('.....Calculating logarithmic error......\n')
opt_params.gamma = gamma;
opt_params.epsilon = 0.0001;

[policy_true, p, v0_vec] = valueiteration(mdp, opt_params);

figure()
for i = 1:size(vecV)
    semilogy(abs(v0_vec(i,end)-vecV(i,1:n)))
    hold all
end
title('Error Speedy Q Learning')
xlabel('Episodes')
ylabel('Error')
xlim([0; n])
drawnow;



[~,policy] = max(Q,[],2);
Policy = figure;

marker = 90*ones(size(policy_true));

figure(); clf;colormap('jet')
xlabel('angle (rad)')
ylabel('rate (rad/s)')
zlabel('torque (Nm)')
title('Policy comparison')
axis equal
subplot(1,2,1)
scatter3(mdp.S(:,1), mdp.S(:,2), policy, marker, policy, 'filled');
view(2)
subplot(1,2,2)
scatter3(mdp.S(:,1), mdp.S(:,2), policy_true, marker, policy_true, 'filled');
view(2)

