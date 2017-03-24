clear
close all

S = 3;
A = 2;

% Rewards generated according to Gaussian distribution

% load('P')
% load('r')

% Transition probability matrix sampled from uniform distribution
% P{s}(k, a) is the transition probability from 'state s' to 'state k' under
% 'action a'

r = randn(S, A);
P = cell(S, 1);

for s=1:S
    P{s} = rand(S, A);
    for a=1:A
        P{s}(:, a) = P{s}(:, a)./sum(P{s}(:, a));
    end
end

% Discount factor
lambda = .9;

% save('P')
% save('r')

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection

Q = rand(S,A);
Q_prev = Q;



episodes =1000;
steps = 1000;
cum_reward= 0;
speedyV = [];
rewardperepisode = [];
visited = zeros(S,A);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPEEDY Q LEARNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:episodes
    state = 1;
    R = 0;
    cum_reward= 0;
    
    for j = 1:steps
        
        %choose action
        action = egreedy(state,epsilon,Q);
        visited(state,action) = visited(state,action)+1;
        %apply control
        cdf = cumsum(P{state}(:,action));
        s_prime = min(find(rand < cdf(:)));
        
        
        R = r(state,action);
        cum_reward = cum_reward + R;
        
        temp = Q;
        
        
        Q(state,action) =  Q(state,action) + alpha/(visited(state,action)) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state,action) )+ ...
            (1-alpha/visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        
        state = s_prime;
        Q_prev = temp;
        
        
    end
    V = max(Q,[],2);
    speedyV = [speedyV V];
    rewardperepisode = [rewardperepisode, cum_reward/steps];
    disp(['Episode: ',int2str(i),'  Reward:',num2str(cum_reward),' epsilon: ',num2str(epsilon)])
end


figure()

for i = 1:length(V)
    plot(speedyV(i,:))
    hold all
end

title('Speedy Q Learning')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DELAYED Q LEARNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Q = max(max(r))/(1-gamma)*ones(S,A);

l       = zeros(S,A);
t       = zeros(S,A);
LEARN   = ones(S,A);
m = 10;
U = zeros(S,A);
epsilon     = 0.001;

vecV = [];

t_prime = 0;

cum_reward = 0;


state = 1;
R = 0;
cum_reward= 0;

for j = 1:1e5
    
    [v, action] = max(Q(state,:));
    
    R = r(state,action);
    cum_reward = cum_reward+R;
    cdf = cumsum(P{state}(:,action));
    s_prime = min(find(rand < cdf(:)));
    
    if LEARN(state,action)
        U(state,action) = U(state,action) + R  + gamma * max(Q(s_prime, :));
        l(state,action) = l(state,action)+1;
        if l(state,action) == m
            if Q(state,action)-U(state,action)/m >= 2*epsilon
                Q(state,action) = U(state,action)/m + epsilon;
                t_prime = j;
            else if t(state,action) >= t_prime
                    LEARN(state,action) = false;
                end
            end
            t(state,action) = j;
            U(state,action) = 0;
            l(state,action) = 0;
            
        end
    else if t(state,action) < t_prime
            LEARN(state,action) = true;
        end
    end
    
    state = s_prime;
    V = max(Q,[],2);
    vecV = [vecV V];
    
end

figure()

for i = 1:length(V)
    plot(vecV(i,:))
    hold all
end

title('Delayed Q Learning')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q LEARNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = rand(S,A);
Q_prev = Q;
epsilon     = 0.1;
vecV = [];

for i = 1:episodes
    state = 1;
    R = 0;
    cum_reward= 0;
    
    for j = 1:steps
        
        %choose action
        action = egreedy(state,epsilon,Q);
        visited(state,action) = visited(state,action)+1;
        %apply control
        cdf = cumsum(P{state}(:,action));
        s_prime = min(find(rand < cdf(:)));
        
        
        R = r(state,action);
        cum_reward = cum_reward + R;
        
        temp = Q;
        
        
        Q(state,action) =  Q(state,action) + alpha/(visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - Q(state,action) );
        
        state = s_prime;
        Q_prev = temp;
        
        
    end
    V = max(Q,[],2);
    vecV = [vecV V];
    rewardperepisode = [rewardperepisode, cum_reward/j];
    disp(['Episode: ',int2str(i),'  Reward:',num2str(cum_reward),' epsilon: ',num2str(epsilon)])
end


figure()

for i = 1:length(V)
    plot(vecV(i,:))
    hold all
end


title('Q Learning')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VALUE ITERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Discount factor
err_max =  0.0001;
n = 0;
v0 = zeros(S, 1);
v1 = ones(S, 1);
policy = zeros(S, 1);

v0_vec = v0;
policy_vec = policy;


while max(abs(v1-v0)) >= err_max*(1-gamma)/(2*gamma)
    n = n+1;
    v1 = v0;
    for s=1:S
        tmp_s = r(s, :)' + gamma*P{s}'*v1;
        [v0(s), policy(s)] = max(tmp_s);
    end
    
    v0_vec = [v0_vec v0];
    policy_vec = [policy_vec policy];
    if mod(n,50) == 0
        fprintf('Iter: %d\n',n);
    end
    if n>5000
        break;
    end
    
end


figure()

for i = 1:size(v0_vec,1)
    plot(v0_vec(i,:))
    hold all
end


title('Value Iteration')

figure()
for i = 1:size(v0_vec,1)
    semilogy(abs(v0_vec(i,end)-speedyV(i,:)))
    hold all
end
title('Error Speedy Q Learning')