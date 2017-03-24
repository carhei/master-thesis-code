clear
close all

s = 5;
a = 3;

% r = randn(s, a);
load('r.mat')
load('P.mat')
load('A.mat')
load('S.mat')

S = linspace(0,1,5);

% P = cell(s, 1);
% 
% for i=1:s
%     P{i} = rand(s, a);
%     for j=1:a
%         P{i}(:, j) = P{i}(:, j)./sum(P{i}(:, j));
%     end
% end


% S = rand(s,1);
% A = rand(a,1);
xmean = S';%[S(1);S(2);S(3);S(4);S(5)];%[ 0.15; 0.45; 0.65; 1];
 
 alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.3;  % probability of a random action selection



steps = 5e5;
visited = zeros(s,a);
theta = zeros(size(xmean,1),a);
theta_prev = -theta;


save('S','S')
save('A','A')
save('r','r')
save('P','P')


state = 1;
x = S(state);
R = 0;
bw =  0.0425;%[0.01; 0.01; 0.01; 0.01];
    


for n = 1:steps
    phi = rbf(x, xmean, bw);
    %choose action
    if rand > epsilon
        [v, action] = max(theta'*phi);
    else
        action = randi(a);
    end
    u = A(action);
    visited(state,action) = visited(state,action)+1;
    %apply control
    cdf = cumsum(P{state}(:,action));
    s_prime = min(find(rand < cdf(:)));
    x_prime = S(s_prime);
    phi_prime = rbf(x_prime,xmean,bw);
    R = r(state,action);
    
    
    Qmax = max(theta'*phi_prime);    
    delta = R + gamma*Qmax - theta(:,action)'*phi;
    theta_prev = theta;
    theta(:,action) =  theta(:,action) + alpha/(n^.6)*delta*phi;
    
    
    state = s_prime;
    x = x_prime;
    
    for i = 1:s
        V(i, n) = max(theta'*rbf(S(i),xmean,bw));
    end
end

figure()

for j = 1:s
plot(V(j,:))
hold all
end


err_max =  0.0001;
n = 0;
v0 = zeros(s, 1);
v1 = ones(s, 1);
policy = zeros(s, 1);

v0_vec = v0;
policy_vec = policy;


while max(abs(v1-v0)) >= err_max*(1-gamma)/(2*gamma)
    n = n+1;
    v1 = v0;
    for i=1:s
        tmp_s = r(i, :)' + gamma*P{i}'*v1;
        [v0(i), policy(i)] = max(tmp_s);
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

for j = 1:s
plot(v0_vec(j,:))
hold all
end

figure()
[Ssorted, inds] = sort(S);
plot(Ssorted, v0(inds))

figure()
plot(Ssorted, V(inds,end))

% 
% figure()
% for i = 1:size(v0_vec,1)
%     semilogy(abs(v0_vec(i,end)-V(i,:)))
%     hold all
% end
% title('Error Approximate Q Learning')