function [policy_vec, v_vec, n] = policyiteration(mdp,params)

% Discount factor
gamma = params.gamma;

r = mdp.R;
S = size(mdp.S,1);
% A = size(mdp.A,1);
P = mdp.P;

n = 0;
v = zeros(S, 1);
p1 = ones(S, 1);
p1(1, 1) = 2;
p0 = ones(S, 1);

v_vec = v;
policy_vec = p0;

while  ~isequal(p1, p0)
    n = n+1;
    p1 = p0;
    
    r_policy = zeros(S, 1);
    P_policy = zeros(S, S);
    
    for s=1:S
        r_policy(s, 1) = r(s, p1(s));
        for k=1:S
            P_policy(s, k) = P{s}(k, p1(s));
        end
    end
    
    v = (eye(S) - gamma*P_policy)\r_policy;
    v_vec = [v_vec v];
    
    for s=1:S
        tmp_s = r(s, :)' + gamma*P{s}'*v;
        [~, p0(s)] = max(tmp_s);
    end
    if mod(n,50) == 0
        fprintf('Iter: %d\n',n);
    end
    policy_vec = [policy_vec p0];
    
end
