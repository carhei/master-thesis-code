% Value Iteration
% By Sadegh Talebi
% November 6, 2016

function [policy, n, v0_vec] = valueiteration(mdp,params) 



S = size(mdp.S,1);
A = size(mdp.A,1);

r = mdp.R;

P = mdp.P;



% Discount factor
gamma = params.gamma;%.99;
epsilon = params.epsilon;% 0.0001;
n = 0;
v0 = zeros(S, 1);
v1 = ones(S, 1);
policy = zeros(S, 1);

v0_vec = v0; 
policy_vec = policy;


while max(abs(v1-v0)) >= epsilon*(1-gamma)/(2*gamma)
    n = n+1;
    v1 = v0;
    for s=1:S
        tmp_s = r(s, :)' + gamma*P{s}'*v1;
        [v0(s), policy(s)] = max(tmp_s);
    end
    
    v0_vec = [v0_vec v0];
    policy_vec = [policy_vec policy];
%     if mod(n,50) == 0
%         fprintf('Iter: %d\n',n);
%     end
    if n>5000
        break;
    end
    
end
