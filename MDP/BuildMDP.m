function [ mdp ] = BuildMDP( system )
%BUILDMPD Summary of this function goes here
%   Detailed explanation goes here


% STATES

for i = 1: system.nstates
    range{i} = linspace(system.grid.state_bounds(i,1),system.grid.state_bounds(i,2),system.grid.state_steps(i));
end


grids = cell(1,system.nstates);
[grids{:}] = ndgrid(range{:});
S = [];

for i = 1:system.nstates
    temp = reshape(grids{i},numel(grids{i}),1);
    S = [S,temp];
end

% ACTIONS


A = linspace(system.grid.input_bounds(1),system.grid.input_bounds(2), system.grid.input_steps)';


% REWARDS

R = 0;

for i = 1:system.nstates
    R = R + system.params.h*system.reward(i)*(S(:,i)).^2;
   
end


R = R*ones(1,size(A,1));


for i = 1:system.nstates
    x = S(:,i);
    xDel = system.edge_width*diff(system.grid.state_bounds(i,:));
    edge_reward = system.edge_reward(i);

    m = 3/xDel;
    yUpp = tanh(m*(x-system.grid.state_bounds(i,2)));
    yLow = tanh(-m*(x-system.grid.state_bounds(i,1)));
    smoothed_edge_reward = edge_reward*(2 + yLow + yUpp);
    R = R+ smoothed_edge_reward*ones(1,size(A,1));
end


R_input = system.params.h*system.input_reward*ones(size(S,1),1)*A'.^2;

R = R+R_input;


% TRANSITION PROBABILITIES

P = transitionprobs(S,A, system.params);


mdp.A = A;
mdp.S = S;
mdp.R = R;
mdp.P = P;
end

