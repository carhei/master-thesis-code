function [ mdp ] = BuildMDP( system )
%BUILDMPD Summary of this function goes here
%   builds an MDP model of the given system with the given grid and reward
%   parameters



% STATES


for i = 1: system.nstates
    if system.grid.varyspacing
        length = (diff(system.grid.state_bounds(i,:)));
        fraction = ceil(system.grid.state_steps(i)/5);
        
        range1 = linspace(system.grid.state_bounds(i,1),system.grid.state_bounds(i,1)+1/4*length-fraction,fraction);
        range2 = linspace(system.grid.state_bounds(i,1)+1/4*length,system.grid.state_bounds(i,1)+3/4*length,system.grid.state_steps(i)-2*fraction);
        range3 = linspace(system.grid.state_bounds(i,1)+3/4*length+fraction,system.grid.state_bounds(i,2),fraction);
        range{i} = {[range1 range2 range3]};
    else
        range{i} = linspace(system.grid.state_bounds(i,1),system.grid.state_bounds(i,2),system.grid.state_steps(i));
    end
    
end


if system.grid.varyspacing
    range = [range{:}];
end

grids = cell(1,system.nstates);
[grids{:}] = ndgrid(range{:});
S = [];

for i = 1:system.nstates
    temp = reshape(grids{i},numel(grids{i}),1);
    S = [S,temp];
end

% ACTIONS


if system.grid.varyspacing
    length = (diff(system.grid.input_bounds(:)));
    fraction = ceil(system.grid.input_steps/5);
    
    range1 = linspace(system.grid.input_bounds(1),system.grid.input_bounds(1)+1/4*length-fraction,fraction);
    range2 = linspace(system.grid.input_bounds(1)+1/4*length,system.grid.input_bounds(1)+3/4*length,system.grid.input_steps-2*fraction);
    range3 = linspace(system.grid.input_bounds(1)+3/4*length+fraction,system.grid.input_bounds(2),fraction);
    A = [range1 range2 range3]';
    
else    
    A = linspace(system.grid.input_bounds(1),system.grid.input_bounds(2), system.grid.input_steps)';
end

% REWARDS


if strcmp(system.reward.type,'quadratic')
    R = 0;
    
    for i = 1:system.nstates
        R = R + system.params.h*system.reward.value(i)*(S(:,i)).^2;
        
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
    
else if strcmp(system.reward.type,'exponential')
        sigma = 0.5*(system.grid.state_bounds(2,2)-system.grid.state_bounds(2,1));
        R = 1-(1-exp(-sum(abs(S).^2,2)/(sigma^2)));
        R = R*ones(1,size(A,1));
        
        % %        S(S(:,1)>pi) || S(1)<-pi || x_prime(2)>4 || x_prime(2)<-4
    else
        
        disp('Possible reward types are exponential or quadratic.')
    end
    
end

% TRANSITION PROBABILITIES


if strcmp(system.purpose, 'forValueIteration')
    P = transitionprobs(S,A, system.params);
    
    mdp.P = P;
    
end

mdp.A = A;
mdp.S = S;

mdp.R = R;
end

