clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILDING MDP FOR THE INVERTED PENDULUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Physical pendulum parameters
pendulum.nstates = 2;
pendulum.params.g = 9.81;               % gravity constant
pendulum.params.m = 1;                  % mass of pendulum
pendulum.params.l = 1;                  % length of pendulum
pendulum.params.h = 0.1;                % sample time
pendulum.params.noise = 0.0;            % standard deviation input disturbance


% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 15;
x2_bounds = [-5,5];
x2_steps = 15;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];
pendulum.grid.input_bounds = [-10; 10];
pendulum.grid.input_steps =  3;
pendulum.grid.varyspacing = 0;

pendulum.purpose = 'forValueIteration';
pendulum.reward.type = 'exponential';   %exponential reward

mdp = BuildMDP(pendulum);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATING INITIAL SAFE SET BASED ON DISTURBANCE ESTIMATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d0 = 1;                                %conservative disturbance estimate
init = pi/2;

dMax{1} = d0*ones(x1_steps, x2_steps);
dMin{1} = -dMax{1};

count = 1;
maxcount = 5;

rows = ceil(sqrt(maxcount));
cols = ceil(maxcount / rows);

GPfigure=figure;
SafeFigure = figure;
SpeedyFigure = figure;

while count <=maxcount
    figure(SafeFigure)
    fprintf('================Safe Set Calculation=====================\n')
    [S0, u_opt] = safeController(mdp, pendulum, dMin{count}, dMax{count}, init, SafeFigure);
    
    safeStates{count} = S0>0;
    safeControl{count} = u_opt(:);
    %
    boundaryMatrix = conv2(double(safeStates{count}), ones(3)/9, 'same');
    reallySafeStates{count} = boundaryMatrix(:)>=0.75;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SAFE SPEEDY Q-LEARNING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% TODO: Recursive Gaussian, sample time
    
    
    fprintf('================Speedy Q-learning====================\n')
    alpha       = 1;   % learning rate
    gamma       = 0.9;   % discount factor
    epsilon     = 0.1;  % probability of a random action selection
    
    Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));
    Q_prev = Q;
    
    
    steps = 1000;
    states = zeros(steps,2);
    actions = zeros(steps,1);
    vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps);
    rewardperepisode = [];
    visited = zeros(size(mdp.S,1), size(mdp.A,1));
    
    
    x1 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
    x2 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
    x = [x1,x2];
    R = 0;
    cum_reward= 0;
    state = discretize(x, mdp.S);
    
    for j = 1:steps
        states(j,:) = x;
        %choose action
        if safeStates{count}(state)
            action = egreedy(state,epsilon,Q);
            u = mdp.A(action);
        else
            u = safeControl{count}(state);
            action = find(mdp.A==u);
        end
        actions(j) = u;
        visited(state,action) = visited(state,action)+1;
        %apply control
        x_prime = rungekutta(x, u, pendulum.params);
        s_prime = discretize(x_prime, mdp.S);
        
        R = mdp.R(state,action);
        cum_reward = cum_reward + R;
        temp = Q;
        
        Q(state,action) =  Q(state,action) + alpha/visited(state,action) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state,action) )+ ...
            (1-alpha/visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        
        state = s_prime;
        x = x_prime;
        Q_prev = temp;
        if abs(x') <= 1.5* pendulum.grid.state_bounds(:,2)
            x = x_prime;
        else
            x1 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
            x2 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
            x = [x1,x2];
        end
        
        
        V = max(Q,[],2);
        vecV(:,j) = V;
        if mod(j,100)==0
            disp(['Episode ' num2str(j)])
        end
    end

    opt_params.gamma = gamma;
    opt_params.epsilon = 0.0001;
    
    
    [policy, n, v0_vec] = valueiteration(mdp, opt_params);
    

    figure(SpeedyFigure)
    
    subplot(rows, cols, count)
    for i = 1:size(v0_vec,1)
        semilogy(abs(v0_vec(i,end)-vecV(i,:)))
        hold all
    end
    title('Error Speedy Q Learning')
    hold off
    
    fprintf('===========Disturbance Estimation with GP=====================\n')
    
    [statesGP, idx] = datasample(states,50,1);
    actionsGP = actions(idx);
    [mean, stdev, disturbance] = GaussianProcess(mdp, statesGP, actionsGP,pendulum.params);
    
    
    figure(GPfigure)
    subplot(rows, cols, count)
    surf(unique(mdp.S(:,1)),unique(mdp.S(:,2)),mean)
    hold all
    % surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    % surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    title('GP posterior')
    xlabel('x1')
    ylabel('x2')
    zlabel('$$\hat{d}$$','Interpreter','Latex')
    plot3(statesGP(:,1),statesGP(:,2),disturbance,'o')
    hold off
    count = count+1;
    
    dMax{count}= mean+2*sqrt(stdev);
    dMin{count} = mean-2*sqrt(stdev);
    
    
    
end