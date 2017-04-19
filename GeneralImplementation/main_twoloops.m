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
pendulum.params.h_safe = 0.02;                % sample time
pendulum.params.h_learn = 0.2;
pendulum.params.h= pendulum.params.h_learn;
pendulum.params.noise = 0.0;            % standard deviation input disturbance

ratio = ceil(pendulum.params.h_learn/pendulum.params.h_safe);

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

dMax = d0*ones(x1_steps, x2_steps);
dMin = -dMax;

count = 1;
maxcount = 4;

rows = ceil(sqrt(maxcount));
cols = ceil(maxcount / rows);

GPfigure=figure;
SafeFigure = figure;
SpeedyFigure = figure;
%======================
% GP Init

x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));

[X1,X2] = meshgrid(x1,x2);
xTest = [X1(:) X2(:)];

% Speedy Q Learning Init

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection

Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));
Q_prev = Q;


steps = 40000;
states = zeros(steps*maxcount,2);
states_safe =  zeros(steps*maxcount*ratio,2);
actions = zeros(steps*maxcount,1);
actions_safe =  zeros(steps*maxcount*ratio,1);
vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps*maxcount);
rewardperepisode = [];
visited = zeros(size(mdp.S,1), size(mdp.A,1));


while count <=maxcount
    figure(SafeFigure)
    fprintf('================Safe Set Calculation=====================\n')
    [S0, u_opt] = safeController(mdp, pendulum, dMin, dMax, init, SafeFigure);
    
    safeStates{count} = S0>0;
    safeControl{count} = u_opt(:);
    %
    boundaryMatrix = conv2(double(safeStates{count}), ones(3)/9, 'same');
    reallySafeStates{count} = boundaryMatrix(:)>=0.75;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SAFE SPEEDY Q-LEARNING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    fprintf('================Speedy Q-learning====================\n')
    
    x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
    x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
    
    x = [x01,x02];
    R = 0;
    state = discretize(x, mdp.S);
    reset = 0;
    if count > 1
        reset = 1;
    end
    
    for j = 1:steps
        %         states(j+steps*(count-1),:) = x;
        
        action = egreedy(state,epsilon,Q);
        u_learn = mdp.A(action);
        %         actions(j+steps*(count-1)) = u_learn;
        pendulum.params.h = pendulum.params.h_safe;
        x_safe = x;
        state_safe = state;
        for k = 1:ratio
            %choose action
            if safeStates{count}(state_safe)
                u = u_learn;
            else
                u = safeControl{count}(state_safe);
                action = find(mdp.A==u);
            end
            if reset
                states_safe(k + ratio*((j-1)+steps*(count-1)),:) = [NaN NaN];
                actions_safe(k + ratio*((j-1)+steps*(count-1))) = NaN;
                reset = 0;
            else
                states_safe(k + ratio*((j-1)+steps*(count-1)),:) = x_safe;
                actions_safe(k + ratio*((j-1)+steps*(count-1))) = u;
            end
            x_safe = rungekutta(x_safe, u, pendulum.params);
            state_safe = discretize(x_safe, mdp.S);
        end
        pendulum.params.h = pendulum.params.h_learn;
        visited(state,action) = visited(state,action)+1;
        %apply control
        x_prime = x_safe;%rungekutta(x, u, pendulum.params);
        s_prime = state_safe;%discretize(x_prime, mdp.S);
        
        R = mdp.R(state,action);
        temp = Q;
        
        Q(state,action) =  Q(state,action) + alpha/visited(state,action) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state,action) )+ ...
            (1-alpha/visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        
        state = s_prime;
        x = x_prime;
        Q_prev = temp;
        if abs(x') <= 1.05* pendulum.grid.state_bounds(:,2)
            x = x_prime;
        else
            disp(j)
            reset = 1;
            x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
            x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
            x = [x01,x02];
        end
        
        
        V = max(Q,[],2);
        vecV(:,j+steps*(count-1)) = V;
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
        semilogy(abs(v0_vec(i,end)-vecV(i,1:j+steps*(count-1))))
        hold all
    end
    title('Error Speedy Q Learning')
    xlabel('Episodes')
    ylabel('Error')
    
    drawnow;
    
    %     fprintf('===========Disturbance Estimation with GP=====================\n')
    pendulum.params.h = pendulum.params.h_safe;
    [~, idx] = datasample(states_safe,1000,1);
    idx = idx(idx>2&idx<length(states_safe)-2);
    %
    %     statesGP = states_safe(1:1000,:);
    %     actionsGP = actions_safe(1:1000,:);
    
    idx_faulty = isnan(states_safe(idx)) | isnan(states_safe(idx+1));
    idx(idx_faulty) = [];
    statesGP = states_safe([idx;idx+1],:);
    actionsGP = actions_safe(idx);
    actionsGP = actionsGP(:);
    
    %     Fourth order central differences
    
    %     statesGP = states_safe([idx-2;idx-1;idx;idx+1;idx+2],:);
    %     actionsGP = actions_safe(idx);
    %     actionsGP = actionsGP(:);
    
    % %     Central differences
    %     idx_faulty = isnan(states_safe(idx)) | isnan(states_safe(idx+1))| isnan(states_safe(idx-1));
    %     idx(idx_faulty) = [];
    %         statesGP = states_safe([idx-1;idx;idx+1],:);
    %         actionsGP = actions_safe(idx);
    %         actionsGP = actionsGP(:);
    %
    [m, s, disturbance] = GaussianProcess(mdp, statesGP, actionsGP,pendulum.params);
    
    figure(GPfigure)
    surf(x1,x2',m)
    hold all
    %     plot3(statesGP(:,1),statesGP(:,2),disturbance,'o')
    %         plot3(statesGP(2:3:end-1,1),statesGP(2:3:end-1,2),disturbance,'o')
    plot3(statesGP(1:2:end-1,1),statesGP(2:2:end,2),disturbance,'o')
    %     plot3(statesGP(3:5:end-2,1),statesGP(3:5:end,2),disturbance,'o')
    %     surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    %     surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    title('GP posterior')
    xlabel('x1')
    ylabel('x2')
    zlabel('$$\hat{d}$$','Interpreter','Latex')
    drawnow;
    count = count+1;
    
    dMax= m+2*sqrt(s);
    dMin = m-2*sqrt(s);
    %
    %
    
end