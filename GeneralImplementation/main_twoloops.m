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
pendulum.params.noise = 2.0;            % standard deviation input disturbance

ratio = ceil(pendulum.params.h_learn/pendulum.params.h_safe);

% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 15;
x2_bounds = [-5,5];
x2_steps = 15;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];
pendulum.grid.input_bounds = [-10; 10];
pendulum.grid.input_steps =  5;
pendulum.grid.varyspacing = 0;

pendulum.purpose = 'forValueIteration';
pendulum.reward.type = 'exponential';   %exponential reward

mdp = BuildMDP(pendulum);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count = 1;
maxcount = 3;
rows = ceil(sqrt(maxcount));
cols = ceil(maxcount / rows);
GPfigure=figure;
SafeFigure = figure;
SpeedyFigure = figure;
Simulation = figure;

%====================================
% Initialize Safe Set Calculation

d0 = 3;                                %conservative disturbance estimate
init = 0.8*pi;
dMax = d0*ones(x1_steps, x2_steps);
dMin = -dMax;
safeStates = cell(1,maxcount);
safeControl = cell(1,maxcount);
%====================================
% Initialize Speedy Q Learning

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.5;  % probability of a random action selection
Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));
Q_prev = Q;
steps = 40000;
vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps*maxcount);
visited = zeros(size(mdp.S,1), size(mdp.A,1));
chosesafe = cell(1,maxcount);
%====================================
% Initialize Gaussian Process

x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
[X1,X2] = meshgrid(x1,x2);
xTest = [X1(:) X2(:)];
states =  zeros(steps*maxcount*ratio,2);
actions =  zeros(steps*maxcount*ratio,1);


%% MAIN LOOP

while count <=maxcount
    fprintf('================Safe Set Calculation=====================\n')
    [S0, u_opt,g] = safeController(mdp, pendulum, dMin, dMax, init);
    figure(SafeFigure)
    subplot(rows, cols, count)
    contour(g.xs{1}, g.xs{2}, S0, [0,0], 'r');
    title('Safe Set via HJI')
    xlabel('x1')
    ylabel('x2')
    drawnow;
    safeStates{count} = S0>0;
    safeControl{count} = u_opt(:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SAFE SPEEDY Q-LEARNING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('================Safe Speedy Q-learning====================\n')
    unsafe = 1;
    chosesafe{count} = zeros(size(mdp.S,1), size(mdp.A,1));
    while unsafe
        x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
        x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
        x = [x01,x02];
        state = discr(x, mdp.S);
        if safeStates{count}(state)
            unsafe = 0;
        end
    end
    R = 0;
    
    if count > 1
        reset = 1;
    else
        reset = 0;
    end    
    for j = 1:steps
        action = egreedy(state,epsilon,Q);
        u_learn = mdp.A(action);
        pendulum.params.h = pendulum.params.h_safe;
        state_start = state;
        for k = 1:ratio
            %Safety loop
            if safeStates{count}(state)
                u = u_learn;
            else
                u = safeControl{count}(state);
                safeaction = find(mdp.A==u);
                chosesafe{count}(state,safeaction) = chosesafe{count}(state,safeaction)+1;
            end
            if reset
                states(k + ratio*((j-1)+steps*(count-1)),:) = [NaN NaN];
                actions(k + ratio*((j-1)+steps*(count-1))) = NaN;
                reset = 0;
            else
                states(k + ratio*((j-1)+steps*(count-1)),:) = x;
                actions(k + ratio*((j-1)+steps*(count-1))) = u;
            end
            x = rungekutta(x, u, pendulum.params);
            state = discr(x, mdp.S);
        end
        pendulum.params.h = pendulum.params.h_learn;
        visited(state_start,action) = visited(state_start,action)+1;
        %apply control
        x_prime = x;
        s_prime = state;
        if safeStates{count}(state_start)
            R = mdp.R(state_start,action)/(1+visited(state_start,action));
        else
            R = -1;
        end
        temp = Q;
        Q(state_start,action) =  Q(state_start,action) + alpha/visited(state_start,action) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state_start,action) )+ ...
            (1-alpha/visited(state_start,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        state = s_prime;
        x = x_prime;
        Q_prev = temp;
        if abs(x') <= 1.05* pendulum.grid.state_bounds(:,2)
            x = x_prime;
        else
            reset = 1;
            unsafe = 1;
            while unsafe
                x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
                x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
                x = [x01,x02];
                state = discr(x, mdp.S);
                if safeStates{count}(state)
                    unsafe = 0;
                end
            end
        end
        
        V = max(Q,[],2);
        vecV(:,j+steps*(count-1)) = V;
        if mod(j,1000)==0
            disp(['Episode ' num2str(j)])
        end
    end
    fprintf('.....Calculating logarithmic error......\n')
    opt_params.gamma = gamma;
    opt_params.epsilon = 0.0001;
    
    [policy_true, n, v0_vec] = valueiteration(mdp, opt_params);
    
    figure(SpeedyFigure)
    
    subplot(rows, cols, count)
    for i = 1:size(v0_vec,1)
        semilogy(abs(v0_vec(i,end)-vecV(i,1:j+steps*(count-1))))
        hold all
    end
    title('Error Speedy Q Learning')
    xlabel('Episodes')
    ylabel('Error')
    xlim([0; j+steps*(count-1)])
    drawnow;
    
    fprintf('===========Disturbance Estimation with GP=====================\n')
    
    fprintf('.....Sampling states......\n')
    pendulum.params.h = pendulum.params.h_safe;
    x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand(1000,1);
    x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand(1000,1);
    x_grid = [x01, x02];
    idx = knnsearch(states, x_grid);
    idx = idx(idx<length(states));
    idx_faulty = isnan(states(idx)) | isnan(states(idx+1));
    idx(idx_faulty) = [];
    idx = idx';
    statesGP = states([idx;idx+1],:);
    actionsGP = actions(idx);
    actionsGP = actionsGP(:);
    
    fprintf('.....Calculating Gaussian Process......\n')
    [m, s, disturbance] = GaussianProcess(mdp, statesGP, actionsGP,pendulum.params);
    
    figure(GPfigure)
    subplot(rows, cols, count)
    surf(x1,x2',m)
    hold all
    plot3(statesGP(1:2:end-1,1),statesGP(2:2:end,2),disturbance,'o')
    surf(x1,x2,m+2*sqrt(reshape(s,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    surf(x1,x2,m-2*sqrt(reshape(s,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    title('GP posterior')
    xlabel('x1')
    ylabel('x2')
    zlabel('$$\hat{d}$$','Interpreter','Latex')
    drawnow;
    count = count+1;
    
    dMax=  0.5+(m+3*sqrt(s));
    dMin = -0.5+(m-3*sqrt(s));
end




[~,policy_est] = max(Q,[],2);
Policy = figure;

marker = 90*ones(size(policy_true));

figure(Policy); clf; colormap('jet')
xlabel('angle (rad)')
ylabel('rate (rad/s)')
zlabel('torque (Nm)')
title('Policy comparison')
axis equal
subplot(1,2,1)
scatter3(mdp.S(:,1), mdp.S(:,2), policy_est, marker, policy_est, 'filled');
view(2)
subplot(1,2,2)
scatter3(mdp.S(:,1), mdp.S(:,2), policy_true, marker, policy_true, 'filled');
view(2)

k = 1;
trajectories = [];

[rows, cols] = find(isnan(states));
Indices = [0 unique(rows)' size(states,1)+1];

for i = 1:length(Indices) - 1
    Temp = states(Indices(i) + 1:Indices(i + 1) - 1,:);
    if (~isempty(Temp))
        trajectories{k} = Temp;
        k = k+1;
    end;
end
figure(Simulation)
hold all
for i = 1:length(trajectories)
plot(trajectories{i}(:,1),trajectories{i}(:,2))
plot(trajectories{i}(1,1),trajectories{i}(1,2),'r.', 'MarkerSize',15);
plot(trajectories{i}(end,1),trajectories{i}(end,2),'k.', 'MarkerSize',15);
end
line([-pi/2 -pi/2], [-8.3 8.3]); 
line([pi/2 pi/2], [-8.3 8.3]); 