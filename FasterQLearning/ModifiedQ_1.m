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

ratio = ceil(pendulum.params.h_learn/pendulum.params.h_safe);

% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 19;
x2_bounds = [-5,5];
x2_steps = 19;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];
pendulum.grid.input_bounds = [-25; 25];
pendulum.grid.input_steps =  5;
pendulum.grid.varyspacing = 0;

pendulum.purpose = 'forValueIteration';
pendulum.reward.type = 'exponential';   %exponential reward
        % standard deviation input disturbance

        pendulum.params.noise = zeros(x1_steps*x2_steps,1);
mdp = BuildMDP(pendulum);

pendulum.params.noise = 0.5*sin(5*mdp.S(:,1))+0.0*randn(x1_steps*x2_steps,1);    

mdp = BuildMDP(pendulum);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count = 1;
maxcount = 4;
rows = ceil(sqrt(maxcount));
cols = ceil(maxcount / rows);
GPfigure=figure;
SafeFigure = figure;
SpeedyFigure = figure;
Simulation = figure;

%====================================
% Initialize Safe Set Calculation

d0 = 13.0;                                %conservative disturbance estimate
init = 0.9*pi;
dMax = d0*ones(x1_steps, x2_steps);
dMin = -dMax;
safeStates = cell(1,maxcount);
safeControl = cell(1,maxcount);
%====================================
% Initialize Speedy Q Learning

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection
steps = 10000;
vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps*maxcount);
visited = zeros(size(mdp.S,1), size(mdp.A,1));
chosesafe = cell(1,maxcount);

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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    if count > 1
        reset = 1;
    else
        reset = 0;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                Main Loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for n=1:steps-1
        
        action = egreedy(state,epsilon,Q);
        u_learn = mdp.A(action);
        pendulum.params.h = pendulum.params.h_safe;
        state_t = state;
        for k = 1:ratio
%             Safety loop
            if safeStates{count}(state)
                u = u_learn;
            else
                u = safeControl{count}(state);
                safeaction = find(mdp.A==u);
                chosesafe{count}(state,safeaction) = chosesafe{count}(state,safeaction)+1;
            end
            if reset
                disp('Reset')
                states(k + ratio*((n-1)+steps*(count-1)),:) = [NaN NaN];
                actions(k + ratio*((n-1)+steps*(count-1))) = NaN;
                reset = 0;
            else
                states(k + ratio*((n-1)+steps*(count-1)),:) = x;
                actions(k + ratio*((n-1)+steps*(count-1))) = u;
            end
            x = rungekutta(x, u, pendulum.params, state);
            state = discr(x, mdp.S);
        end
        pendulum.params.h = pendulum.params.h_learn;
        
        visited(state_t,action) = visited(state_t,action)+1;
%         apply control
        x_prime = x;
        s_prime = state;
        
        if B(state_t, action) <= t_star
            L(state_t, action) = 1;
        end
        
        if L(state_t, action) == 1
            if C(state_t, action) == 0
                B(state_t, action) = n;
            end
            if safeStates{count}(state_t)
                R = mdp.R(state_t,action);%+ 1/(1+visited(state_t,action));
            else
                R = -1;
            end
            C(state_t, action) = C(state_t, action) + 1;
            U(state_t, action) = U(state_t, action) + R + gamma*max(Q(s_prime, :));
            
            if C(state_t, action) == m(state_t, action)
                q = U(state_t, action)./m(state_t, action);
                if abs(Q(state_t, action) - q) >= epsilon_1
                    Q(state_t, action) = q;
                    t_star = n;
                elseif B(state_t, action) > t_star
                    fprintf('Discarding update.\n');
                    L(state_t, action) = 0;
                end
                
                U(state_t, action) = 0;
                C(state_t, action) = 0;
                m(state_t, action) = ceil(min(1.02*m(state_t, action) + 1, 500));
                epsilon_1 = min(epsilon_1*1.1, epstarget);
            end
        end
        state = s_prime;
        
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
        vecV(:,n+steps*(count-1)) = V;
        if mod(n,1000)==0
            disp(['Episode ' num2str(n)])
        end
    end
    
    fprintf('.....Calculating logarithmic error......\n')
    opt_params.gamma = gamma;
    opt_params.epsilon = 0.0001;
    
    [policy_true, p, v0_vec] = valueiteration(mdp, opt_params);
    
    figure(SpeedyFigure)
    safe_inds = find(safeStates{count}>0);
    subplot(rows, cols, count)
    for i = 1:size(safe_inds)
        semilogy(abs(v0_vec(safe_inds(i),end)-vecV(safe_inds(i),1:n+steps*(count-1))))
        hold all
    end
    title('Error Speedy Q Learning')
    xlabel('Episodes')
    ylabel('Error')
    xlim([0; n+steps*(count-1)])
    drawnow;
    
    fprintf('===========Disturbance Estimation with GP=====================\n')
    
    fprintf('.....Sampling states......\n')
    pendulum.params.h = pendulum.params.h_safe;
%     x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand(1000,1);
%     x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand(1000,1);
%     x_grid = [x01, x02];
%     idx = knnsearch(states, x_grid);
    
    idx = randi([1 n+steps*(count-1)-1],1,1000);
    idx = idx(idx<length(states));
    idx_faulty = isnan(states(idx)) | isnan(states(idx+1));
    idx(idx_faulty) = [];
%     idx = idx';
    statesGP = states([idx;idx+1],:);
disp(length(statesGP))
%     idx = idx(2<idx & idx<length(states));
%     idx_faulty = isnan(states(idx-1))| isnan(states(idx)) | isnan(states(idx+1));
%     idx(idx_faulty) = [];
%     idx = idx';
%     statesGP = states([idx-1;idx;idx+1],:);

    actionsGP = actions(idx);
    actionsGP = actionsGP(:);
    
    fprintf('.....Calculating Gaussian Process......\n')
    [mean, std, disturbance] = GaussianProcess(mdp, statesGP, actionsGP,pendulum.params);
    
    figure(GPfigure)
    subplot(rows, cols, count)
%     surf(x1,x2',mean)
%     hold all
% %      plot3(statesGP(2:3:end-1,1),statesGP(2:3:end,2),disturbance,'o')
%     plot3(statesGP(1:2:end-1,1),statesGP(1:2:end,2),disturbance,'o')
%     surf(x1,x2,mean+3*sqrt(reshape(std,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
%     surf(x1,x2,mean-3*sqrt(reshape(std,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
%     title('GP posterior')
%     xlabel('x1')
%     ylabel('x2')
%     zlabel('$$\hat{d}$$','Interpreter','Latex')
%% Create 3D grid containing distance to closest surface
nx = 19;
nz = 19;
ny = 19;

[x,y] = meshgrid(x1,x2);
z1 = mean-3*sqrt(reshape(std,sqrt(length(mdp.S)),sqrt(length(mdp.S))));
z2 = mean+3*sqrt(reshape(std,sqrt(length(mdp.S)),sqrt(length(mdp.S))));

%% Create 3D grid containing distance to closest surface
[y3,x3,z3] = ndgrid(x2',x1',linspace(-5,5,nz));
v3 = zeros(size(z3));
for r=1:ny
    for c=1:nx
        for s=1:nz
            d1 = z3(r,c,s) - z1(r,c);
            d2 = z2(r,c) - z3(r,c,s);
            if d1 < 0
                v3(r,c,s) = d1;
            elseif d2 < 0
                v3(r,c,s) = d2;
            else
                v3(r,c,s) = min(d1,d2);
            end
        end
    end
end
%% Create isosurface

% p = [patch(isosurface(x3,y3,z3,v3,0)), ...
%      patch(isocaps(x3,y3,z3,v3,0))];
isocaps(x3,y3,z3,v3,0);
% isonormals(x3,y3,z3,v3,p(1)) 
% set('FaceColor',[227 229 227]/255)
% set(p'EdgeColor','none')
% set(p,'FaceLighting','gouraud')
colormap('jet')
view(3)
camlight right
hold on
surf(x,y,z1)
surf(x,y,mean)
colormap('jet')
surf(x,y,z2)
plot3(statesGP(1:2:end-1,1),statesGP(1:2:end,2),disturbance,'o', 'MarkerSize', 8)
zlabel('$$\hat{d}$$','Interpreter','Latex')
xlabel('x_1')
ylabel('x_2')

drawnow;
count = count+1;

dMax=  0.5+(mean+3*sqrt(std));
dMin = -0.5+(mean-3*sqrt(std));
end




[~,policy_est] = max(Q,[],2);
Policy = figure;


marker = 90*ones(size(policy_true));
figure(Policy); clf;colormap('jet')
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
    end
end
figure(Simulation)
hold all
for i = 1:length(trajectories)
    plot(trajectories{i}(:,1),trajectories{i}(:,2))
    plot(trajectories{i}(1,1),trajectories{i}(1,2),'r.', 'MarkerSize',15);
    plot(trajectories{i}(end,1),trajectories{i}(end,2),'k.', 'MarkerSize',15);
end
    contour(g.xs{1}, g.xs{2}, S0, [0,0], 'Color', [98 146 46]/255);