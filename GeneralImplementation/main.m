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
pendulum.params.h = 0.0001;                % sample time
pendulum.params.noise = 3.0;            % standard deviation input disturbance


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
maxcount = 3;

rows = ceil(sqrt(maxcount));
cols = ceil(maxcount / rows);

GPfigure=figure;
SafeFigure = figure;
SpeedyFigure = figure;
%======================
% GP Init

global net gpopt ep;

x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));

[X1,X2] = meshgrid(x1,x2);
xTest = [X1(:) X2(:)];

% initialising the GP
ogp(2, ...	      	      % input dimension
    1, ...	      	      % output dimension
    'sqexp', ...        % kernel type
    log([1 1 10 1]));
% HYPERPRIORS used in hyperparameter optimisation.
ogphypcovpar(5e-3);

% assigning a likelihood to the data
ogpinit(@c_reg_gauss,...      % address for log-evid and its ders.
    0.025, ...            % likelihood (hyper-) parameters
    @em_gauss);
net.thresh    = 1e-3;         % the admission threshold for new BVs
net.maxBV     = 60;

% Initialising GPOPT - optimisation and display options
gpopt       = defoptions; % NO log-average
gpopt.postopt.isep  = 1;      % USING the TAP/EP algorithm
gpopt.postopt.itn   = 2;      % number of EP iteration with CHANGING
gpopt.postopt.fixitn= 1;      % FIXING the BV set.
%
% % parameters to the (NETLAB) optimisation algorithm
gpopt.covopt.opt    = foptions;
gpopt.covopt.opt(1) =  -1;    % DISPLAY/NOT training values
gpopt.covopt.opt(9) =   0;    % CHECK/NOT   gradients
gpopt.covopt.opt(2) =1e-3;    % precision in computing optimum values
gpopt.covopt.opt(3) =1e-3;
gpopt.covopt.opt(14)=   floor(1.5*length(ogppak));    % # of steps
gpopt.covopt.fnopt  = 'conjgrad';



% Speedy Q Learning Init

alpha       = 1;   % learning rate
gamma       = 0.9;   % discount factor
epsilon     = 0.1;  % probability of a random action selection

Q = max(max(mdp.R))/(1-gamma)*ones(size(mdp.S,1), size(mdp.A,1));
Q_prev = Q;


steps = 40000;
states = zeros(steps*maxcount,2);
actions = zeros(steps*maxcount,1);
vecV = zeros(pendulum.grid.state_steps(1)*pendulum.grid.state_steps(2),steps*maxcount);
rewardperepisode = [];
visited = zeros(size(mdp.S,1), size(mdp.A,1));
xTrain = [];
yTrain = [];

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
       
    
    fprintf('================Speedy Q-learning====================\n')
    
    x01 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand;
    x02 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand;
    x = [x01,x02];
    R = 0;
    state = discr(x, mdp.S);
    
    for j = 1:steps
        states(j+steps*(count-1),:) = x;
        %choose action
        if safeStates{count}(state)
            action = egreedy(state,epsilon,Q);
            u = mdp.A(action);
        else
            u = safeControl{count}(state);
            action = find(mdp.A==u);
        end
        actions(j+steps*(count-1)) = u;
        visited(state,action) = visited(state,action)+1;
        %apply control
        x_prime = rungekutta(x, u, pendulum.params);
        s_prime = discr(x_prime, mdp.S);
        
        R = mdp.R(state,action);
        temp = Q;
        
        Q(state,action) =  Q(state,action) + alpha/visited(state,action) * ( R + gamma*max(Q_prev(s_prime,:)) - Q(state,action) )+ ...
            (1-alpha/visited(state,action)) * ( R + gamma*max(Q(s_prime,:)) - R - gamma*max(Q_prev(s_prime,:)));
        
        
        state = s_prime;
        x = x_prime;
        Q_prev = temp;
        if abs(x') <= 1.5* pendulum.grid.state_bounds(:,2)
            x = x_prime;
        else
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
    
    fprintf('===========Disturbance Estimation with GP=====================\n')
    
    [~, idx] = datasample(states,1000,1);
    idx = idx(idx>2&idx<length(states)-2);
    
    %     statesGP = states(1:1000,:);
    %         actionsGP = actions(1:1000,:);
    
    
    %             statesGP = states([idx;idx+1],:);
    %         actionsGP = actions(idx);
    %         actionsGP = actionsGP(:);
    %
    %   Fourth order central differences
    %         statesGP = states([idx-2;idx-1;idx;idx+1;idx+2],:);
    %         actionsGP = actions(idx);
    %         actionsGP = actionsGP(:);
    
%     Central differences
    statesGP = states([idx-1;idx;idx+1],:);
    actionsGP = actions(idx);
    actionsGP = actionsGP(:);
    %
%         [m, s, disturbance] = GaussianProcess(mdp, statesGP, actionsGP,pendulum.params);
%     
%     
%     dotX = (statesGP(3:3:end,:)-statesGP(1:3:end-2,:))/(2*pendulum.params.h);
%     distGP = dotX(:,2) - (1/(pendulum.params.m*pendulum.params.l)*actionsGP+pendulum.params.g/pendulum.params.l*sin(statesGP(2:3:end-1,1)));
%     
%     xTrain = statesGP(2:3:end-1,:);%[xTrain; statesGP(2:3:end-1,:)];
%     yTrain = distGP;%[yTrain; distGP];
    
    % resetting the posterior is NEEDED
    ogpreset;
    % more than a single iteration to fit the data
    for iFit = 1:5;
        ogptrain(xTrain,yTrain);
        ogpreset;
    end;
    % computing the posterior
    ogppost(xTrain,yTrain);
    
    [meanT, varT] = ogpfwd(xTest);
    stdT          = sqrt(varT);
    stdPr         = sqrt(varT + net.likpar);
    meanBV        = ogpfwd(net.BV);
    
    m = reshape(meanT,length(x1),length(x2));
    s = reshape(stdT,length(x1),length(x2));
    figure(GPfigure)
    surf(x1,x2',m)
    hold all
%     plot3(statesGP(2:3:end-1,1),statesGP(2:3:end-1,2),disturbance,'o')
    
    %     surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    %     surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    title('GP posterior')
    xlabel('x1')
    ylabel('x2')
    zlabel('$$\hat{d}$$','Interpreter','Latex')
    plot3(xTrain(:,1),xTrain(:,2),yTrain,'o')
    drawnow;
    count = count+1;
    
    dMax{count}= m+2*sqrt(s);
    dMin{count} = m-2*sqrt(s);
    
    
    
end