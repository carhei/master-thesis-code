close all
clear
clearvars -global;

load('mdp.mat')
load('policy.mat')


params.g = 9.81;       % gravity constant
params.m = 1;          % mass of pendulum
params.l = 1;          % length of pendulum
params.b = 0.2;        % damping coefficient
params.h = 0.02;%01*params.period;


X = [];

x = [pi/4-0.2, 1.0];
actions = [];
for i = 1:1000
    X = [X; x];
    state_index = discr(x, mdp.S);
    action = mdp.A(policy(state_index));
    actions = [actions;action];
    params.noise = 0;%2*sin(10*x(1))+0*randn();%*sin(10*x(1))
    x = rungekutta(x, action, params);
end

x_plusone = rungekutta(X(end,:), actions(end), params);
diffX = diff([X; x_plusone])/params.h;

y = diffX(:,2) - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));

% X = randn(1000,2);
% y = 2*sin(5*X(:,1));

global net gpopt ep;


x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
% 
% x1 = linspace(-2,2,30);
% x2 = linspace(-2,2,30);
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


nTrain = 0;
% setting up the plots
% performing iterations
for i = 1:10
    
    % generating data
    nTrain = nTrain+100;
    
    xTrain = X(1:nTrain,:);
    yTrain = y(1:nTrain);
    % resetting the posterior is NEEDED
    ogpreset;
    % more than a single iteration to fit the data
    for iFit = 1:10;
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
    s2 = reshape(stdT,length(x1),length(x2));
    figure(1)
    surf(x1,x2',m)
    hold all
%     surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
%     surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
    title('GP posterior')
    xlabel('x1')
    ylabel('x2')
    zlabel('$$\hat{d}$$','Interpreter','Latex')
    plot3(xTrain(:,1),xTrain(:,2),yTrain,'o')
    drawnow;
end;

