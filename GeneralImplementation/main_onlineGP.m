close all
clear
clearvars -global;

load('mdp.mat')
load('policy.mat')


% mdp = mdp_asym;
% policy = policy_asym;

params.g = 9.81;       % gravity constant
params.m = 1;          % mass of pendulum
params.l = 1;          % length of pendulum
params.b = 0.2;        % damping coefficient
params.h = 0.02;%01*params.period;


X = [];

x = [pi/4-0.2, 0.0];
actions = [];
for i = 1:1000
    X = [X; x];
    state_index = discr(x, mdp.S);
    action = mdp.A(policy(state_index));
    actions = [actions;action];
    params.noise = 0*x(2);%2*sin(5*x(2))+0*randn();%*sin(10*x(1))
    x = rungekutta(x, action, params);
end

x_plusone = rungekutta(X(end,:), actions(end), params);
diffX = diff([X; x_plusone])/params.h;

y = diffX(:,2) - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));
x = X;



global net gpopt ep;

% General information
allBV  = 500:10:500;	      % BV set sizes
totExp = 10;		      % number of experiments for EACH BV set
totHyp = 15;		      % Number of EM-steps in learning model
% hyperparameters
totSCG = 4;		      % number of SCG steps
nTest  = 225;


% the type of and magnitude (or inverse magn.) of the additive noise.
nType  = 'posexp';	      % noise type: gauss, laplace, posexp
sig02  = 0;		      % noise variance

% TEST/VISUALISATION data
% [xTest,yTest] = sincdata(nType,nTest,0,[],1);
x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
[X1,X2] = meshgrid(x1,x2);
xTest = [X1(:) X2(:)];

nTrain = 0;

figure(1);
indBV = 0;

for maxB = allBV;
    indBV = indBV + 1;
    fprintf('\n %4d :\n\n', allBV(indBV));
    
    
    %     [xTrain, yTrain] = sincdata(nType,nTrain,sig02,[],0);
    % initialising the GP
    ogp(2, ...	      	      % input dimension
        1, ...	      	      % output dimension
        'sqexp', ...        % kernel type
        log([1/100 1/100 10 1]));    % kernel parameter
    % Assigning a Likelihood.
    % For regression the choices are 'reg_lapl' and 'reg_gauss'.
    ogpinit(@c_reg_gauss,...   % address for log-evid and its ders.
        10,...	      % likelihood (hyper-) parameters
        @em_exp);	      % adjusting likelihood parameters
    net.thresh    = 1e-3;     % the admission threshold for new BVs
    net.maxBV     = maxB;
    net.proj      = 0;
    
    % HYPERPRIORS used in hyperparameter optimisation.
    ogphypcovpar(1e-1);
    
    % Initialising GPOPT - optimisation and display options
    gpopt       = defoptions; % NO log-average
    gpopt.ptest = 0;	      % YES test error computation
    gpopt.xtest = xTest;      % the test inputs
    %     gpopt.ytest = yTest;      % desired outputs
    gpopt.disperr=0;
    gpopt.erraddr={@err_abs}; % function that measures the test error
    gpopt.freq  = 100;     % frequency of measuring the errors
    
    gpopt.postopt.isep  = 1;  % USING the TAP/EP algorithm
    gpopt.postopt.itn   = 2;  % number of EP iteration with CHANGING BVs
    gpopt.postopt.fixitn= 1;  % FIXING the BV set.
    
    % the control of HYPPAR optimisation
    gpopt.covopt.opt    = foptions;% default options
    gpopt.covopt.opt(1) = 1;  % display values
    gpopt.covopt.opt(9) = 1;  % CHECK/NOT gradients
    gpopt.covopt.opt(2) =1e-7;% precision in computing f. value
    gpopt.covopt.opt(3) =1e-7;% display values
    gpopt.covopt.opt(14)=totSCG;% number of iterations
    gpopt.covopt.fnopt ='conjgrad';
    
    % setting up indices
    lE = 0;
    % training the GP
    for bInd= 1:totExp;	      % iteration for different datasets
        % generating data
        nTrain = nTrain+100;
        xTrain = X(1:nTrain,:);
        yTrain = y(1:nTrain);
        
        for iHyp = 1:totHyp;
            ogptrain(xTrain,yTrain);
            % recomputation of the posterior is NEEDED since OGPTRAIN changed
            % the process parameters.
            ogpreset;
            ogppost(xTrain,yTrain);
            
            [meanT, varT] = ogpfwd(xTest);
            stdT          = sqrt(varT);
            meanBV        = ogpfwd(net.BV);
        end;
        
        m = reshape(meanT,15,15);
        s2 = reshape(stdT,15,15);
        figure(1)
        surf(x1,x2',m)
        hold all
        surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
        surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
        title('GP posterior')
        xlabel('x1')
        ylabel('x2')
        zlabel('$$\hat{d}$$','Interpreter','Latex')
        plot3(xTrain(:,1),xTrain(:,2),yTrain,'o')
        drawnow;
        
    end;
    
    
end;


