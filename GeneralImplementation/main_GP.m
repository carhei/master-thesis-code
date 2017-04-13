close all
clear

load('mdp.mat')
load('policy.mat')


% mdp = mdp_asym;
% policy = policy_asym;

params.g = 9.81;       % gravity constant
params.m = 1;          % mass of pendulum
params.l = 1;          % length of pendulum
params.b = 0.2;        % damping coefficient
params.h = 0.001;%01*params.period;


X = [];

x = [pi/4-0.2, 2.0];
actions = [];
for i = 1:2000
    X = [X; x];
    state_index = discretize(x, mdp.S);
    action = mdp.A(policy(state_index));
    actions = [actions;action];
    params.noise = 2*sin(10*x(2))+0*randn();%*sin(10*x(1))
    x = rungekutta(x, action, params);
end

dotX = diff([X; X(end,:)])/params.h;

d1 = dotX(:,1) - X(:,2);
d2 = dotX(:,2) - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));

y = d2;
x = X;

next_state = rungekutta(X, actions, params);
d1 = next_state(:,1)- X(:,2);
y = next_state(:,2)- (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));



% xs1 = linspace(-pi/4,pi/4,55);
% xs2 = linspace(-5,5,55);
% xs1 = linspace(-2+min(X(:,1)), 2+max(X(:,1)), 50)';
% xs2 = linspace(-2+min(X(:,2)), 2+max(X(:,2)), 50)';
% xs = [xs1,xs2];
% [X1,X2] = meshgrid(xs1,xs2);
% x_test = [X1(:) X2(:)];
% 
x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
[X1,X2] = meshgrid(x1,x2);
x_test = [X1(:) X2(:)];

meanfunc = @meanZero;                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);

% m = reshape(mu, 55, 55);
% surf(xs1,xs2,m)
% hold all
% plot3(X(:,1),X(:,2),d2,'o')

m = reshape(mu, length(x2), length(x1));
s = reshape(s2,length(x2), length(x1));

figure()
surf(x1,x2',m)
hold all
% surf(x1,x2,m+2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
% surf(x1,x2,m-2*sqrt(reshape(s2,sqrt(length(mdp.S)),sqrt(length(mdp.S)))))
title('GP posterior')
xlabel('x1')
ylabel('x2')
zlabel('$$\hat{d}$$','Interpreter','Latex')

plot3(X(:,1),X(:,2),y,'o')

Dmax = zeros(length(x1), length(x2));
Dmin = Dmax;
% 
for i = 1:length(x1)
    for j = 1:length(x2)
        maxD(i,j) = max(abs(m(i,j)+2*sqrt(s(i,j))),abs(m(i,j)-2*sqrt(s(i,j))));
    end
end

 save('/home/caro/Dokument/KTH/MasterThesis/Implementation/SafeControl/maxD.mat','maxD')

