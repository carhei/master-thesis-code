function [mean,stdev,y] = GaussianProcess(mdp,X,actions, params)

% t = linspace(0,max(X(:,2)),size(X,1));
% 
% f = spline(t,X(:,2));
% fp = fnder(f);
% fnplt(f)
% fnplt(fp)
% dotX = fnval(fp,t);
% y = dotX' - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));


% dotX = diff([X; X(end,:)])/params.h;

% d1 = dotX(:,1) - X(:,2);
% d2 = dotX(:,2) - (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));

% y = next_state(:,2);
x_plusone = rungekutta(X(end,:), actions(end), params);
x_plustwo = rungekutta(x_plusone, actions(end), params);

X_extend = [X; x_plusone; x_plustwo];
diffX = (X_extend(2:end-1,:)-X_extend(1:end-2,:))/params.h;
diff2X = (X_extend(3:end,:)-2*X_extend(2:end-1,:)+X_extend(1:end-2,:))*(params.h/2)/params.h^2;

dotX = diffX-diff2X;

y = dotX(:,2)- (1/(params.m*params.l)*actions+params.g/params.l*sin(X(:,1)));


x = X;

x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
[X1,X2] = meshgrid(x1,x2);
x_test = [X1(:) X2(:)];

meanfunc = @meanZero;                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -50, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);


mean = reshape(mu, length(x2), length(x1));
stdev = reshape(s2,length(x2), length(x1));
