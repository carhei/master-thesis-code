function [mean,stdev,y] = GaussianProcess(mdp,X,u, params)

%%%% DIFFERENT IDEAS

% SPLINES
% t = linspace(0,max(X(:,2)),size(X,1));
% f = spline(t,X(:,2));
% fp = fnder(f);
% fnplt(f)
% fnplt(fp)
% dotX = fnval(fp,t);
% y = dotX' - (1/(params.m*params.l)*u+params.g/params.l*sin(X(:,1)));


% HIGHER ORDER APPOX
% x = X;
% x_plusone = rungekutta(X(end,:), u(end), params);
% dotX = diff([X;x_plusone])/params.h;
% % d1 = dotX(:,1) - X(:,2);
% d2 = dotX(:,2) - (1/(params.m*params.l)*u+params.g/params.l*sin(X(:,1)));
% y = d2;



% X_extend = [X(1,:);X(1,:);X; X(end,:);X(end,:)];
% %
% dotX = (-X_extend(5:end,:)+8*X_extend(4:end-1,:)-8*X_extend(2:end-3,:)+X_extend(1:end-4,:))/(12*params.h);
%



% X_extend = [X; x_plusone; x_plustwo];
% diffX = (X_extend(2:end-1,:)-X_extend(1:end-2,:))/params.h;
% diff2X = (X_extend(3:end,:)-2*X_extend(2:end-1,:)+X_extend(1:end-2,:))*(params.h/2)/params.h^2;
%
% dotX = diffX-diff2X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% %
% dotX = (-X(5:5:end,:)+8*X(4:5:end-1,:)-8*X(2:5:end-3,:)+X(1:5:end-4,:))/(12*params.h);
% y = dotX(:,2)- (1/(params.m*params.l)*u+params.g/params.l*sin(X(3:5:end-2,1)));
%
% x = X(3:5:end,:);

% 
dotX = (X(2:2:end,:)-X(1:2:end-1,:))/params.h;
y = dotX(:,2) - (1/(params.m*params.l)*u+params.g/params.l*sin(X(1:2:end-1,1)));

x = X(1:2:end-1,:);
% 
% dotX = (X(3:3:end,:)-X(1:3:end-2,:))/(2*params.h);
% y = dotX(:,2) - (1/(params.m*params.l)*u+params.g/params.l*sin(X(2:3:end-1,1)));
% 
% x = X(2:3:end-1,:);


x1 = unique(mdp.S(:,1));
x2 = unique(mdp.S(:,2));
[X1,X2] = meshgrid(x1,x2);
x_test = [X1(:) X2(:)];

meanfunc = @meanZero;                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -30, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);


mean = reshape(mu, length(x2), length(x1));
stdev = reshape(s2,length(x2), length(x1));
