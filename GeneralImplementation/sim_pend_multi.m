
test.nSim = 16;
test.duration = 10;
test.nTime = ceil(test.duration/pendulum.params.h);
test.t = linspace(0,test.duration,test.nTime);
test.x10 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand(1,test.nSim);
test.x20 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand(1,test.nSim);

figure(3); clf; hold on;

for i = 1:test.nSim
    x0 = [test.x10(i); test.x20(i)];
    [t, y] = ode45(@(t,y) sys(t,y,mdp,pendulum.params, policy), test.t, x0);
    plot(y(:,1),y(:,2),'k-')
    plot(y(1,1),y(1,2),'r.', 'MarkerSize',15);
    plot(y(end,1),y(end,2),'b.', 'MarkerSize',15);

end


% test.nSim = 1;
% test.duration = 4;
% test.nTime = ceil(test.duration/pendulum.params.h);
% test.t = linspace(0,test.duration,test.nTime);
% test.x10 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand(1,test.nSim);
% % test.x20 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand(1,test.nSim);

% z0 = [test.x10; test.x20];
% 
% % z0 = [-0.8151; 4.501];
% 
% 
% [nState,nSim] = size(z0);
% nTime = length(test.t);
% x = zeros(nState,nSim,nTime);
% x(:,:,1) = z0;
% for k = 1:nSim
%     for i=1:(nTime-1)
%         state_index = discr(x(:,k,i)', mdp.S);
%         action = mdp.A(policy(state_index));
%         x(:,k,i+1) = rungekutta(x(:,k,i)', action, pendulum.params);
%     end
% end
% 
% 
% figure(5); clf; hold on;
% for i=1:test.nSim
%     thTest = reshape(x(1,i,:),test.nTime,1);
%     wTest = reshape(x(2,i,:),test.nTime,1);
%     plot(thTest, wTest, 'k-');
%     plot(thTest(1), wTest(1),'r.', 'MarkerSize',15);
%     plot(thTest(end), wTest(end), 'b.', 'MarkerSize',15);
% end
% 
% 
% % 