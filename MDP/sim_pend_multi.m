% function x = sim(z0, t)

test.nSim = 100;
test.duration = 1.5;
test.nTime = ceil(test.duration/pendulum.params.h);
test.t = linspace(0,test.duration,test.nTime);
test.x10 = pendulum.grid.state_bounds(1,1) + diff(pendulum.grid.state_bounds(1,:))*rand(1,test.nSim);
test.x20 = pendulum.grid.state_bounds(2,1) + diff(pendulum.grid.state_bounds(2,:))*rand(1,test.nSim);

z0 = [test.x10; test.x20];

[nState,nSim] = size(z0);
nTime = length(test.t);
x = zeros(nState,nSim,nTime);
x(:,:,1) = z0;
for k = 1:nSim
    for i=1:(nTime-1)
        state_index = discretize(x(:,k,i)', mdp.S);
        action = mdp.A(policy(state_index));
        x(:,k,i+1) = rungekutta(x(:,k,i)', action, pendulum.params);
    end  
end


figure(3); clf; hold on;
for i=1:test.nSim
    thTest = reshape(x(1,i,:),test.nTime,1);
    wTest = reshape(x(2,i,:),test.nTime,1);
    %u = pendulumController([thTest,wTest]',A,P,pendulum.grid);
    plot(thTest, wTest, 'k-');
     plot(thTest(1), wTest(1),'r.', 'MarkerSize',15);
%     plot3(thTest(end), wTest(end), u(end), 'b.', 'MarkerSize',15);
end

% end