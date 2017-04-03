clear
close all
clc

pendulum.nstates = 2;
pendulum.params.g = 9.81;       % gravity constant
pendulum.params.m = 1;          % mass of pendulum
pendulum.params.l = 1;          % length of pendulum
pendulum.params.b = 0.2;        % damping coefficient
% pendulum.params.period = 2*pi*sqrt(pendulum.params.l/pendulum.params.g);
pendulum.params.h = 0.2;%01*pendulum.params.period;
pendulum.params.noise = 0.0;      % standard deviation input disturbance



% MDP grid params
x1_bounds = [-pi/2, pi/2];
x1_steps = 15;
x2_bounds = [-5,5];
x2_steps = 20;
pendulum.grid.state_bounds = [x1_bounds; x2_bounds];
pendulum.grid.state_steps = [x1_steps; x2_steps];

pendulum.grid.input_bounds = [-20; 20];
pendulum.grid.input_steps =  25;

pendulum.grid.varyspacing = 0;


pendulum.purpose = 'forValueIteration';




pendulum.reward.type = 'exponential';  %exponential reward

mdp = BuildMDP(pendulum);
% Solve the MDP model

opt_params.gamma = 0.9;
opt_params.epsilon = 0.0001;

[policy, n, v0_vec] = valueiteration(mdp, opt_params);
% [policy_vec, v_vec, n] = policyiteration(mdp,opt_params);
% policy = policy_vec(:,end);

marker = 90*ones(size(policy));

figure(1); clf; colormap('jet')
scatter3(mdp.S(:,1), mdp.S(:,2), policy, marker, policy, 'filled');
xlabel('angle (rad)')
ylabel('rate (rad/s)')
zlabel('torque (Nm)')
title('Optimal Policy')
view(2)
axis equal

figure(2); clf; colormap('jet')
scatter3(mdp.S(:,1), mdp.S(:,2),v0_vec(:,end), marker, v0_vec(:,end), 'filled');
xlabel('angle (rad)')
ylabel('rate (rad/s)')
zlabel('value ')
title('Value Function')
view(2)
axis equal
