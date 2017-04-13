function  dx = sys_gaussianProcesses(t, x, params, actions)
% x = reshape(x,2000,2)
m = params.m;
l = params.l;
g = params.g;

noise = params.noise;


dx1 = x(:,2);
dx2 = 1/(m*l)*actions+g/l*sin(x(:,1))+noise;%-b/m*x2


disp(size(dx1))
dx = [dx1, dx2]';

end