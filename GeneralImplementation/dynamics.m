function next_state = dynamics(state, action, sampletime, noise, params)
     m = params.m;
     l = params.l;
     g = params.g;
     b = params.b;
    
    
    x1 = state(1);
    u = action;
    x2 = state(2);
    h = sampletime;
    
%     x1_next = x2;
% 
%     x2_next = 1/(m*l)*u-g/l*sin(x1)+noise;%-b/m*x2
    
    x1_next = x1 + h*x2;
    x2_next = x2 + h*(1/(m*l)*u+g/l*sin(x1)+noise);%-b/m*x2
    
    next_state = [x1_next, x2_next];
     
    
end
