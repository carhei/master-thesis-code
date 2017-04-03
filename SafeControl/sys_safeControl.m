function  [dx] = sys_safeControl(t, x, states, params, u_opt_array, d_opt_array)
    
    m = params.m;
    l = params.l;
    g = params.g;
%     b = params.b;
  
    state_index = discretize(x', states);
    u_opt = u_opt_array(state_index);
    d_opt = d_opt_array(state_index);
    
    dx1 = x(2);
    dx2 = 1/(m*l)*u_opt+g/l*sin(x(1))+d_opt;%-b/m*x2
       
    
    dx = [dx1, dx2]';

end