function  [dx, action] = sys(t, x, mdp, params, policy)
    
    m = params.m;
    l = params.l;
    g = params.g;
    b = params.b;
%     noise = params.std*randn();    
noise = params.noise;
  
    state_index = discr(x', mdp.S);
    action = mdp.A(policy(state_index));
    
    
    dx1 = x(2);
    dx2 = 1/(m*l)*action+g/l*sin(x(1))+noise;%-b/m*x2
    
    
    
    dx = [dx1, dx2]';

end