function next_state = rungekutta(state, action, params)


    
    noise = params.std*randn();
            
    k1 = continous_dynamics(state, action, noise, params);
    k2 = continous_dynamics(state+params.h/2*k1, action, noise, params);
    k3 = continous_dynamics(state+params.h/2*k2, action, noise, params);
    k4 = continous_dynamics(state+params.h*k3, action, noise, params);
    
    
next_state = state + (params.h/6)*(k1+2*k2+2*k3+k4);