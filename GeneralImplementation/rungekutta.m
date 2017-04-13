function next_state = rungekutta(state, action, params)


% if state(1)>pi
%     state(1)=state(1)-2*pi;
% end
% 
% if state(1)<-pi
%     state(1)=state(1)+2*pi;
% end

% state(:,1) = wrapToPi(state(:,1));

noise = params.noise;%params.std*randn();

k1 = continous_dynamics(state, action, noise, params);
k2 = continous_dynamics(state+params.h/2*k1, action, noise, params);
k3 = continous_dynamics(state+params.h/2*k2, action, noise, params);
k4 = continous_dynamics(state+params.h*k3, action, noise, params);


next_state = state + (params.h/6)*(k1+2*k2+2*k3+k4);

% if next_state(1)>pi
%     next_state(1)=next_state(1)-2*pi;
% end
% 
% if next_state(1)<-pi
%     next_state(1)=next_state(1)+2*pi;
% end
% next_state(:,1) = wrapToPi(next_state(:,1));