function index_discrete_state = discretize(state, S)
%DISCRETIZE Summary of this function goes here
%   Detailed explanation goes here
   
    state = repmat(state,size(S,1),1);
    
    [d,index_discrete_state] = min(sum(abs(state-S),2));


end
