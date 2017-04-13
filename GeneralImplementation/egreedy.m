function a = egreedy(s,epsilon,Q)

actions = size(Q,2);
	
if (rand()>epsilon) 
    [v, a] = max(Q(s,:));
else
    a = randi(actions);
end