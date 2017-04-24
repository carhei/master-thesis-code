function P = transitionprobs(S,A, params)


P = cell(size(S,1), 1);
for s=1:size(S,1)
    P{s} = zeros(size(S,1), size(A,1));
end


transitions = cell(size(S,1), 1);

for s = 1:size(S,1)
    for i = 1:50
        for a = 1:size(A,1)
            state = S(s,:);
            action = A(a,:);
            next_state = rungekutta(state, action, params);
            
            index_next_state = discr(next_state, S);           
            transitions{s}(i,a) = index_next_state;
        end
    end
end


for s = 1:size(S,1)
    for a = 1:size(A,1)
        y = transitions{s}(:,a);
        g = grp2idx(y);
        count = accumarray(g,1);
        p = count(g) ./ numel(g);
        P{s}(y, a) = p;
    end
end

