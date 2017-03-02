% Value Iteration
% By Sadegh Talebi
% November 6, 2016

function [policy, n, v0_vec] = valueiteration(mdp,params) 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Problem data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = size(mdp.S,1);
A = size(mdp.A,1);

r = mdp.R;

P = mdp.P;



% Discount factor
lambda = params.lambda;%.99;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epsilon = params.epsilon;% 0.0001;
n = 0;
v0 = zeros(S, 1);
v1 = ones(S, 1);
policy = zeros(S, 1);

v0_vec = v0; 
policy_vec = policy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                Main Loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while max(abs(v1-v0)) >= epsilon*(1-lambda)/(2*lambda)
    n = n+1;
    v1 = v0;
    for s=1:S
        tmp_s = r(s, :)' + lambda*P{s}'*v1;
        [v0(s), policy(s)] = max(tmp_s);
    end
    
    v0_vec = [v0_vec v0];
    policy_vec = [policy_vec policy];
    if mod(n,50) == 0
        fprintf('Iter: %d\n',n);
    end
    if n>50000
        break;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display(n);
% display(v0');
% display(policy');



% randomly choosing some indicies to plot
% 
% Line_width=3;
% Ft_size_1=20;
% Ft_size_2=18;
% 
% figure(1);
% fullscreen = get(0,'ScreenSize');
% figure('PaperType','A1','PaperSize',[59.399975052 84.099964678]);
% plot(v0_vec(1, :), '-b','LineWidth', Line_width); 
% hold on; 
% plot(v0_vec(2, :), '-r','LineWidth', Line_width);
% hold on; plot(v0_vec(3, :), '-g','LineWidth', Line_width);
% grid on;
% xlim([0 1000]);
% xlabel('Number of iterations','FontSize', Ft_size_1,'FontName','Arial');
% ylabel('Values','FontSize', Ft_size_1,'FontName','Arial');
% legend('Location', 'northwest')
% legend('Value Iteration (\epsilon = 0.01)'); 
% set(gca,'FontSize', Ft_size_2, 'FontName', 'Arial')
% 
% Line_width=3;
% Ft_size_1=20;
% Ft_size_2=18;
% figure(2);
% fullscreen = get(0,'ScreenSize');
% figure('PaperType','A1','PaperSize',[59.399975052 84.099964678]);
% semilogx(policy_vec(1, :), '-b','LineWidth', Line_width); 
% hold on; 
% plot(policy_vec(2, :), '-r','LineWidth', Line_width); 
% hold on; 
% plot(policy_vec(3, :), '-g','LineWidth', Line_width);
% grid on;
% xlim([0 1000]);
% xlabel('Number of iterations','FontSize', Ft_size_1,'FontName','Arial');
% ylabel('Actions','FontSize', Ft_size_1,'FontName','Arial');
% legend('Location', 'northwest')
% set(gca,'FontSize', Ft_size_2, 'FontName', 'Arial')
% 
% 
% 




