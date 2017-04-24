function states = sim_pend(policy, mdp, params)


x = [-pi+0.1, 0];


states = [];
for i = 1:100
    
    state_index = discr(x, mdp.S);
    action = mdp.A(policy(state_index));  
    disp(x)
    x = rungekutta(x, action, params);
    states = [states; x];
    
    theta = x(1);
    l=3;

    pxp=[0 l*sin(theta)];
    pyp=[1.25 1.25+l*cos(theta)];

    arrowfactor_x=sign(action)*2.5;
    if (sign(arrowfactor_x)>0)
        text_arrow = strcat('==>> ',int2str(action));
    else if (sign(arrowfactor_x)<0)
            text_arrow = strcat(int2str(action),' <<==');
        else
            text_arrow='=0=';
            arrowfactor_x=0.25;
        end
    end


    figure(4)


    hold on

    plot(pxp,pyp,'-k','LineWidth',5);
    plot(pxp,pyp,'-k','LineWidth',5);
    plot(pxp(1),pyp(1),'.g','LineWidth',2,'Markersize',10,'MarkerEdgeColor','k');
    plot(pxp(2),pyp(2),'gO','LineWidth',2,'Markersize',15,'MarkerEdgeColor','k','MarkerFaceColor','r');

    text(arrowfactor_x - 0.5 ,0.8,text_arrow);
    axis([-6 6 0 6])


    pause(0.5)
    drawnow;
    cla
    hold off




end

end