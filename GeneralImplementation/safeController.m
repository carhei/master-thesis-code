function [data, u_opt, g] = safeController(mdp, pendulum, Dmin, Dmax, init)

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing
%   agreement contained in the file LICENSE in the top directory of
%   the distribution.
%
% Ian Mitchell, 3/26/04
% Subversion tags for version control purposes.
% $Date: 2012-07-04 14:27:00 -0700 (Wed, 04 Jul 2012) $
% $Id: air3D.m 74 2012-07-04 21:27:00Z mitchell $


maxU = max(mdp.A);
D_MAX = max(abs(Dmin),abs(Dmax));

%---------------------------------------------------------------------------
% Integration parameters.
tMax = 10;                  % End time.
plotSteps = 6;               % How many intermediate plots to produce?
t0 = 0;                      % Start time.
singleStep = 0;              % Plot at each timestep (overrides tPlot).

% Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (plotSteps - 1);

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% What kind of dissipation?
dissType = 'global';

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;
% Visualize the 3D reachable set.
displayType = 'contour';
% Pause after each plot?
pauseAfterPlot = 0;
% Delete previous plot before showing next?
deleteLastPlot = 1;
% Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 0;

%---------------------------------------------------------------------------
% Approximately how many grid cells?
%   (Slightly different grid cell counts will be chosen for each dimension.)
Nx = pendulum.grid.state_steps(1);
Ny = pendulum.grid.state_steps(2);

% Create the grid.
g.dim = 2;
g.min = mdp.S(1,:)';
g.max = mdp.S(end,:)';
g.bdry = { @addGhostExtrapolate; @addGhostPeriodic};
% Roughly equal dx in x and y (so different N).
g.N = [ Nx; Ny ];
g.max(2) = g.max(2) * (1 - 1 / g.N(2));
% Need to trim max bound in \psi (since the BC are periodic in this dimension).
g = processGrid(g);


%---------------------------------------------------------------------------
% Create initial conditions (cylinder centered on origin).
data = shapeComplement(shapeRectangleByCenter(g, [0;0], [init;2*mdp.S(end,2)]));
% data = shapeComplement(shapeSphere(g, [ 0; 0 ], initRadius));
% data = shapeDifference(dataX,dataY);
% data = shapeSphere(g, [ 0; 0 ], targetRadius);

data0 = data;

%---------------------------------------------------------------------------
% Set up spatial approximation scheme.
schemeFunc = @termLaxFriedrichs;
schemeData.hamFunc = @pendulumHamFunc;%air3DHamFunc;
schemeData.partialFunc = @pendulumPartialFunc;%air3DPartialFunc;
schemeData.grid = g;

% The Hamiltonian and partial functions need problem parameters.
schemeData.U_MAX = maxU;
schemeData.D_MAX = D_MAX;

schemeData.D_min = Dmin;
schemeData.D_max = Dmax;


schemeData.g = pendulum.params.g;
schemeData.m = pendulum.params.m;
schemeData.l = pendulum.params.l;
%---------------------------------------------------------------------------
% Choose degree of dissipation.

switch(dissType)
    case 'global'
        schemeData.dissFunc = @artificialDissipationGLF;
    case 'local'
        schemeData.dissFunc = @artificialDissipationLLF;
    case 'locallocal'
        schemeData.dissFunc = @artificialDissipationLLLF;
    otherwise
        error('Unknown dissipation function %s', dissFunc);
end

%---------------------------------------------------------------------------

accuracy = 'medium';


% Set up time approximation scheme.
integratorOptions = odeCFLset('factorCFL', 0.75, 'stats', 'on');

% Choose approximations at appropriate level of accuracy.
switch(accuracy)
    case 'low'
        schemeData.derivFunc = @upwindFirstFirst;
        integratorFunc = @odeCFL1;
    case 'medium'
        schemeData.derivFunc = @upwindFirstENO2;
        integratorFunc = @odeCFL2;
    case 'high'
        schemeData.derivFunc = @upwindFirstENO3;
        integratorFunc = @odeCFL3;
    case 'veryHigh'
        schemeData.derivFunc = @upwindFirstWENO5;
        integratorFunc = @odeCFL3;
    otherwise
        error('Unknown accuracy level %s', accuracy);
end

if(singleStep)
    integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on');
end

%---------------------------------------------------------------------------
% Restrict the Hamiltonian so that reachable set only grows.
%   The Lax-Friedrichs approximation scheme MUST already be completely set up.
innerFunc = schemeFunc;
innerData = schemeData;
clear schemeFunc schemeData;

% Wrap the true Hamiltonian inside the term approximation restriction routine.
schemeFunc = @termRestrictUpdate;
schemeData.innerFunc = innerFunc;
schemeData.innerData = innerData;
schemeData.positive = 0;

%---------------------------------------------------------------------------
% Initialize Display


% set(0,'CurrentFigure',SafeFigure)
% Set up subplot parameters if necessary.
if(useSubplots)
    rows = ceil(sqrt(plotSteps));
    cols = ceil(plotSteps / rows);
    plotNum = 1;
    subplot(rows, cols, plotNum);
end

% h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(t0) ]);

hold on;
axis(g.axis);
xlim([-2,2]);
drawnow;
%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff).
global u_opt
global d_opt

tNow = t0;
startTime = cputime;
while(tMax - tNow > small * tMax)
    
    % Reshape data array into column vector for ode solver call.
    y0 = data(:);
    
    % How far to step?
    tSpan = [ tNow, min(tMax, tNow + tPlot) ];
    
    % Take a timestep.
    [ t,y, schemeData] = feval(integratorFunc, schemeFunc, tSpan, y0,...
        integratorOptions, schemeData);
    tNow = t(end);
    
    % Get back the correctly shaped data array
    data = reshape(y, g.shape);
    
    if(pauseAfterPlot)
        % Wait for last plot to be digested.
        pause;
    end
    
%     % Get correct figure, and remember its current view.
%     figure(SafeFigure);
%     [ view_az, view_el ] = view;
%     
    % Delete last visualization if necessary.
%     if(deleteLastPlot)
%         delete(h);
%     end
    
    % Move to next subplot if necessary.
    if(useSubplots)
        plotNum = plotNum + 1;
        subplot(rows, cols, plotNum);
    end
    
    % Create new visualization.
%     h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tNow) ]);
%     
%     % Restore view.
%     view(view_az, view_el);
%     
end

endTime = cputime;
fprintf('Total execution time %g seconds\n', endTime - startTime);


% simulation(g, u_opt, d_opt, data)


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function [hamValue, schemeData] = pendulumHamFunc(t, data, deriv, schemeData)
% air3DHamFunc: analytic Hamiltonian for 3D collision avoidance example.

%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   deriv	 Cell vector of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%
%   hamValue	 The analytic hamiltonian.
%
% schemeData is a structure containing data specific to this Hamiltonian
%   For this function it contains the field(s):
%
% Ian Mitchell 3/26/04

% checkStructureFields(schemeData, 'grid', 'velocityA', 'velocityB', ...
%     'inputA', 'inputB');

grid = schemeData.grid;


minD = (deriv{2}<=0).*schemeData.D_max+(deriv{2}>0).*schemeData.D_min;


hamValue = -(grid.xs{2}.* deriv{1} ...
    + schemeData.g/schemeData.l*sin(grid.xs{1}).* deriv{2} ...
    + 1/(schemeData.m*schemeData.l)*schemeData.U_MAX * abs(deriv{2}) ...
    + minD.*deriv{2});

global u_opt
global d_opt

u_opt = schemeData.U_MAX*sign(deriv{2});
d_opt = minD;

u_opt(u_opt==0)=-schemeData.U_MAX;
%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function alpha = pendulumPartialFunc(t, data, derivMin, derivMax, schemeData, dim)
% air3DPartialFunc: Hamiltonian partial fcn for 3D collision avoidance example.

% It calculates the extrema of the absolute value of the partials of the
%   analytic Hamiltonian with respect to the costate (gradient).
%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   derivMin	 Cell vector of minimum values of the costate (\grad \phi).
%   derivMax	 Cell vector of maximum values of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%   dim          Dimension in which the partial derivatives is taken.
%
%   alpha	 Maximum absolute value of the partial of the Hamiltonian
%		   with respect to the costate in dimension dim for the
%                  specified range of costate values (O&F equation 5.12).
%		   Note that alpha can (and should) be evaluated separately
%		   at each node of the grid.
%
% schemeData is a structure containing data specific to this Hamiltonian
%
% Ian Mitchell 3/26/04schemeData

grid = schemeData.grid;

switch dim
    case 1
        alpha = abs(grid.xs{2});
        
    case 2
        alpha = schemeData.g/schemeData.l*abs(sin(grid.xs{1}))+ 1/(schemeData.m*schemeData.l)* schemeData.U_MAX + schemeData.D_MAX;
        
        
    otherwise
        error([ 'Partials for the game of two identical vehicles' ...
            ' only exist in dimensions 1-3' ]);
end



function simulation(g, u_opt_array, d_opt_array, data)


duration = 2;
nSim = 50;
nTime = duration/0.0032;
t = linspace(0,duration,nTime);

x10 = g.min(1) + diff([g.min(1); g.max(1)])*rand(1,nSim);
x20 = g.min(2) + diff([g.min(2); g.max(2)])*rand(1,nSim);


[X1,X2] = meshgrid(g.vs{1,1}, g.vs{2,1});
states = [X1(:) X2(:)];
u_opt_array = u_opt_array(:);
d_opt_array = d_opt_array(:);

params.g = 9.81;
params.m = 1;
params.l = 1;

figure(); clf; hold on;

for i = 1:nSim
    x0 = [x10(i); x20(i)];
    [t, y] = ode45(@(t,y) sys_safeControl(t,y,states,params, u_opt_array,d_opt_array), t, x0);
    plot(y(:,1),y(:,2),'k-')
    plot(y(1,1),y(1,2),'r.', 'MarkerSize',15);
    plot(y(end,1),y(end,2),'b.', 'MarkerSize',15);


end

    contour(g.xs{1}, g.xs{2}, data, [0,0], 'g','LineWidth', 2);

