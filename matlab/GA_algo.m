clc
clear
clear global

% Ensure that the environement is set to all zeros before calling GA
% This is required, because, when the simulation crashes, the file (model)
% gets changed and is not all zero as it should be.
% So it is necessary to set everything to zero each time we begin.

% Set up to open Magnet

global eps
eps = 0;

global grid_no
grid_no = 6;

agent_eps_len = ceil(grid_no^2)/2;
% GA parameters

nPop = agent_eps_len*2*4;              % Population size
nGen = 100;              % Generation size
stallGen = 15;          % Number of stall generations
total_iter = nPop*nGen;      % Number of iterations

% Define the objective dunciton here
% The input is a vector, x, i.e. the deisgn or optimization variables
% The function should return a scalar value which has to be minimized
% fobj = @(x)steps_worm (x);
fobj = @(x)steps_worm(x);
% Call GA
% nVars = 36*2; % number of variables, i.e. 75x2 = 150
nVars = 18*2; % number of variables for conventional setting

lb = zeros(1,nVars);                % Lower bound
ub = ones(1,nVars);                 % Upper bound

optionsGA = gaoptimset('Display','iter','PopulationSize',nPop,'Generations',nGen,'StallGenLimit',stallGen,'PlotFcns',{@gaplotbestf,@gaplotbestindiv,@gaplotexpectation});
[best,fval,exitflag,output,population,scores]  = ga(fobj, nVars,[],[],[],[],lb,ub,[],1:length(lb),optionsGA);
save('GA_Algo_Run_worm_tunnel_PS'); % save the workspace


