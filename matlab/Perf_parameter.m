function [T_avg,T] = Perf_parameter(MN, adv_angle)

invoke(MN,'processCommand','Call getDocument().setParameter("","CurrentRated","%CurrentRatedMax",infoNumberParameter)');
invoke(MN,'processCommand',['Call getDocument().setParameter("","xCurrentAdvanceAngle","',num2str(adv_angle),'",infoNumberParameter)']);
invoke(MN,'processCommand','error = getDocument().solveTransient2dWithMotion()');
invoke(MN,'processcommand','Call setVariant(0, error, "MATLAB")');
invoke(MN,'processCommand','CALL getDocument().getSolution().getGlobalSolutionTimeInstants(1,t)');
invoke(MN,'processcommand','Call setVariant(1,t,"MATLAB")');
torque = cell2mat(invoke(MN,'getVariant',1,'MATLAB'));

T = zeros(length(torque),1);
for i = 1:length(torque)
    invoke(MN, 'processCommand', ['Call getDocument().getSolution().getTorqueOnBody(Array(1,',num2str(torque(i)),'),2,Array(0,0,0),,,Tz)']);
    invoke(MN,'processcommand','Call setVariant(0,Tz,"MATLAB")');
    T(i) = invoke(MN,'getVariant',0,'MATLAB');
end

T = T*4; 
% *#poles
T_avg = mean(T);                     


end