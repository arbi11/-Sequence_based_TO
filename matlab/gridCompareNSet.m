function [] = gridCompareNSet(MN, n_grid_columns, grid_old)

[r,c] = find(grid_old~=1);
% iron = 'M-19 26 Ga';
iron = 'Printed_Silicon';
air = 'Virtual Air';

meshElementSizeIron = '%MeshElementSizeRotorCore';

n_new_ele = size(r, 1);

%% Setting the back iron
% for i=1:n_grid_columns
%     comp_name = ['Component#', int2str(0), '_', int2str(i-1)];
%     command = ['Call getDocument().setParameter("',comp_name,'","Material", "',iron,'",infoStringParameter)'];
%     invoke(MN,'processCommand',command);
% 	command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
%     invoke(MN,'processCommand',command);
%     
%     comp_name = ['Component#', int2str(0), '_', int2str(2*n_grid_columns - i)];
%     command = ['Call getDocument().setParameter("',comp_name,'","Material", "',iron,'",infoStringParameter)'];
%     invoke(MN,'processCommand',command);
% 	command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
%     invoke(MN,'processCommand',command);     
% end

%% Setting the blank (iron) geometry

for j=1:(n_grid_columns+1)
    for i=1:n_grid_columns
        comp_name = ['Component#', int2str(j-1), '_', int2str(i-1)];
        command = ['Call getDocument().setParameter("',comp_name,'","Material", "',iron,'",infoStringParameter)'];
        invoke(MN,'processCommand',command);
        command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
        invoke(MN,'processCommand',command);
            comp_name = ['Component#', int2str(j-1), '_', int2str(2*n_grid_columns - i)];
        command = ['Call getDocument().setParameter("',comp_name,'","Material", "',iron,'",infoStringParameter)'];
        invoke(MN,'processCommand',command);
        command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
        invoke(MN,'processCommand',command);     
    end
end
    
%% Setting the new material
for i = 1:n_new_ele
    comp_name = ['Component#', int2str(r(i)), '_', int2str(c(i)-1)];
    command = ['Call getDocument().setParameter("',comp_name,'","Material", "',air,'",infoStringParameter)'];
    invoke(MN,'processCommand',command);    
	command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
    invoke(MN,'processCommand',command);    
    
    comp_name = ['Component#', int2str(r(i)), '_', int2str(2*n_grid_columns - c(i))];
    command = ['Call getDocument().setParameter("',comp_name,'","Material", "',air,'",infoStringParameter)'];
    invoke(MN,'processCommand',command);    
	command = ['Call getDocument().setParameter("',comp_name,'","MaximumElementSize", "',meshElementSizeIron,'",infoNumberParameter)'];
    invoke(MN,'processCommand',command);    
    
end
