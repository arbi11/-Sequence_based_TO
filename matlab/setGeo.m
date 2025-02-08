function [T_avg] = setGeo(data)

folder_path = 'C:\Users\chetanm\Desktop\Deep Learning\IGTE_worm_GA';

y = 38;

n_grid_rows, n_grid_columns = size(data)
n_grid_columns = 6;
% symetricity :gets mirrored and multiplied by 2

n_grid_rows = 6;              

Visible = 0;
model_name = '1_Reset_Geo_PrintedSilicon';

grid_old = data;
% grid_old3 = rot90(grid_old, 3); % Making it suitable for Magnet
    
% grid_old = zeros(n_grid_rows,n_grid_columns);
% grid_old(1,1) = 1;

% grid_new = data;

MN = actxserver('MagNet.Application');
set(MN,'Visible',Visible);
model_path = [folder_path,'\',model_name,'.mn'];

consts = invoke(MN,'getConstants');
document = invoke(MN,'openDocument',model_path);
view = invoke(document,'getView',int32(1));
invoke(view,'setScaledToFit',get(consts,'infoTrue'));

gridCompareNSet(MN, n_grid_columns, grid_old);
[T_avg,T] = Perf_parameter(MN, y);



model_name = '2_Set_Geo_PrintedSilicon';
model_path = [folder_path,'\',model_name,'.mn'];

command = ['Call getDocument().save("', model_path,'")'];
invoke(MN,'processCommand',command);

invoke(MN,'processCommand','Call close(False)');

% model_name = '2_Set_Geo';
% model_path = [folder_path,'\',model_name,'.mn'];

% command = ['Call getDocument().save("', model_path,'")'];
% invoke(MN,'processCommand',command);

% invoke(MN,'processCommand','Call close(False)');
