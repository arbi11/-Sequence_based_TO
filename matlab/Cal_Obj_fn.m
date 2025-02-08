function [Obj_fn_val] = Cal_Obj_fn(states,count)

iron_cells = count;
% if the number of iron cells is greater than a normal C core (180) or less than
% a minimum value, set it to inf thereby applying a volume constraint
% for 5X5 SyRM: Min: 10, Max: 30
% for 13X13 SyRM: Min: 50, Max: 150/120

y1 = max(iron_cells-18,0);
y2 = abs(min(iron_cells-10,0));

if y1==0 && y2==0
      
    Obj_fn_val = -1*setGeo(states);
    
else
    Obj_fn_val = y1+y2;
    
end
end
