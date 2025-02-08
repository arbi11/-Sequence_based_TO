function [reward] = steps_worm(actions)
% actions is 1x150 ROW vector

global eps
eps = eps+1;

% mkdir ./Data num2str(eps)
a = ['C:\Users\chetanm\Desktop\Deep Learning\IGTE_worm_GA\Data_worm_tunnel_PS2'];
% mkdir(a);

state_dim_r = 6; % Grid is [6, 6]; rows go to rotor - airgap intersection
state_dim_c = 6; % Mirrored to 12

max_steps = 18;
start_Row = 2;
end_Row = state_dim_c+1;
start_col = 2;
end_col = state_dim_c+1;
step = int16(1);

state = ones(state_dim_r+2 , state_dim_c+3)*3; % state is [8, 8]
state(start_Row:end_Row, start_Row:end_Row) = 1;
state(start_Row:end_Row, 2) = 1;
state(end_Row, start_col:end_col) = 1;
state(start_Row:end_Row, end-1) = 2;

posR = int16(2);
posC = int16(5);
[state, issue] = worm_move(state, posR, posC, step);

reward = 0.0;

%%% Action description %%%
%       00 = 0 = UP      %
%       01 = 1 = DOWN    %
%       10 = 2 = RIGHT   %
%       11 = 3 = LEFT    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

act_2d = reshape(actions, [2, max_steps])';
action_dec = bi2de(act_2d);

for i = 1:max_steps
    
    issue = false;
    
    if (action_dec(i) == 0)
        
        posR = posR - step;
        [state, issue] = worm_move(state, posR, posC, step);
        if issue == true
            posR = posR + step;
        end
    
    elseif (action_dec(i) == 1)
        
        posR = posR + step;
        [state, issue] = worm_move(state, posR, posC, step);
        if issue == true
            posR = posR - step;
        end
                
    elseif (action_dec(i) == 2)
        
        posC = posC + step;
        [state, issue] = worm_move(state, posR, posC, step);
        if issue == true
            posC = posC - step;
        end
    elseif (action_dec(i) == 3)
        
        posC = posC - step;
        [state, issue] = worm_move(state, posR, posC, step);
        if issue == true
            posC = posC + step;
        end            
    end
    
    state(start_Row:end_Row, 2) = 1;
    state(start_Row:end_Row, end-1) = 2;
    state(posR, posC) = 5;    
    
%     imwrite(state(start_R:end_R, start_R:end_R)/5, [a, '\', num2str(i), '-action-', num2str(action_dec(i)),'.png'])
%     if count > 180
%         break
%     end
    
end

state(state == 5) = 0;
motor = rot90(state(start_Row:end_Row, start_Row:end_Row), 3);
count = sum(motor(:) == 0);

reward = Cal_Obj_fn(motor,count); % state (5x5) and count(iron cells)
imwrite(state(start_Row:end_Row, start_Row:end_Row), [a, '\', num2str(reward), '=reward_eps=', num2str(eps),'.png'])
