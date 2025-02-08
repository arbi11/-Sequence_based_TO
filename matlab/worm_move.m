function [new_state, issue] = worm_move(state, posR, posC, step)
issue = false;

% posR
% posC
% state(posR, posC)
% posR > 16 || posR < 1 || posC < 1 || posC > 22 || 
state(state == 5) = 0;

if state(posR, posC) == 3
    issue = true;
elseif state(posR, posC) == 2
    state(posR, posC) = 5;
else
        
%     change = floor(step/ 2);
%     state(posR-change: posR+change, posC-change: posC+change) = 1;
    state(posR, posC) = 5;
end

% new_state = cleanup(state); 
% No need for cleanup here since we are
% slicing the state before feeding back

new_state = state;