import numpy as np
import femm
import imageio
import warnings
from  skimage import img_as_float, img_as_uint
#import matplotlib.pyplot as plt
import re
import os
import constants as con
from shutil import copyfile
import pickle

FEMM_PATH = con.PATH + '\\'
#FEMM_PATH = 'C:\\Users\\Arbi\\Desktop\\Arbi\\'
MODEL_PATH = con.MODEL_PATH + '\\'
TRAIN_PATH = con.TRAIN_PATH + '\\'

class WormFemmEnv2:

    def __init__(self, env_dim = [21, 35], startR=3 , startC= 4, max_steps= con.max_steps):
        
        self.env_dim = env_dim        
        self.step_count = 0
        self.posR = startR
        self.posC = startC
        self.done = 0.0
        self.repeat = 0
        self.issue = 0
        self.max_steps = max_steps
        self.past_rewards = []
        self.past_states = []
        self.past_actions = []
        self.state_hist = []
        self.past_BnStates = []
        self.past_force = []
        self.past_BArms = []
        self.net_force = 0.0
        self.R = 0.0
        self.count = 0
        self.buffer = 3
        self.B_state = np.zeros([15,])
        self.Bxy_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])        
        self.Mu_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])   

        
        self.coil_start = 2
        self.coil_width = 3
        self.coil_length = 9
        
        self.arm_start = 27
        self.arm_width = 6
        self.arm_length = 15
        
        self.cont_start_col = self.coil_start + self.coil_width
        self.window_width = self.arm_start - (self.coil_start + self.coil_width)
        self.window_length = np.max([self.arm_length, 15])
        self.frame_idx = 0
        self.worm_step_size = 3
        self.flip = 0
        self.flip_actions = {
                                0:1,
                                1:0,
                                2:2,
                                3:3,
                                4:5,
                                5:4,
                                6:6,
                                7:7
                                }
    ##### ACTIONS #######
    # 0     : RIGHT     #
    # 1     : LEFT      #
    # 2     : UP        #
    # 3     : DOWN      #
    ##-----------------##
    # 4     : RIGHT     #
    # 5     : LEFT      #
    # 6     : UP        #
    # 7     : DOWN      #
    #####################
            
    def create_geo(self):
        
        femm.openfemm(1);
        femm.opendocument(FEMM_PATH + 'actuator.fem');

        femm.mi_getmaterial('Air')
#        femm.mi_addmaterial('LinearIron', 2100, 2100, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        femm.mi_getmaterial('Cold rolled low carbon strip steel')            
        femm.mi_getmaterial('Copper')
        femm.mi_addcircprop('icoil', 1, 1);
        
        # femm.mi_setblockprop('Air', 0, 10, '<None>', 0, 0, 0);
        femm.mi_clearselected();

        for i in range(17):
            for j in range(35):
                
                femm.mi_selectlabel(i + 0.5, j + 0.5);
                
        femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 0, 0);
        femm.mi_clearselected();

        coil2_start = 8 + 8 - (self.coil_start + self.coil_width)
        femm.mi_clearselected();
        
        femm.mi_selectrectangle(0 + 0.1, self.coil_start + 0.1, self.coil_length - 0.1, self.coil_start + self.coil_width - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()
        femm.mi_drawrectangle(0, self.coil_start, self.coil_length, self.coil_start + self.coil_width)
        femm.mi_addblocklabel((0 + self.coil_length)/ 2, (2*self.coil_start + self.coil_width)/ 2)
        
        femm.mi_clearselected();

        femm.mi_selectrectangle(0 + 0.1, coil2_start + 0.1, self.coil_length - 0.1, coil2_start + self.coil_width - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()        
        femm.mi_drawrectangle(0, coil2_start, self.coil_length, coil2_start + self.coil_width)
        femm.mi_addblocklabel((0 + self.coil_length)/ 2, (2*coil2_start + self.coil_width)/ 2)
        femm.mi_clearselected();

        femm.mi_selectlabel((0 + self.coil_length)/ 2, (2*coil2_start + self.coil_width)/ 2);
        femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, 500); 
        femm.mi_clearselected();

        femm.mi_selectlabel((0 + self.coil_length)/ 2, (2*self.coil_start + self.coil_width)/ 2)
        femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, -500); 
        
        #        femm.mi_setblockprop('Copper', 0, 0.25, 'icoil', 0, 1, 500);        
        femm.mi_clearselected();

        femm.mi_selectrectangle(0 + 0.1, self.arm_start + 0.1, self.arm_length - 0.1, self.arm_start + self.arm_width - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()        
        femm.mi_drawrectangle(0, self.arm_start, self.arm_length, self.arm_start + self.arm_width)
        femm.mi_addblocklabel((0 + self.arm_length)/ 2, (2*self.arm_start + self.arm_width)/ 2)
        femm.mi_selectlabel((0 + self.arm_length)/ 2, (2*self.arm_start + self.arm_width)/ 2);
        femm.mi_setblockprop('Cold rolled low carbon strip steel', 0, 0.5, '<None>', 0, 5, 0);
        femm.mi_clearselected();
                
        femm.mi_drawrectangle(0, 100, 100, -100)
        femm.mi_addblocklabel(50, 0);
        femm.mi_selectlabel(50, 0);
        femm.mi_setblockprop('Air', 50, 0, '<None>', 0, 0, 0);
        femm.mi_clearselected();

        femm.mi_addboundprop('neumann', 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0)
        femm.mi_selectrectangle(-0.5, 100.5, 0.2, -100.5, 4)
        femm.mi_setsegmentprop('neumann', 0, 0, 0, 5)
        femm.mi_clearselected();       
        
        femm.mi_selectsegment(95, -95)
        femm.mi_selectsegment(55, 95)
        femm.mi_selectsegment(55, -95)
        femm.mi_addboundprop('dirichlet', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_setsegmentprop('dirichlet', 0, 0, 0, 0)
        femm.mi_clearselected();       

        femm.mi_zoomnatural();
        femm.mi_saveas(FEMM_PATH + 'actuator2.fem');  
        femm.closefemm()
        
        #####################################################################################
        #                                   VARIABLES                                       #       
        #####################################################################################
        
        # step_count           :       step# in an episode.
        # done          :       episode over or not.
        # snap          :       store material dist as png (mat2gray).
        # act_py        :       action (material dist) sent to matlab.
        # state_abs     :       action with discrete values(0/1) & flattened.
        # changes       :       changes made between (n-1) & 'n'th actions.
        # act_mat       :       action as matlab double.
        # count         :       No. of iron (1's) in the action.
        # frame_idx     :       episode #.
        
        #####################################################################################
        # mat_res       :       output from matlab:                                         #
        #                               nargout= 2 (means 2 variables from matlab)          #
        # field_xy      :                   var1 = Bx, By (concatenated) dim: (2, 21, 15)   #
        # Fx            :                   var2 = Force magnitude dim: 1(scalar)           #
        #####################################################################################
        
        # reward        :       Reward for the agent (Fx(n) - Fx(n-1)).
        # state3        :       environment state for the agent. dim: (3, 21, 15).
        
        #####################################################################################
        
    def move(self, mat_state):
        if (self.worm_step_size == 3):
            mat_state[self.posR -1 : self.posR + 2, self.posC -1 : self.posC + 2] = 1    
            mat_state[self.posR , self.posC] = 5
        elif (self.worm_step_size == 1 and self.past_actions[-1]<4):
            mat_state[self.posR -1 : self.posR + 2, self.posC -1 : self.posC + 2] = 1    
            mat_state[self.posR , self.posC] = 5
        else:
            mat_state[self.posR , self.posC] = 5

        return(mat_state)

    def reward(self, mat_state):
        
        #print('entered REWARD process with a shape of:', mat_state.shape)

        femm.openfemm(1);
        femm.opendocument(FEMM_PATH + 'actuator2.fem');
        #femm.mi_addmaterial('LinearIron', 2100, 2100, 0, 0, 0, 0, 0, 1, 0, 0, 0)

        indices = np.nonzero(mat_state[self.buffer:-1, :] - self.state_hist[-1])
        
        len_ind = np.size(indices[0])
        for i in range(len_ind):
            #print(indices[0][i], ',', indices[1][i])
            femm.mi_selectlabel(indices[0][i] + 0.4, indices[1][i] + 0.4);
            femm.mi_deleteselectedlabels()
            femm.mi_addblocklabel(indices[0][i] + 0.4, indices[1][i] + 0.4)
            femm.mi_clearselected();

        for i in range(len_ind):
            femm.mi_selectlabel(indices[0][i] + 0.4, indices[1][i] + 0.4);
            femm.mi_setblockprop('Cold rolled low carbon strip steel', 0, 0, '<None>', 0, 3, 0);
            
        femm.mi_clearselected();
        femm.mi_zoomnatural()
        femm.mi_saveas('actuator2.fem')
        femm.mi_analyse(0)
        femm.mi_loadsolution()
        
        B_sum = 0.0        
        for arm in range(6):
            
            Bx, By = femm.mo_getb(0.2, 27.5 + arm)
            B = np.sqrt(Bx ** 2 + By ** 2)
            B_sum += B
               
        femm.mo_groupselectblock(5)
        self.net_force = femm.mo_blockintegral(19) *(-10)
        self.past_force.append(self.net_force)
                                     
        Bag_state = np.zeros([15, 2])
        for row in range(15):
            Bag_state[row, :] = femm.mo_getb(row+0.5, 26.5)
            
        Bxy_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1], 2])
        
        for r in range(Bxy_state.shape[0]):
            for c in range(self.env_dim[1]):
                Bxy_state[r, c, :] = femm.mo_getb(r+0.5, c+0.5)
        
        self.Mu_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])
        
#        for r in range(self.Mu_state.shape[0]):
#            for c in range(self.env_dim[1]):
#                self.Mu_state[r, c] = femm.mo_getmu(r+0.5, c+0.5)
#        
        self.B_dist = np.round(np.sum(np.square(Bxy_state), axis=-1), decimals=2)
            
        self.B_state = np.sum(np.square(Bag_state), axis=1)
        
        femm.closefemm()
        
        reward = (B_sum/6)*1000
                
        return(reward)
        
#ob[self.buffer: self.buffer + self.coil_length, 
#           self.coil_start: self.coil_start + self.coil_width] = 2
#   
#        coil2_start = 8 + 8 - (self.coil_start + self.coil_width)
#   
#     ob[self.buffer: self.buffer + self.coil_length , 
#           coil2_start : coil2_start + self.coil_width] = 2
#
#        ob[self.buffer: self.buffer + self.arm_length, 
#           self.arm_start : self.arm_start + self.arm_width] = 4
#             
           
    def clean_up(self, mat_state):
        
        mat_state[self.buffer: self.buffer + self.coil_length, 
                  self.coil_start: self.coil_start + self.coil_width] = 2
        coil2_start = 8 + 8 - (self.coil_start + self.coil_width)
        mat_state[:, 0:self.coil_start] = 3
        mat_state[self.buffer: self.buffer + self.coil_length, 
                  coil2_start : coil2_start + self.coil_width] = 2
                  
        mat_state[self.buffer + self.coil_length:, 
                  self.coil_start: self.coil_start + self.coil_width] = 3                  
        mat_state[0:self.buffer, :] = 3
        mat_state[:, self.arm_start- 1] = 3
        mat_state[self.window_length + self.buffer:, :] = 3
        mat_state[self.buffer: self.buffer + self.arm_length, 
                  self.arm_start : self.arm_start + self.arm_width] = 4
        
        mat_state[mat_state == 5] = 1
        mat_state[self.posR, self.posC] = 5

        return mat_state
    
    ##### ACTIONS #######
    # 0     : RIGHT     #
    # 1     : LEFT      #
    # 2     : UP        #
    # 3     : DOWN      #
    ##-----------------##
    # 4     : RIGHT     #
    # 5     : LEFT      #
    # 6     : UP        #
    # 7     : DOWN      #
    #####################
    
    def step(self, action):
        
        state = np.copy(self.past_states[-1])
        self.step_count +=1
        self.R = 0.0
        self.count = 0
        self.net_force = 0.0
        self.B_dist = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])
        
        if self.flip == 1:
            action = self.flip_actions.get(action)
            
        if action == 0:
            self.worm_step_size = 1
            self.posC += self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.issue = 2
                self.posC -= self.worm_step_size
                        
        elif action == 1:
            self.worm_step_size = 1                               
            self.posC -= self.worm_step_size
            
            if state[self.posR, self.posC] == 0:
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.posC += self.worm_step_size
                self.issue = 2
        
        elif action == 2:
            self.worm_step_size = 1                               
            self.posR -= self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.posR += self.worm_step_size
                self.issue = 2
        
        elif action == 3:
            self.worm_step_size = 1    
            self.posR += self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                state2 = self.move(state)
                self.issue = 1
            else:
                self.posR -= self.worm_step_size
                self.issue = 2
        
        elif action == 4:
            if self.past_actions[-1]<4:
                self.worm_step_size = 1
            else:
                self.worm_step_size = 3  
                
            self.posC += self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.issue = 2
                self.posC -= self.worm_step_size
                        
        elif action == 5:
            if self.past_actions[-1]<4:
                self.worm_step_size = 1
            else:
                self.worm_step_size = 3                    
            
            self.posC -= self.worm_step_size
            
            if state[self.posR, self.posC] == 0:
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.posC += self.worm_step_size
                self.issue = 2
        
        elif action == 6:
            if self.past_actions[-1]<4:
                self.worm_step_size = 1
            else:
                self.worm_step_size = 3
                
            self.posR -= self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                self.issue = 1
                state2 = self.move(state)
            else:
                self.posR += self.worm_step_size
                self.issue = 2
        
        elif action == 7:
            if self.past_actions[-1]<4:
                self.worm_step_size = 1
            else:
                self.worm_step_size = 3
            self.posR += self.worm_step_size
            
            if state[self.posR, self.posC] == 0:    
                state2 = self.move(state)
                self.issue = 0
            elif state[self.posR, self.posC] == 1:
                state2 = self.move(state)
                self.issue = 1
            else:
                self.posR -= self.worm_step_size
                self.issue = 2
                
        if self.issue == 0:    
            #print('Going for reward with a shape of:', state2.shape)
            new_state = self.clean_up(state2)
            BArm = self.reward(new_state)
#            BMax_inc = np.mean(self.Bxy_state.ravel().sort()[-10:])
            
            self.R = BArm - self.past_BArms[-1]
            self.past_BArms.append(np.copy(BArm))
            self.past_rewards.append(np.copy(self.R))
           
        elif self.issue == 1:
                                 
            new_state = self.clean_up(state2)
            BArm = self.reward(new_state)
#            BMax_inc = np.mean(self.Bxy_state.ravel().sort()[-10:])
            
            self.R = BArm - self.past_BArms[-1]
            self.past_BArms.append(np.copy(BArm))
            self.past_rewards.append(np.copy(self.R))
           
        elif self.issue == 2:
            
            new_state = state
#            self.R = constants_DQL.penalty
#            F = self.past_rewards[-1]
#            self.past_rewards.append(F)
            
            #print('NOT Going for REWARD with a shape of:', new_state.shape)
            #print('NOT Going for REWARD coz of state[',self.posR,',',self.posC, '] = ', state[self.posR, self.posC])
            
            if state[self.posR, self.posC] == 2:
                self.R = con.penalty
                # coil
            elif state[self.posR, self.posC] == 3:
                self.R = con.penalty - 1
                # outside air
            else:
                self.R = con.penalty
                # Repeat
                
        unique, counts = np.unique(new_state, return_counts=True)
        index_1 = np.where (unique == 1)
        index_5 = np.where (unique == 5)
        try:
            self.count = np.asscalar(counts[index_1[0]] + 1)
            
        except:
            self.count = 0        
        
        if self.count > con.max_iron or self.step_count == self.max_steps:
            self.done = 1.0
            #print("Steps:", self.step_count, "Iron Count:", count, "Reward", self.past_rewards[-1])
#            R = self.reward(new_state)
#            print("Steps:", self.step_count, "Iron Count:", count, "Reward", R)    
#        elif count > 80:
#            R += 1
#        elif count > 80 and self.issue == 2:
#            R = 0
#        print("Steps:", self.step_count, "Iron Count:", count, "Reward", R, "Action- ", action)
        
        if new_state[self.posR, self.posC] != 5:
            raise('Problem in code. Worm outside the geometry')
            
        elif np.asscalar(counts[index_5[0]]) > 1:
            raise('Problem in code. More than one worm heads <<MEDUSA>>')
        
        self.past_states.append(np.copy(new_state))
        self.past_actions.append(np.copy(action))
#        , "Changes:", changes-22
        
        new_state2 = new_state[self.buffer: self.buffer+15, 5:27]
#        print('new_state2.shape', new_state2.shape)        
#        new_state3 = np.copy(new_state2)
        #new_state3 = np.concatenate((new_state2, np.copy(self.B_state)), axis=0)
#        new_state3[:, -1] = np.copy(self.B_state)
        
        if self.flip == 1:
            new_state3 = np.lib.pad(np.copy(new_state2), ((0, 21)), 'reflect')
            new_channel = np.lib.pad(self.B_dist[self.buffer: self.buffer+15, 5:27], ((0, 21)), 'reflect')
#            print('new_state3.shape', new_state3.shape)
#            print('new_channel.shape', new_channel.shape)                    

            new_state4 = np.concatenate((np.expand_dims(new_state3, 0), np.expand_dims(new_channel, 0)), axis=0)
#            print('new_state4.shape', new_state4.shape)
            
            new_state5 = new_state4[:, :15, -22:]                         
        else:
            new_state4 = np.expand_dims(np.copy(new_state2), 0)
#            print('new_state4.shape', new_state4.shape)
            new_channel = np.expand_dims(self.B_dist[self.buffer: self.buffer+15, 5:27], 0)
#            print('new_channel.shape', new_channel.shape)  
            new_state5 = np.concatenate((new_state4, new_channel), axis=0)
                                
        self.past_BnStates.append(np.copy(new_state5))
        
#        fd = open(eps_dir + str(frame_idx) + '.csv','a')
#        writer=csv.writer(fd,delimiter=',',quoting=csv.QUOTE_MINIMAL)
#        writer.writerow([self.step_count, R, count])
#        fd.close()
#        new_state = np.expand_dims(new_state, axis= -1)    
#        state_index = np.argmax(new_state[1:16, 5:26])
#        print('new_state5.shape', new_state5.shape)
        
        return(new_state5, self.R, self.done, {})
#            , self.issue
#############################################################################
#                             Reset Environment                             #       
#############################################################################
        
    def reset(self):
        
    #    state_dim = [21, 31]  # size of geometry

        self.worm_step_size = 3        
#        self.posC = 1+self.worm_step_size
#        self.posR = 1+self.worm_step_size
        self.done = 0.0
        
        self.max_steps = con.max_steps
        self.past_rewards = []
        self.past_states = []
        self.past_actions = []
        self.state_hist = []
        self.past_BnStates = []
        self.past_force = []
        self.past_BArms = []
        self.net_force = 0.0
        self.R = 0.0
        self.count = 0                
        self.step_count = 0
        self.B_state = np.zeros([15,])
        self.Bxy_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])       
        self.Mu_state = np.zeros([self.env_dim[0]-self.buffer, self.env_dim[1]])   

        self.flip = np.random.randint(2)
        # self.flip = 0
        
#        self.coil_start = np.random.randint(1, 6)
#        self.arm_start = np.random.randint(25, 31)
#        self.coil_width = np.random.randint(2, 8- self.coil_start + 1)
#        self.arm_width = np. random.randint(3, 35 - self.arm_start)
#        self.arm_length = np.random.randint(13, 17)
#        self.window_length = np.random.randint(self.arm_length, 17)
#        self.coil_length = np.random.randint(6, self.window_length -5)
#        self.window_width = self.arm_start - self.coil_start - self.coil_width - 1
        
        self.coil_start = 2
        self.arm_start = 27
        self.coil_width = 3
        self.arm_width = 6
        self.arm_length = 15
        self.window_length = 15
        self.coil_length = 9
        self.window_width = 21

        self.create_geo()
                
        #print('RESETing with a state[',self.posR,',',self.posC, '] = ', ob[self.posR, self.posC])
#        self.posC = self.coil_start + self.coil_width + self.worm_step_size -2
#        self.posR = self.worm_step_size-1
        
        self.buffer = 3
        ob = np.ones(self.env_dim)*3
        ob[self.buffer : self.buffer + self.window_length, 
           self.coil_start + self.coil_width : 
               self.coil_start + self.coil_width + self.window_width] = 0
        ob[self.buffer: self.buffer + self.coil_length, 
           self.coil_start: self.coil_start + self.coil_width] = 2
        coil2_start = 8 + 8 - (self.coil_start + self.coil_width)
        ob[self.buffer: self.buffer + self.coil_length , 
           coil2_start : coil2_start + self.coil_width] = 2
        ob[self.buffer: self.buffer + self.arm_length, 
           self.arm_start : self.arm_start + self.arm_width] = 4
        
        #print('RESETing with a state[',self.posR,',',self.posC, '] = ', ob[self.posR, self.posC])
        self.posC = self.posC = self.coil_start + self.coil_width + self.worm_step_size -2
        self.posR = self.worm_step_size + 1
                  
        self.state_hist.append((np.copy(ob[self.buffer:-1, :])))

        ob = self.move(ob)    
#        ob = self.clean_up(ob)
        self.past_states.append(np.copy(ob))
        self.past_actions.append(4)
        B_Arm = self.reward(ob)
        self.past_BArms.append(np.copy(B_Arm))
        warnings.filterwarnings("ignore")
        imageio.imwrite(FEMM_PATH + 'first.png', img_as_uint(ob/5))
        self.frame_idx += 1
      #  print('reset done')
#        state_index = np.argmax(ob[1:16, 5:26])
        ob2 = ob[self.buffer: self.buffer + 15, 5:27]
#        ob3 = np.copy(ob2)
#        ob3[:, -1] = np.copy(self.B_state)
        
        if self.flip == 1:
            ob3 = np.lib.pad(ob2, ((0, 21)), 'reflect')
            new_channel = np.lib.pad(self.B_dist[self.buffer: self.buffer+15, 5:27], ((0, 21)), 'reflect')
            ob4 = np.concatenate((np.expand_dims(ob3, 0), np.expand_dims(new_channel, 0)), axis=0)
            ob5 = ob4[:, :15, -22:]                         
        else:
            ob4 = np.expand_dims(ob2, 0)
            ob5 = np.concatenate((ob4, np.expand_dims(self.B_dist[self.buffer: self.buffer+15, 5:27], 0)), axis=0)
                       
        self.past_BnStates.append(np.copy(ob5))
#        print('ob5.shape', ob5.shape)
      
        return(ob5)

    def render(self, train= False):

        if train == True:    
            eps_dir = FEMM_PATH + TRAIN_PATH + '\\Eps_' + str(self.frame_idx) + '\\'
        else:
            eps_dir = FEMM_PATH + MODEL_PATH + 'Eps_' + str(self.frame_idx) + '\\'   
        
        if not (os.path.exists(FEMM_PATH + MODEL_PATH)):
            os.mkdir(FEMM_PATH + MODEL_PATH)

        if not (os.path.exists(eps_dir)):
            os.mkdir(eps_dir)

        toP = np.copy(self.past_states[-1])
        toP = toP[self.buffer : self.buffer+15, 5:26]
#        toP[toP == 1] = 5
        toP = toP/np.max(toP)
        imageio.imwrite(eps_dir + str(self.frame_idx) + '_Step-' + str(self.step_count) + '_Issue-' + str(self.issue) + '_reward-' + str(self.R) + '.png', img_as_uint(toP))
        
def save_file(src, fileName):
    dst = FEMM_PATH + MODEL_PATH + fileName + '.py'    
    copyfile(src, dst)

def get_file(path_file):
    path_components = path_file.split('\\')
    for component in path_components:
        file_sre = re.search('(.*).py', component)
#        file_name = file_sre.group(1)
    return file_sre     

def get_file_back_slash(path_file):
    path_components = path_file.split('/')
    for component in path_components:
        file_sre = re.search('(.*).py', component)
#        file_name = file_sre.group(1)
    return file_sre     

def create_txt_file(txt, name):
    with open(FEMM_PATH + MODEL_PATH + name + "_history.txt", "w") as myfile:
        myfile.write(txt)
        
def append_txt_file(txt, name):
    with open(FEMM_PATH + MODEL_PATH + name + "_history.txt", "a") as myfile:
            myfile.write(txt)
      
def save_obj(obj, name):    
    with open(FEMM_PATH + MODEL_PATH + name + '.pkl', 'wb') as f:        
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)            

def load_obj(name ):    
    with open('obj/' + name + '.pkl', 'rb') as f:        
        return pickle.load(f)

    ##### ACTIONS #######
    # 0     : RIGHT     #
    # 1     : LEFT      #
    # 2     : UP        #
    # 3     : DOWN      #
    ##-----------------##
    # 4     : RIGHT     #
    # 5     : LEFT      #
    # 6     : UP        #
    # 7     : DOWN      #
    #####################
    
