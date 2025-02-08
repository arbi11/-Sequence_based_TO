M_NO                = 1
MACHINE             = 'Acer'
#PATH = "D:\\Documents\\Codes\\RL-Acer\\RL\\PPO"
PATH                = r'C:\Users\chetanm\Desktop\Deep Learning\A2C_Tf2_CNN_stack_flip'

MODEL_PATH          = 'Acer_Data_Eval_DQL_1N3_A2C_NL_BAG_Stack_Stride_CNN'
TRAIN_PATH          = 'Ampere_Data_Train_DQL_1N3_A2C_NL'
max_steps           = 75
env_dim             = [18, 35]
action_dim          = 8
state_size          = [8, 15, 22] # 4*2: 4 previous states and 2 for channels
max_iron            = 190

lr                  = 0.0007
batch_size          = 1024
miniBatchSize       = 512
penalty             = -10

gamma               = 0.95                             # Discounting rate
playTime            = 2

warmUpSteps         = 500 #200
experienceLength    = 2500
hidden_size         = [256, 64]
kernel_size 		= [3, 3]
filter_no   		= [8, 16]
strides 			= [1, 2]
variance            = 0.4
num_episodes        = 500

max_eps             = 1.0
min_eps             = 0.01
decay_rate          = 0.001
 
tau                 = 0.001
entropy             = 0.001
value_multiplier    = 0.5
