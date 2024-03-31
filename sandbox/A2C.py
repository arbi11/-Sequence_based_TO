#import gym
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import constants as con
from env_Worm_rB_1N3_BMag_CNN import WormFemmEnv2, append_txt_file, create_txt_file

#import matplotlib
#import matplotlib.pyplot as plt

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.conv1 = kl.Convolution2D(filters= con.filter_no[0], kernel_size = 3, padding= "same", activation='relu')
        self.conv2 = kl.Convolution2D(filters= con.filter_no[1], kernel_size = 3, strides= con.strides[1], padding= "same", activation='relu')
        self.flatten = kl.Flatten()
        
        self.hidden1 = kl.Dense(con.hidden_size[0], activation='relu')
        self.hidden2 = kl.Dense(con.hidden_size[1], activation='relu')
        
        self.value = kl.Dense(1, name='value')
        
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
    
class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma'     : con.gamma,
            'value'     : con.value_multiplier,
            'entropy'   : con.entropy
        }
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=con.lr),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
    
    def train(self, env, batch_sz=con.max_steps*3, updates=500):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) +  tuple(con.state_size))
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        obs = env.reset()
#        print('obs.shape',obs.shape)
        obs = np.concatenate((obs, obs, obs, obs), axis=0)
#        print('obs.shape',obs.shape)
        

        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = obs.copy()
                actions[step], values[step] = self.model.action_value(obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_obs = np.concatenate((next_obs, obs[:-2]), axis=0)                
                env.render(train=False)
#                env.render(train=True)
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)                    
                    logging.info("Episode: %03d, Reward: %.2f, Count: %d, Net Force: %.3f Flip: %d" % (env.frame_idx, ep_rews[-2], env.count, env.net_force, env.flip))
                    append_txt_file("\n Episode: %03d, Reward: %.2f, Count: %d, Net Force: %.3f, Flip: %d" % (env.frame_idx, ep_rews[-2], env.count, env.net_force, env.flip), file_name)
                    next_obs = env.reset()
                    next_obs = np.concatenate((next_obs, obs[:-2]), axis=0)
                else:
                    obs = next_obs.copy()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update+1, updates, losses))
            
            if update % con.playTime == 0:
                print('Playing after Epoch: {:d} and update {:d} ---> \n'.format(env.frame_idx, update))
                append_txt_file('\n \t Playing after Epoch: {:d} and update {:d} ---> \n'.format(env.frame_idx, update), file_name)
                rewards_sum, net_force, iron_c = agent.test(env)
                
                print("Total Episode Reward: {:.2f} iron_c: {} with net force {:.2f} & Flip {:d}".format(rewards_sum, iron_c, net_force, env.flip))
                append_txt_file("\n \t Total Episode Reward: {:.2f} iron_c: {} with net force {:.2f} & Flip {:d}".format(rewards_sum, iron_c, net_force, env.flip), file_name)
                
        return ep_rews

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        obs = np.concatenate((obs, obs, obs, obs), axis=0)

        while not done:
            action, _ = self.model.action_value(obs[None, :])
            next_obs, reward, done, _ = env.step(action.item())
            next_obs = np.concatenate((next_obs, obs[:-2]), axis=0)            

            ep_reward += reward
            if render:
                env.render(train= False)                
            if not done:
                obs = next_obs.copy()
                
        return ep_reward, env.net_force, env.count

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages
    
    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss
        
env = WormFemmEnv2()
model = Model(num_actions=con.action_dim)
#model.action_value(env.reset()[None, :])  

#env = gym.make('CartPole-v0')
#model = Model(num_actions=env.action_space.n)
agent = A2CAgent(model)  

file_name = "A2C_lr_{:.4f}_gamma_{:.2f}_entropy_{:.4f}_value_{:.2f}".format(con.lr, con.gamma, con.entropy, con.value_multiplier)
text = 'Acer_Data_Eval_DQL_1N3_A2C_NL_BAG_Stack_Stride_CNN \n'
create_txt_file(text, file_name)
text = "Penalty: {}, state_size {}, max_iron {}, hidden_size {} \n".format(con.penalty, con.state_size, con.max_iron, con.hidden_size)
append_txt_file(text, file_name)
text = "kernel_size {}, strides {} \n".format(con.kernel_size, con.strides)
append_txt_file(text, file_name)

#rewards_sum = agent.test(env)
#print("Total Episode Reward: %d out of 200" % agent.test(env))

logging.getLogger().setLevel(logging.INFO)

rewards_history = agent.train(env)
print("Finished training.")