import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np

from RL.agent_utils import *
from baselines.common.mpi_running_mean_std import RunningMeanStd

tf.keras.backend.set_floatx('float32')

EPSILON = 1e-16



class Actor(Model):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.dense1_layer = layers.Dense(32, activation=tf.nn.relu)
        self.dense2_layer = layers.Dense(32, activation=tf.nn.relu)
        self.mean_layer = layers.Dense(self.action_dim)
        self.stdev_layer = layers.Dense(self.action_dim)
        small_sigma = tf.zeros(self.action_dim)
        self.small_sigma = tf.add(small_sigma, 1e-3)


    @tf.function
    def call(self, state):
        # Get mean and standard deviation from the policy network
        a1 = self.dense1_layer(state)
        a2 = self.dense2_layer(a1)
        mu = self.mean_layer(a2)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]  ????????????
        log_sigma = self.stdev_layer(a2)
        log_sigma = tf.clip_by_value(log_sigma, -10, 10)
        #print(9898988, log_sigma)
        sigma = tf.exp(log_sigma)
        # sigma = tf.maximum(sigma, self.small_sigma)


        # Use re-parameterization trick to deterministically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev


        dist = tfp.distributions.Normal(mu, sigma, allow_nan_stats=False)
        action_ = dist.sample()
        #print(8888, action_)
        #print(55, mu)
        #print(666,sigma)

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        action = tf.tanh(action_)

        # Calculate the log probability
        log_pi_ = dist.log_prob(action_)
        #print(log_pi_)

        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper
        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 + EPSILON), axis=1,
                                         keepdims=True)
        # 
        # if tf.math.is_nan (tf.reduce_sum(action)):
        #     #print(state)
        #     #print(tf.reduce_sum(state))
        # 
        #     #print(tf.reduce_sum(a1))
        #     print (tf.reduce_sum (a2))
        #     print (tf.reduce_sum (mu))
        #     print (tf.reduce_sum (log_sigma))
        #     #print(tf.reduce_sum(sigma))
        #     print (tf.reduce_sum (action_))
        #     print (action)
        #     print (tf.reduce_sum (log_pi))
        #     #print(self.trainable_variables)
        #     raise Exception("NAN")

        return action, log_pi

    @property
    def trainable_variables(self):#????????????????????
        return self.dense1_layer.trainable_variables + \
                self.dense2_layer.trainable_variables + \
                self.mean_layer.trainable_variables + \
                self.stdev_layer.trainable_variables

class Critic(Model):

    def __init__(self):
        super().__init__()
        self.dense1_layer = layers.Dense(32, activation=tf.nn.relu)
        self.dense2_layer = layers.Dense(32, activation=tf.nn.relu)
        self.output_layer = layers.Dense(1)

    @tf.function
    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        a1 = self.dense1_layer(state_action)
        a2 = self.dense2_layer(a1)
        q = self.output_layer(a2)
        return q

    @property
    def trainable_variables(self):
        return self.dense1_layer.trainable_variables + \
                self.output_layer.trainable_variables + \
                self.dense2_layer.trainable_variables


class SoftActorCritic(tf.Module):

    def __init__(self, memory, action_space, obs_shape, writer, epoch_step=1, learning_rate=0.0003, batch_size=128,
                 alpha=0.2, gamma=0.99, obs_range=(-5., 5.), return_range=(-5000.0, 5000.0),
                polyak=0.995, norm_obs=True):
        self.policy = Actor(action_space)
        self.q1 = Critic()
        self.q2 = Critic()
        self.target_q1 = Critic()
        self.target_q2 = Critic()
        self.batch_size = batch_size

        self.writer = writer
        self.epoch_step = epoch_step
        self.obs_range = obs_range

        self.alpha = tf.Variable(0.0, dtype=tf.float32)
        self.target_entropy = -tf.constant(action_space, dtype=tf.float32)
        self.gamma = gamma
        self.polyak = polyak

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.return_range =return_range

        self.memory = memory

        # Observation normalization.
        self.norm_obs = norm_obs
        # Observation normalization.
        if self.norm_obs:
            with tf.name_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=obs_shape)
        else:
            self.obs_rms = None

    def store_transition(self, obs0, action, reward, obs1, terminal1):

        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.norm_obs:
            self.obs_rms.update(np.array([obs0]))

    @tf.function
    def step(self, obs):

        # normalized_obs = tf.clip_by_value (normalize (obs, self.obs_rms), self.obs_range[0], self.obs_range[1])
        # normalized_obs = tf.reshape(normalized_obs, [1, -1])
        # action, _ = self.policy(normalized_obs)
        obs = tf.reshape(obs, [1, -1])
        action, _ = self.policy(obs)
        return action[0]

    @tf.function
    def update_q_network(self, current_states, actions, rewards, next_states, ends):

        with tf.GradientTape() as tape1:
            # Get Q value estimates, action used here is from the replay buffer
            q1 = self.q1(current_states, actions)
            # Sample actions from the policy for next states
            pi_a, log_pi_a = self.policy(next_states)

            # Get Q value estimates from target Q network
            q1_target = self.target_q1(next_states, pi_a)
            q2_target = self.target_q2(next_states, pi_a)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - self.alpha * log_pi_a
            y = tf.stop_gradient(rewards + self.gamma * ends * soft_q_target)

            critic1_loss = tf.reduce_mean((q1 - y)**2)

        with tf.GradientTape() as tape2:
            # Get Q value estimates, action used here is from the replay buffer
            q2 = self.q2(current_states, actions)

            # Sample actions from the policy for next states
            pi_a, log_pi_a = self.policy(next_states)

            # Get Q value estimates from target Q network
            q1_target = self.target_q1(next_states, pi_a)
            q2_target = self.target_q2(next_states, pi_a)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - self.alpha * log_pi_a
            y = tf.stop_gradient(rewards + self.gamma * ends * soft_q_target)

            critic2_loss = tf.reduce_mean((q2 - y)**2)

        grads1 = tape1.gradient(critic1_loss, self.q1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads1,
                                                   self.q1.trainable_variables))

        grads2 = tape2.gradient(critic2_loss, self.q2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(grads2,
                                                   self.q2.trainable_variables))

        with self.writer.as_default():
            for grad, var in zip(grads1, self.q1.trainable_variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)
            for grad, var in zip(grads2, self.q2.trainable_variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return critic1_loss, critic2_loss

    @tf.function
    def update_policy_network(self, current_states):
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = self.policy(current_states)
            #print(1111,pi_a)
            #print(1112, log_pi_a)
            # Get Q value estimates from target Q network
            q1 = self.q1(current_states, pi_a)
            q2 = self.q2(current_states, pi_a)
            #print(2222,q1)
            #print(3333,q2)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q = tf.minimum(q1, q2)
            #print(4444, min_q)
            soft_q = min_q - self.alpha * log_pi_a
            #print(5555,soft_q)
            #print(5555, self.alpha )

            actor_loss = -tf.reduce_mean(soft_q)
            #print(actor_loss)

        variables = self.policy.trainable_variables
        # copy = [tf.Variable(x)for x in self.policy.trainable_variables]
        # ssss = self.policy(current_states)

        # if tf.math.is_nan(tf.reduce_sum([tf.reduce_sum(x) for x in self.policy.trainable_variables])):
        #     #print(tf.reduce_sum([tf.reduce_sum(x) for x in self.policy.trainable_variables]))
        #     #print("---------", self.policy.trainable_variables)
        # 
        #     #print()
        #     #
        # 
        #     raise Exception("ddddddddddddddddd")

        grads = tape.gradient(actor_loss, variables)
        self.actor_optimizer.apply_gradients(zip(grads, variables))

        # if  tf.math.is_nan (tf.reduce_sum([tf.reduce_sum(x) for x in self.policy.trainable_variables])):
        #     #print(tf.reduce_sum(current_states))
        #     #print(1, current_states)
        #     #print(2, grads)
        #     #print(4442, copy)
        #     #print(9999, ssss)
        # 
        #     #print()
        #     #
        # 
        #     raise Exception("dsdsdds")
        #
        # with self.writer.as_default():
        #     for grad, var in zip(grads, variables):
        #         tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
        #         tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return actor_loss

    @tf.function
    def update_alpha(self, current_states):
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = self.policy(current_states)

            alpha_loss = tf.reduce_mean( - self.alpha*(log_pi_a +
                                                       self.target_entropy))

        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))

        with self.writer.as_default():
            for grad, var in zip(grads, variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return alpha_loss


    def train(self):

        obs0, actions, rewards, obs1, ends = self.memory.fetch_sample(batch_size=self.batch_size)
        # obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.obs_range[0], self.obs_range[1])
        # obs1 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.obs_range[0], self.obs_range[1])

        obs0, obs1 = tf.constant(obs0), tf.constant(obs1)
        actions, rewards, ends = tf.constant(actions), tf.constant(rewards), tf.constant(ends, dtype=tf.float32)

        # Update Q network weights
        critic1_loss, critic2_loss = self.update_q_network(obs0, actions, rewards, obs1, ends)

        # Update policy network weights
        actor_loss = self.update_policy_network(obs0)
        alpha_loss = self.update_alpha(obs0)

        # Update target Q network weights
        #self.update_weights()

        #if self.epoch_step % 10 == 0:
        #    self.alpha = max(0.1, 0.9**(1+self.epoch_step/10000))
        #    #print("alpha: ", self.alpha, 1+self.epoch_step/10000)

        return critic1_loss, critic2_loss, actor_loss, alpha_loss

    # @tf.function
    def update_weights(self):

        for theta_target, theta in zip(self.target_q1.trainable_variables,
                                       self.q1.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

        for theta_target, theta in zip(self.target_q2.trainable_variables,
                                       self.q2.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta
