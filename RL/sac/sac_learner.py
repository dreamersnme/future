import os
import time

import tensorflow as tf
import numpy as np

from RL.agent_load_manager import AgentManager
from RL.memory import Memory
from RL.sac.replay_buffer import ReplayBuffer
from RL.sac.sac import SoftActorCritic
obs_range=(-5., 5.)

def learn(env,
          lr=1e-3,
          gamma=0.99,
          polyak=0.995,
          nb_epochs=10000,
          noise_rt = 0.1,
          train_steps =1000,
          batch_size = 128,
          render=True,
          log_dir='./sac_log/',
          summary_dir ='./summary/sac'):

    os.makedirs (log_dir, exist_ok=True)
    os.makedirs (summary_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(summary_dir, filename_suffix=None)

    state_shape = env.observation_space.shape
    # TODO: fix this when env.action_space is not `Box`
    action_shape = env.action_space.shape
    action_space = action_shape[0]
    # Initialize Replay buffer.
    memory=Memory (limit=int (1e6), action_shape=action_shape, observation_shape=state_shape)

    agent = SoftActorCritic (memory, action_space, state_shape, writer,
                           learning_rate=lr,
                           gamma=gamma, polyak=polyak)

    MANAGER = AgentManager (agent, log_dir, log_dir)


    # Repeat until convergence
    global_step = 1
    for episode in range(nb_epochs):
        episode += 1
        # Observe state
        current_state = env.reset ()
        step = 1
        episode_reward = 0
        done = False
        clock = time.time()
        print(current_state)

        while not done:
            action = agent.sample_action (current_state) if np.random.uniform () < noise_rt else env.action_space.sample ()
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            end = 0 if done else 1
            agent.store_transition (current_state, action, reward, next_state, end)
            current_state = next_state

            step += 1
            global_step += 1
        S_TM = int(time.time() - clock)
        if render: env.render ()

        clock = time.time()
        for epoch in range (train_steps):
            # Randomly sample minibatch of transitions from replay buffer
            current_states, actions, rewards, next_states, ends = memory.fetch_sample (batch_size=batch_size)
            critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.train (current_states, actions, rewards, next_states, ends)

            with writer.as_default ():
                tf.summary.scalar ("actor_loss", actor_loss, agent.epoch_step)
                tf.summary.scalar ("critic1_loss", critic1_loss, agent.epoch_step)
                tf.summary.scalar ("critic2_loss", critic2_loss, agent.epoch_step)
                tf.summary.scalar ("alpha_loss", alpha_loss, agent.epoch_step)

            agent.epoch_step += 1
            agent.update_weights ()



        T_TM = int(time.time() - clock)
        MANAGER.save()

        print("=====================================")
        print(f"sampling {S_TM} train: {T_TM}")
        print (f"Episode {episode} reward: {episode_reward}")
        with writer.as_default ():
            tf.summary.scalar ("episode_reward", episode_reward, episode)

