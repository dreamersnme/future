import os
import shutil
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
          train_steps =500,
          batch_size = 128,
          render=True,
          log_dir='./sac_log/',
          summary_dir ='./summary/sac'):

    os.makedirs (log_dir, exist_ok=True)
    shutil.rmtree(summary_dir, ignore_errors=True)
    os.makedirs (summary_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(summary_dir, filename_suffix=None)

    state_shape = env.observation_space.shape
    # TODO: fix this when env.action_space is not `Box`
    action_shape = env.action_space.shape
    action_space = action_shape[0]
    # Initialize Replay buffer.
    memory=Memory (limit=int (1e6), action_shape=action_shape, observation_shape=state_shape)

    agent = SoftActorCritic (memory, action_space, state_shape, writer,
                           learning_rate=lr,batch_size=batch_size,
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


        while not done:
            action = agent.step (tf.constant(current_state, dtype=tf.float32)) if np.random.uniform () < noise_rt else env.action_space.sample ()
            try:
                next_state, reward, done, _ = env.step(action)
            except:
                print(current_state)
                print(action)
                raise Exception

            episode_reward += reward
            end = 0 if done else 1
            agent.store_transition (current_state, action, reward, next_state, end)
            current_state = next_state

            step += 1
            global_step += 1
        S_TM = int(time.time() - clock)
        if render: env.render ()

        clock = time.time()
        ttt = 0
        www = 0

        al=[]
        cl1=[]
        cl2=[]
        apl=[]
        for epoch in range (train_steps):
            # Randomly sample minibatch of transitions from replay buffer
            st = time.time()
            critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.train ()
            ttt += (time.time()-st)

            st = time.time()
            al.append(actor_loss)
            cl1.append(critic1_loss)
            cl2.append(critic2_loss)
            apl.append(alpha_loss)

            # with writer.as_default ():
            #     tf.summary.scalar ("actor_loss", actor_loss, agent.epoch_step)
            #     tf.summary.scalar ("critic1_loss", critic1_loss, agent.epoch_step)
            #     tf.summary.scalar ("critic2_loss", critic2_loss, agent.epoch_step)
            #     tf.summary.scalar ("alpha_loss", alpha_loss, agent.epoch_step)


            www += (time.time() - st)
            agent.epoch_step += 1
            agent.update_weights ()


        T_TM = int(time.time() - clock)
        MANAGER.save()

        print("=====================================")
        print(f"sampling {S_TM} train: {T_TM}")
        print(f"ttt {ttt} www: {www}")
        print (f"Episode {episode} reward: {episode_reward}")
        with writer.as_default ():
            tf.summary.scalar ("episode_reward", episode_reward, episode)
            tf.summary.scalar ("actor_loss", np.mean(np.array(al)), agent.epoch_step)
            tf.summary.scalar ("critic1_loss", np.mean(np.array(cl1)), agent.epoch_step)
            tf.summary.scalar ("critic2_loss", np.mean(np.array(cl2)), agent.epoch_step)
            tf.summary.scalar ("alpha_loss", np.mean(np.array(apl)), agent.epoch_step)
