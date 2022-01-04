import os
import os.path as osp
import time
from collections import deque
import pickle

from baselines.ddpg.agent_load_manager import AgentManager
from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds

from baselines import logger
import tensorflow as tf
import numpy as np
from baselines.ddpg.things import *



def get_noise_type(noise_type, nb_actions):
    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    return action_noise, param_noise


def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=10000, # with default settings, perform 1M steps total
          nb_rollout_steps=10000,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=3e-4,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=1000, # per epoch cycle and MPI worker,
          nb_eval_steps=50,
          batch_size=128, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          load_path= './model/',
          save_path = './model/',
          **network_kwargs):

    CHECK_MPI_SINGLE()
    result_plot = ResultPlot()
    set_global_seeds(seed)
    rank = 0

    print(type(env))


    nb_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(nb_actions, ob_shape=env.observation_space.shape, network=network, **network_kwargs)
    actor = Actor(nb_actions, ob_shape=env.observation_space.shape, network=network, **network_kwargs)
    action_noise, param_noise = get_noise_type(noise_type, nb_actions)

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    MANAGER = AgentManager(agent, load_path, save_path)


    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    # Prepare everything.
    agent.initialize()
    agent.reset()
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]



    if load_path is None:
        os.makedirs (save_path, exist_ok=True)

    mpi_size = 1

    start_time = time.time()
    total_step = 0
    for epoch in range(nb_epochs):
        rollout_time = 0
        train_time = 0
        episode_reward = 0  # vector
        episode_step = 0  # vector
        epoch_actions = []
        epoch_qs = []

        EPOCH_DONE = False
        while not EPOCH_DONE:
            rollout_start = time.time ()
            for t_rollout in range(nb_rollout_steps):
                episode_step += 1
                total_step +=1

                # Predict next action.
                action, q, _, _ = agent.step(tf.constant(obs), apply_noise=True, compute_Q=True)
                action, q = action.numpy(), q.numpy()
                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch

                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])


                if render: env.render()

                episode_reward += r[0]
                epoch_actions.append(action[0])
                epoch_qs.append(q[0])


                agent.store_transition(obs, action, r, new_obs, done) #the batched data will be unrolled in memory.py's append.
                obs = new_obs
                check_NAN ([action, q, obs])

                if done[0]:
                    EPOCH_DONE = True
                    result_plot.update(epoch+1, info[0])
                    # Episode done.
                    episode_rewards_history.append(episode_reward)
                    agent.reset()

                if EPOCH_DONE: break

            rollout_time = rollout_time + (time.time() - rollout_start)


            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []

            st = time.time ()
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    batch = agent.memory.sample(batch_size=batch_size)
                    obs0 = tf.constant(batch['obs0'])
                    distance = agent.adapt_param_noise(obs0)
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            train_time = train_time + (time.time () - st)

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0




        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = int(episode_reward)
        combined_stats['rollout/return_history'] = int(np.mean(episode_rewards_history))
        combined_stats['rollout/episode_steps'] = episode_step
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)

        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = int(duration)
        combined_stats['total/steps_per_second'] = float(total_step) / float(duration)
        combined_stats['total/epi,epochs'] = "{} / {}".format(epoch + 1, nb_epochs)
        combined_stats['total/steps'] = total_step

        combined_stats['A/rollout_time'] = int (rollout_time)
        combined_stats['A/train_time'] = int (train_time)



        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        combined_stats = {k : v  for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)
        MANAGER.save()

    return agent
