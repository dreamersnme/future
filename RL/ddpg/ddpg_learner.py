import os


from RL.agent_load_manager import AgentManager
from RL.ddpg.ddpg import DDPG
from RL.ddpg.models import Actor, Critic
from RL.memory import Memory
from RL.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from baselines import logger
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
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
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                            sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    return action_noise, param_noise


def learn(env, network='mlp',
          seed=None,
          total_timesteps=None,
          nb_epochs=10000,  # with default settings, perform 1M steps total
          nb_rollout_steps=100,
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
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          nb_eval_steps=50,
          batch_size=128,  # per MPI worker
          tau=0.01,
          param_noise_adaption_interval=50,
          log_dir='./ddpg_log/',
          summary_dir='./summary/ddpg'):

    os.makedirs (log_dir, exist_ok=True)
    os.makedirs (summary_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(summary_dir, filename_suffix=None)

    nb_actions = env.action_space.shape[-1]

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(nb_actions, ob_shape=env.observation_space.shape, network=network )
    actor = Actor(nb_actions, ob_shape=env.observation_space.shape, network=network)
    action_noise, param_noise = get_noise_type(noise_type, nb_actions)

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    MANAGER = AgentManager(agent, log_dir, log_dir)

    # Prepare everything.
    agent.initialize()
    agent.reset()
    obs = env.reset()


    total_step = 0
    for episode in range(nb_epochs):
        episode += 1
        episode_reward = 0



        done = False
        clock = time.time()
        while not done:
            total_step += 1
            # Predict next action.
            action, q, _, _ = agent.step(tf.constant(obs), apply_noise=True, compute_Q=True)
            action, q = action.numpy(), q.numpy()
            # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch

            new_obs, r, done, info = env.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            if render: env.render()

            episode_reward += r
            agent.store_transition(obs, action, r, new_obs,done)  # the batched data will be unrolled in memory.py's append.
            obs = new_obs

        agent.reset()
        # Train.

        S_TM = int(time.time() - clock)
        if render: env.render ()

        clock = time.time()
        for t_train in range(nb_train_steps):
            # Adapt param noise, if necessary.
            if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                obs0, actions, rewards, obs1, ends = memory.fetch_sample(batch_size=batch_size)
                obs0 = tf.constant(obs0)
                distance = agent.adapt_param_noise(obs0)


            cl, al = agent.train()
            agent.update_target_net()
            with writer.as_default ():
                tf.summary.scalar ("actor_loss", al, agent.epoch_step)
                tf.summary.scalar ("critic1_loss", cl, agent.epoch_step)

            agent.epoch_step += 1


        T_TM = int(time.time() - clock)
        MANAGER.save()

        print("=====================================")
        print(f"sampling {S_TM} train: {T_TM}")
        print (f"Episode {episode} reward: {episode_reward}")
        with writer.as_default ():
            tf.summary.scalar ("episode_reward", episode_reward, episode)

