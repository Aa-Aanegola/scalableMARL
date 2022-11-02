from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.spaces import Box, Discrete
import time, os, random, pdb
import algos.maTT.core as core
from algos.maTT.replay_buffer import ReplayBufferSet as ReplayBuffer

__author__ = 'Aakash Aanegola'
__copyright__ = ''
__credits__ = ['Aakash Aanegola', 'SpinningUp']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Aakash Aanegola'
__email__ = 'aakash.aanegola@students.iiit.ac.in'
__status__ = 'Dev'


SET_EVAL_v0 = [
        {'nb_agents': 1, 'nb_targets': 1},
        {'nb_agents': 2, 'nb_targets': 2},
        {'nb_agents': 3, 'nb_targets': 3},
        {'nb_agents': 4, 'nb_targets': 4},
]

def eval_set(num_agents, num_targets):
    agents = np.linspace(num_agents/2, num_agents, num=3, dtype=int)
    targets = np.linspace(num_agents/2, num_targets, num=3, dtype=int)
    params_set = [{'nb_agents':1, 'nb_targets':1},
                  {'nb_agents':4, 'nb_targets':4}]
    for a in agents:
        for t in targets:
            params_set.append({'nb_agents':a, 'nb_targets':t})
    return params_set

def test_agent(test_env, get_action, logger, num_test_episodes, 
                num_agents, num_targets, render=False):
    """ Evaluate current policy over an environment set
    """
    ## Either manually set evaluation set or auto fill
    params_set = SET_EVAL_v0
    # params_set = eval_set(num_agents, num_targets)

    for params in params_set:
        for j in range(num_test_episodes):
            done, ep_ret, ep_len = {'__all__':False}, 0, 0
            obs = test_env.reset(**params)
            while not done['__all__']:
                if render:
                    test_env.render()
                action_dict = {}
                for agent_id, o in obs.items():
                    action_dict[agent_id] = get_action(o, deterministic=False)

                obs, rew, done, _ = test_env.step(action_dict)
                ep_ret += rew['__all__']
                ep_len += 1  
            # logger.store(TestEpRet=ep_ret)


def Qlearning(env_fn, model=core.DeepSetModel2, model_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-2, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=4000, update_every=1, num_test_episodes=5, max_ep_len=200, 
        logger_kwargs=dict(), save_freq=1, lr_period=0.7, grad_clip=5, render=False,
        torch_threads=1, amp=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(torch_threads)

    writer = SummaryWriter(logger_kwargs['output_dir'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = deepcopy(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = model(env.observation_space, env.action_space, **model_kwargs).to(device)
    policy_targ = deepcopy(policy).to(device)

    for p in policy_targ.parameters():
        p.requires_grad = False
        
    q_params = itertools.chain(policy.q.parameters())
    replay_buffer = ReplayBuffer(replay_size, obs_dim, act_dim)

    var_counts = tuple(core.count_vars(module) for module in [policy.q])
    
    q_optimizer = Adam(q_params, lr=lr)

    if amp:
        scaler = torch.cuda.amp.GradScaler()

    def compute_loss_q(data):
        obs = data['obs'].to(device)
        act = data['act'].type(torch.LongTensor).to(device)
        rew = data['rew'].to(device)
        obs2 = data['obs2'].to(device)
        done = data['done'].type(torch.float32).to(device)  

        q = policy.q(obs,act)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                v = policy.q.values(obs2)
                act, logp_a = policy.pi(v)

                q_pi_targ = policy_targ.q(obs2, act)            
                backup = rew.unsqueeze(1) + gamma * (q_pi_targ - alpha * logp_a)

        huber = torch.nn.SmoothL1Loss()
        loss_q = huber(q, backup)

        try:
            q_info = dict(QVals=q.detach().numpy())
        except:
            q_info = dict(QVals=q.cpu().detach().numpy())

        return loss_q, q_info

    def update(data, lr_iter):
        lr = np.clip(0.0005*np.cos(np.pi*lr_iter/(total_steps*lr_period))+0.000501, 1e-5, 1e-3)
        q_optimizer.param_groups[0]['lr'] = lr

        q_optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                loss_q, q_info = compute_loss_q(data)
            scaler.scale(loss_q).backward()
            scaler.unscale_(q_optimizer)
            torch.nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
            scaler.step(q_optimizer)
            scaler.update()

        else:
            loss_q, q_info = compute_loss_q(data)
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(policy.parameters(), grad_clip)
            q_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(policy.parameters(), policy_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(obs, deterministic=False):
        return policy.act(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device), deterministic)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    ep_ret, ep_len, best_test_ret = 0, 0, 0
    obs = env.reset()

    for t in range(total_steps):
        # if t%100 == 0:
        #     print(f"Currently at step {t} out of {total_steps}")

        action_dict = {}
        if t > start_steps:
            for agent_id, o in obs.items():
                action_dict[agent_id] = get_action(o, deterministic=False)
        else:
            for agent_id, o in obs.items():
                action_dict[agent_id] = env.action_space.sample()
        
        obs2, rew, done, info = env.step(action_dict)
        ep_ret += rew['__all__']
        ep_len += 1

        done['__all__'] = False if ep_len==max_ep_len else False

        for agent_id, o in obs.items():
            replay_buffer.store(o, action_dict[agent_id], rew['__all__'], 
                                obs2[agent_id], float(done['__all__']))

        obs = obs2

        if done['__all__'] or (ep_len == max_ep_len):
            ep_ret, ep_len = 0, 0
            obs = env.reset()

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                if t < total_steps*(1-lr_period):
                    lr_iter = 0
                else:
                    lr_iter = t-total_steps*(1-lr_period)

                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, lr_iter=lr_iter)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            
            test_agent(test_env, get_action, None, num_test_episodes, env.num_agents, env.num_targets)

            if (epoch % save_freq == 0) or (epoch == epochs):
                fpath = logger_kwargs['output_dir']+'/state_dict/'
                os.makedirs(fpath, exist_ok=True)
                torch.save(policy.state_dict(), fpath+'model%d.pt'%epoch)


            # Tensorboard logger
            # writer.add_scalar('AverageEpRet', logger.log_current_row['AverageEpRet'],t)
            # writer.add_scalar('AverageTestEpRet', logger.log_current_row['AverageTestEpRet'],t)
            # writer.add_scalar('AverageQ1Vals', logger.log_current_row['AverageQ1Vals'],t)
            # writer.add_scalar('AverageQ2Vals', logger.log_current_row['AverageQ2Vals'],t)
            # writer.add_scalar('HuberLossQ', logger.log_current_row['LossQ'],t)
            # writer.add_scalar('LearningRate', logger.log_current_row['LR'],t)

            # logger.dump_tabular()