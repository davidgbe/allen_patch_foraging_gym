import sys
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import torch
import numpy as np
import os
import gymnasium as gym
from tqdm.auto import trange
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle
from copy import deepcopy as copy
import matplotlib.pyplot as plt

from environments.treadmill_session import TreadmillSession
from environments.components.patch_type import PatchType
from environments.curriculum import Curriculum
from agents.wsls_agent import WSLSAgent
from tools.general import zero_pad, make_path_if_not_exists, compressed_write


# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str, default='run')
parser.add_argument('--curr_style', metavar='cs', type=str, default='FIXED')
parser.add_argument('--env', metavar='e', type=str, default='LOCAL')
args = parser.parse_args()

curr_file_path = Path(__file__)


# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 1
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
MAX_REWARD_SITE_LEN = 2
MIN_REWARD_SITE_LEN = 2
INTERREWARD_SITE_LEN_MEAN = 2
REWARD_DECAY_CONSTS = [0, 10, 30]
REWARD_PROB_PREFACTOR = 0.8
INTERPATCH_LEN = 6
CURRICULUM_STYLE = args.curr_style

# TRAINING PARAMS
NUM_ENVS = 30
N_SESSIONS = 20
N_UPDATES_PER_SESSION = 100
N_STEPS_PER_UPDATE = 200

# OTHER PARMS
OUTPUT_BASE_DIR = './results'
OUTPUT_STATE_SAVE_RATE = 1 # save every session


def make_deterministic_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN

        print('Begin det. treadmill')

        patch_types = []
        for i in range(PATCH_TYPES_PER_ENV):
            def reward_func(site_idx):
                return 1
            patch_types.append(
                PatchType(
                    reward_site_len_for_patches[i],
                    INTERREWARD_SITE_LEN_MEAN,
                    reward_func,
                    i,
                    reward_func_param=0.0,
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patch_types,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            obs_size=PATCH_TYPES_PER_ENV + 1,
            verbosity=False,
        )

        return sesh

    return make_env


def make_stochastic_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx + NUM_ENVS)
        
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        decay_consts_for_reward_funcs = copy(REWARD_DECAY_CONSTS)
        if CURRICULUM_STYLE == 'MIXED':
            np.random.shuffle(decay_consts_for_reward_funcs)

        print('Begin stoch. treadmill')
        print(decay_consts_for_reward_funcs)

        patch_types = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            active = (decay_const_for_i != 0)
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i, active=active):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i) if decay_const_for_i > 0 else 0
                if np.random.rand() < c and active:
                    return 1
                else:
                    return 0
            patch_types.append(
                PatchType(
                    reward_site_len_for_patches[i],
                    INTERREWARD_SITE_LEN_MEAN,
                    reward_func,
                    i,
                    reward_func_param=(decay_consts_for_reward_funcs[i] if active else 0.0),
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patch_types,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            obs_size=PATCH_TYPES_PER_ENV + 1,
            verbosity=False,
        )

        return sesh

    return make_env


def run_treadmill_session():
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp]))
    reward_rates_output_dir = os.path.join(output_dir, 'reward_rates')
    info_output_dir = os.path.join(output_dir, 'state')
    make_path_if_not_exists(reward_rates_output_dir)
    make_path_if_not_exists(info_output_dir)

    curricum = Curriculum(
        curriculum_step_starts=[0],
        curriculum_step_env_funcs=[
            make_stochastic_treadmill_environment,
        ],
    )

    env_seeds = np.arange(NUM_ENVS)
    save_num = 0
    last_snapshot = None

    for session_num in trange(N_SESSIONS, desc='Sessions'):

        agent = WSLSAgent(
            n_envs=NUM_ENVS,
            wait_time_for_reward=DWELL_TIME_FOR_REWARD,
            odor_cues_indices=(1, 4),
            patch_cue_idx=0,
        )

        avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES_PER_SESSION))
        all_info = []
        envs = curricum.get_envs_for_step(env_seeds)
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        for update_num in trange(N_UPDATES_PER_SESSION, desc='Updates in session'):
            if update_num == 0:
                obs, info = envs.reset()

            total_rewards = np.empty((NUM_ENVS, N_STEPS_PER_UPDATE))
            for step in range(N_STEPS_PER_UPDATE):
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = envs.step(action)
                agent.append_reward(reward)
                total_rewards[:, step] = reward
                all_info.append(info)

            avg_rewards_per_update[:, update_num] = np.mean(total_rewards, axis=1)
        
        agent.reset_state()


        padded_save_num = zero_pad(str(save_num), 5)
        np.save(os.path.join(reward_rates_output_dir, f'{padded_save_num}.npy'), avg_rewards_per_update)
        print('Avg reward for session:', np.mean(avg_rewards_per_update))
        if session_num % OUTPUT_STATE_SAVE_RATE == 0:
            try:
                compressed_write(all_info, os.path.join(info_output_dir, f'{padded_save_num}.pkl'))
            except MemoryError as me:
                print('Pickle dump caused memory crash')
                print(me)
                pass
        save_num += 1
        agent.reset_state()

    final_value = avg_rewards_per_update.mean()
    print('final_reward_total', final_value)

    return final_value

if __name__ == "__main__":
    run_treadmill_session()