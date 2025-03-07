import numpy as np
import glob2 as glob
import pickle
import os
from .general import compressed_read


def load_numpy(data_path, averaging_size=1):
    file_names = glob.glob(data_path)
    data = []
    for file_name in file_names:
        data_for_file = np.load(file_name)
        data.append(data_for_file)
    data = np.concatenate(data, axis=1)
    if averaging_size == 1:
        return data
    reduced_data = np.empty((data.shape[0], int(data.shape[1] / averaging_size)))
    for k in range(0, int(data.shape[1] / averaging_size) * averaging_size, averaging_size):
        reduced_data[:, int(k / averaging_size)] = data[:, k:k + averaging_size].mean(axis=1)
    return reduced_data


def load_data(data_path, indices=[], verbose=False):
#     file_names = glob.glob(data_path)
    file_names = sorted(os.listdir(data_path))
    if verbose:
        print(file_names)
    for fidx, file_name in enumerate(file_names):
        if fidx in indices or (len(indices) == 0 and fidx + 1 == len(file_names)):
            data_for_file = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
            yield data_for_file


def load_compressed_data(data_path, indices=[], all=False, verbose=False):
    file_names = sorted(os.listdir(data_path))
    if verbose:
        print(file_names)
    for fidx, file_name in enumerate(file_names):
        if fidx in indices or (len(indices) == 0 and fidx + 1 == len(file_names)) or all:
            data_for_file = compressed_read(os.path.join(data_path, file_name))
            yield data_for_file


def load_behavioral_data(data_path, update_num=None, all=False):
    indices = [] if update_num is None else [update_num]
    state_data = load_compressed_data(
        data_path,
        indices=indices,
        all=all,
    )
    return state_data


def parse_behavioral_data(d, env_idx):
    features = [
        'agent_in_patch',
        'current_patch_start',
        'reward_bounds',
        'current_patch_num',
        'reward_site_idx',
        'action',
        'current_position',
        'reward',
        'patch_reward_param',
        'current_reward_site_attempted',
    ]

    features_to_time_series_dict = {}
    for f in features:
        features_to_time_series_dict[f] = []
        
    for k in np.arange(len(d)):
        for f in features:
            features_to_time_series_dict[f].append(d[k][f][env_idx])
            
    for f in features:
        try:
            features_to_time_series_dict[f] = np.array(features_to_time_series_dict[f])
        except e:
            print(e)

    all_dwell_times = []
    rewards_at_positions = [0]
    reward_attempted_at_positions = [False]
    dwell_time = 0
    last_p = None
    rewards_seen_in_patch = np.zeros((len(d)))
    
    for i, p in enumerate(features_to_time_series_dict['current_position']):
        if last_p is not None and (p != last_p):
            all_dwell_times.append(dwell_time)
            rewards_at_positions.append(0)
            reward_attempted_at_positions.append(False)
            dwell_time = 0
        if last_p is not None:
            dwell_time += 1
        rewards_at_positions[-1] += features_to_time_series_dict['reward'][i]
        reward_attempted_at_positions[-1] = True if features_to_time_series_dict['current_reward_site_attempted'][i] else reward_attempted_at_positions[-1]
        last_p = p

        if features_to_time_series_dict['agent_in_patch'][i]:
            if i > 0:
                rewards_seen_in_patch[i] = rewards_seen_in_patch[i-1] + features_to_time_series_dict['reward'][i]
            else:
                rewards_seen_in_patch[i] = features_to_time_series_dict['reward'][i]
    
    features_to_time_series_dict['rewards_at_positions'] = np.array(rewards_at_positions)
    features_to_time_series_dict['reward_attempted_at_positions'] = np.array(reward_attempted_at_positions)
    features_to_time_series_dict['all_dwell_times'] = np.array(all_dwell_times)
    features_to_time_series_dict['rewards_seen_in_patch'] = rewards_seen_in_patch

    return features_to_time_series_dict