## Gym environment for patch foraging task 

### Usage
This repository contains a script that runs a script within a gym environment that models the patch foraging task, ```/code/scripts/train_treadmill_agent.py```. The output of this script is written to a directory in ```/results```.
These results can be visualized within ```/code/notebooks/session_visualization.ipynb```.

Ex: ```python code/scripts/train_treadmill_agent.py --exp_title wsls_test```

The gym itself can be found within ```/code/environments/treadmill_session.py```. See ```/code/scripts/train_treadmill_agent.py``` for demonstration of setup.


### Patch foraging gym

Agents receive a `1 + num_patch_types`-dimensional set of observations. The first observation is the visual cue, which is 1 when the agent is inside a patch and 0 otherwise. The remaining observations are the one hot encoded odor cues, which indicate the agent is within a reward site of a particular patch. Agents navigate by either moving `1` or remaining `0`.


**List of features logged by ```train_treadmill_agent.py```**

| Feature name                         | Type                  | Description          |
|------------------------------------|-----------------------------------|--------------------------------------|
| `agent_in_patch`                  | List of integers (0 or 1)         | Indicates if the agent is in a patch (1) or not (0) |
| `current_patch_start`             | List of integers                  | Start position of current patch     |
| `reward_bounds`                   | List of tuples (int, int)         | Start position and end position of current reward site  |
| `current_patch_num`               | List of integers                  | Identifier for the current patch    |
| `reward_site_idx`                 | List of integers                  | Index of current reward site within patch; -1 if not in patch  |
| `action`                          | List of integers                  | Actions taken by the agent          |
| `current_position`                | List of integers                  | Position of the agent over time     |
| `reward`                          | List of integers (0 or 1)         | Reward received by the agent       |
| `patch_reward_param`              | List of integers                  | Reward parameter of the current patch |
| `current_reward_site_attempted`   | List of integers                  | Binary encoding of whether the agent attempted reward at time step |
| `rewards_at_positions`                | List of integers        | Rewards observed at different positions             |
| `reward_attempted_at_positions`       | List of integers            | Reward attempts made at different positions         |
| `all_dwell_times`                     | List of integers           | Time spent within current reward site                     |
| `rewards_seen_in_patch`               | List of integers                  | Rewards encountered in the current patch           |

