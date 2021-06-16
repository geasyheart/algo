# -*- coding: utf8 -*-

import numpy as np
import pandas as pd

from viterbi import viterbi


states = ['sleeping', 'eating', 'pooping']
pi = [0.35, 0.35, 0.3]
state_space = pd.Series(pi, index=states, name='states')

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.4, 0.2, 0.4]
q_df.loc[states[1]] = [0.45, 0.45, 0.1]
q_df.loc[states[2]] = [0.45, 0.25, .3]

q = q_df.values

hidden_states = ['healthy', 'sick']
pi = np.array([0.5, 0.5])
state_space = pd.Series(pi, index=hidden_states, name='states')

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

a = a_df.values

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

b = b_df.values

obs_map = {'sleeping': 0, 'eating': 1, 'pooping': 2}
obs = np.array([1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1])
inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)


state_map = {0:'healthy', 1:'sick'}
state_path = [state_map[v] for v in path]

print(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))