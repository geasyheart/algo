import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd
ts_split = TimeSeriesSplit(n_splits=5)

df =pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'y':  np.arange(100)
})
for train_idx, val_idx in ts_split.split(X=df.drop(columns='y'), y=df['y']):
    print(f'{train_idx=}')
    print(f'[{len(val_idx)}] {val_idx=}')
