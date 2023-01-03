import scipy
import pandas as pd
import torch
import statsmodels
from statsmodels import api as sm_api

# ============================================
df = pd.read_csv(r'./datasets/a.csv', encoding='utf-8', names=['test_col'])
print(df.index)
sub_df = df[df['test_col'] == 1]['test_col']
print(1)
