import pandas as pd
import numpy as np
class TargetEncoder(object):
	def __init__(self,tr_df,te_df,smoothing,min_samples_leaf,noise_level,target='label'):
		self.tr_df=tr_df
		self.te_df=te_df
		self.smoothing=smoothing
		self.min_samples_leaf=min_samples_leaf
		self.noise_level=noise_level
		self.target=target
	def add_noise(self,series):
		return series * (1 + self.noise_level * np.random.randn(len(series)))
	def encode1col(self,col):
		tr_series=self.tr_df[col]
		te_series=self.te_df[col]
		target_series=self.tr_df[self.target]
		temp = pd.concat([tr_series, target_series], axis=1)
		averages = temp.groupby([col]).agg({self.target:["mean", "count"]})
		averages.columns=['mean','count']

		# Compute smoothing
		smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
		# Apply average function to all target data
		prior = target_series.mean()
		# The bigger the count the less full_avg is taken into account
		averages['encode'] = prior * (1 - smoothing) + averages["mean"] * smoothing
		averages.drop(["mean", "count"], axis=1, inplace=True)
		# Apply averages to trn and tst series
		tr_col_encode=pd.merge(self.tr_df,averages.reset_index(),how='left',on=[col])['encode'].rename(col+"_encode").fillna(prior)
		te_col_encode=pd.merge(self.te_df,averages.reset_index(),how='left',on=[col])['encode'].rename(col+"_encode").fillna(prior)

		return np.concatenate( (np.array(self.add_noise(tr_col_encode)),np.array(self.add_noise(te_col_encode))),axis=0).reshape([-1,1])
