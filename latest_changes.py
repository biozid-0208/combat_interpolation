# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:57:50 2024

@author: biozid
"""


import numpy as np
import pandas as pd
from scipy.linalg import solve
from statsmodels.regression.linear_model import OLS
import statsmodels as sm

p = 3
n = 10
batch = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

dat = np.array([[1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
                [np.nan, 12, 13, 14, 15, 16, 17, 18, 19, np.nan],
                [21, np.nan, 23, 24, 25, 26, 27, 28, 29, np.nan]])

age = np.array([82, 70, 68, 66, 80, 69, 72, 76, 74, 80])
disease = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
mod = np.column_stack([age, disease])


if np.isnan(dat).any():
        for batch_i in sorted(np.unique(batch)):
            i = np.where(batch == batch_i)[0]
            dat_i = dat[:, i]

            if np.isnan(dat_i).any():
                for j in range(dat.shape[0]):
                    dat_ji = dat_i[j, :]

                    is_na = np.where(np.isnan(dat_ji))[0]

                    if len(is_na) > 0 and len(is_na) < len(i):
                        if len(is_na) == 1:
                            mod_i_is_na = mod[i[is_na], :].reshape(1, -1)
                        else:
                            mod_i_is_na = mod[i[is_na], :]
                        
                        lm_model =  OLS(dat_ji.T, sm.tools.tools.add_constant(mod[i, :]),missing="drop").fit()
                        beta = lm_model.params
                        # beta = np.linalg.lstsq(mod[i, :], dat_ji, rcond=None)[0]
                        beta[np.isnan(beta)] = 0
                        print(beta)

                        dat[j, i[is_na]] = np.dot(np.hstack([np.ones((len(is_na), 1)), mod_i_is_na]), beta)
                    else:
                        dat[j, i[is_na]] = np.mean(dat_ji[~np.isnan(dat_ji)])
    


for batch_i in sorted(np.unique(batch)):
    i = np.where(batch == batch_i)[0]

    if np.isnan(dat[:, i]).any():
        for j in range(dat.shape[0]):
            dat_j = dat[j, :]

            if np.isnan(dat_j[i[0]]):
                if mod is not None:
                    lm_model =  OLS(dat_j.T, sm.tools.tools.add_constant(mod),missing="drop").fit()
                    beta = lm_model.params
                    beta[np.isnan(beta)] = 0

                    dat[j, i] = np.dot(np.hstack([np.ones((len(i), 1)), mod[i, :]]), beta)
                else:
                    dat[j, i] = np.mean(dat_j[~np.isnan(dat_j)])


