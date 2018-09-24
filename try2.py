# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:40:45 2018

@author: Administrator
"""
import sys
import numpy as np
from sklearn.mixture import GaussianMixture




A = np.asarray([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]) # 转移矩阵
B = np.asarray([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
Pi = np.asarray([0.2, 0.4, 0.4]).transpose()
print(A)



A = np.log(A)
B = np.log(B)
Pi = np.log(Pi)
O = np.asarray([0,1,0])  # 观测序列


print(A)
