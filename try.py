# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:01:40 2018

@author: Administrator
"""

import numpy as np

# 模型参数

'''
        A: N×N 的转移矩阵
        B: N×M 的输出矩阵
        Pi: list, 初始状态概率分布
        O: list, 观测序列
'''
A = np.asarray([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]) # 转移矩阵
B = np.asarray([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
Pi = np.asarray([0.2, 0.4, 0.4]).transpose()

O = np.asarray([0,1,0])
T = O.shape[0]
#print(T)

N = A.shape[0]   # 状态数

p_nodes = Pi * B[:, O[0]]     # 记录每个节点的路径概率
#print(p_nodes)
path_nodes = list()           # 记录每个节点的路径
# 计初始化路径
for node in range(N):
    path_nodes.append([node])
#print(path_nodes)
# T 个时刻
for step in range(1, T):
    for this_node in range(N):   # 计算每个节点的新概率
        p_news = list()
        for last_node in range(N):
            p_trans = A[last_node, this_node]  # 转移概率
            p_out = B[this_node, O[step]]       # 输出概率
            print(O[step])
            print(p_out)
            print("------------")
            p_new = p_nodes[last_node] * p_trans * p_out
            #print(p_nodes[last_node])
            #print(p_new)
            #print("----")
            p_news.append(p_new)
        p_nodes[this_node] = np.max(p_news)    # 更新节点路径概率
        last_index = np.argmax(p_news)         # 更新节点路径
        #print(last_index)
        temp = path_nodes[last_index][:]
        #print(temp)
        temp.append(this_node)
        path_nodes[this_node] = temp
                
'''
print(p_nodes)     # 最有一步每个节点的概率
print(path_nodes) 
 
max_index = np.argmax(p_nodes)
max_path = path_nodes[max_index]
print(max_index)
print(max_path)   # 最优路径
print(type(max_path))

state_seq = np.empty((T,), dtype='int')
state_seq[T-1] = 3 - 1

newstate=np.array(max_path)
print(newstate)
print(type(newstate))
'''







'''
 a=np.array([[99,2,4,7],[9,88,6,45],[9,76,3,100]])
 array([[ 99,   2,   4,   7],
       [  9,  88,   6,  45],
       [  9,  76,   3, 100]])
    
    np.argmax(a) :11
    np.argmax(a,axis=0):array([0, 1, 1, 2], dtype=int64) axis=0执行列操作
    np.argmax(a,axis=1)：array([0, 1, 3], dtype=int64) axis=1执行行操作



'''