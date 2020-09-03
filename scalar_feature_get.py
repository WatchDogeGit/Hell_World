#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   scalar_feature_get.py
@Time    :   2020/08/26 16:18:36
@Author  :   A.C. 
@Version :   1.0
'''

import glob
import json
import numpy as np
import random
from config import *

def replay_load(json_path):
    with open(json_path,'r') as f:
        file = f.readlines()
    samples = list(map(lambda x: json.loads(x), file[:-1]))
    return samples

def scalar_feature_select(json_path):
    batch_size = 32                            # batch的长度
    all_obs = replay_load(json_path)            # 样本集
    batch_ind = len(all_obs)//batch_size       # batch的数量
    batch_obs = [] * batch_ind                  # 子batch
    #random.shuffle(all_obs);                    # 乱序的样本集
    #unit_mapping = {0:0, 11:1, 12:2, 13:3, 14:4, 15:5, 19:6, 21:7, 29:8, 31:9, 32:10, 41:11, 42:12, 18:13, 28:14}     # 武器平台类型映射，详定义见最下面
    #wps_id_to_index = {'519':0, '170':1, '360':2}

    for batch_index in range(batch_ind):        # 遍历所有的batch

        batch_obs = all_obs[batch_index * batch_size:batch_index * batch_size + batch_size]

        sim_time = np.zeros((batch_size,1),)

        alive_airport_red = np.zeros((batch_size,1), dtype = int )
        damage_airport_red = np.zeros((batch_size,1), dtype = int )
        num_units_self_red = np.zeros((batch_size,15), dtype = int )        # 己方的units 共13种类型units LX:18 存在但没有定义...LX28 存在但没有定义...
        num_missile_self_red = np.zeros((batch_size,3), dtype = int )       # 共3种导弹 519 170 360
        num_units_qb_red = np.zeros((batch_size,15), dtype = int)           # 观测到的敌方或中立方units LX:18 存在但没有定义...LX28 存在但没有定义...
        num_rockets_opponent_red = np.zeros((batch_size,1), dtype = int)    # 观测到的敌方存活的导弹数量
        damage_commandPost = np.zeros((batch_size,2), dtype= int)           # 蓝方指挥所毁伤情况


        alive_airport_blue = np.zeros((batch_size,1), dtype = int )
        damage_airport_blue = np.zeros((batch_size,1), dtype = int )
        num_units_self_blue = np.zeros((batch_size,15), dtype = int )        # 己方的units 共13种类型units LX:18 存在但没有定义...
        num_missile_self_blue = np.zeros((batch_size,3), dtype = int )       # 共3种导弹 519 170 360
        num_units_qb_blue = np.zeros((batch_size,15), dtype = int)           # 观测到的敌方或中立方units LX:18 存在但没有定义...
        num_rockets_opponent_blue = np.zeros((batch_size,1), dtype = int)    # 观测到的敌方存活的导弹数量

        for index in range(batch_size):                         # 一个batch有batch_size个sample，遍历所有sample
            sim_time[index] = batch_obs[index]['sim_time']      # 仿真时间
            
            # ============== 红方观测量 ==============
            red_obs = batch_obs[index]['red']                   # 红方观测量

            # ------- 机场情况 -------
            alive_airport_red[index] = red_obs['airports'][0]['WH']
            damage_airport_red[index] = red_obs['airports'][0]['DA']

            # ------- 红方已知的在用武器平台 & 不同导弹数量 -------
            for unit in red_obs['units']:

                num_units_self_red[[index],[lx_to_index.get(str(unit.get('LX')))]] += 1
                for item in unit:
                    if(item == 'WP'):
                        for key in unit['WP']:
                            num_missile_self_red[[index],[wps_id_to_index.get(key)]] += unit['WP'][key]
            
            # ------- 红方观测敌方/中立方平台 -------
            for item in red_obs['qb']:
                num_units_qb_red[[index],[lx_to_index.get(str(unit.get('LX')))]] += 1
                if(item['X'] == -129533.05624 and item['LX'] == 41):
                    damage_commandPost[[index],[0]] = item['DA']
                elif(item['X'] == -131156.63859 and item['LX'] == 41):
                    damage_commandPost[[index],[1]] = item['DA']

            # ------- Rocket 数量 -------
            num_rockets_opponent_red[index] = len(red_obs['rockets'])


            # ============== 蓝方观测量 ==============
            blue_obs = batch_obs[index]['blue']                   # 蓝方观测量

            # ------- 机场情况 -------
            alive_airport_blue[index] = blue_obs['airports'][0]['WH']
            damage_airport_blue[index] = blue_obs['airports'][0]['DA']

            # ------- 蓝方已知的在用武器平台 & 不同导弹数量 -------
            for unit in blue_obs['units']:
                num_units_self_blue[[index],[lx_to_index.get(str(unit.get('LX')))]] += 1
                for item in unit:
                    if(item == 'WP'):
                        for key in unit['WP']:
                            num_missile_self_blue[[index],[wps_id_to_index.get(key)]] += unit['WP'][key]
            
            # ------- 蓝方观测敌方/中立方平台 -------
            for item in blue_obs['qb']:
                num_units_qb_blue[[index],[lx_to_index.get(str(item.get('LX')))]] += 1

            # ------- Rocket 数量 -------
            num_rockets_opponent_blue[index] = len(blue_obs['rockets'])
        
        # ============== 在此处输出样本数据至网络 ==============

def main():
    for json_path in glob.glob('./*.json'):
        scalar_feature_select(json_path)

if __name__ == '__main__':
    main()

'''
------- 定义说明 -------
[0]:   unknown
[1]:   歼击机
[2]:   预警机
[3]:   干扰机
[4]:   无人侦察机
[5]:   轰炸机
[6]:   民航
[7]:   护卫舰
[8]:   民船
[9]:   地面防空
[10]:   地面雷达
[11]:   指挥所
[12]:   机场
[13]:   未定义
-------
空空：LX 11 -- 0
空地：LX 15 -- 1
地空：LX 31 -- 2
舰空：LX 21 -- 3

-------
输出：

        sim_time = np.zeros((batch_size,1) )                   # 仿真时间

        alive_airport_red = np.zeros((batch_size,1), dtype = int )          # 机场存活情况 1：存活      2：死亡
        damage_airport_red = np.zeros((batch_size,1), dtype = int )         # 机场损毁程度：0 - 100
        num_units_self_red = np.zeros((batch_size,14), dtype = int )        # 己方的units 共13种类型units  LX:18 存在但没有定义...
        num_missile_self_red = np.zeros((batch_size,4), dtype = int )       # 共4种导弹 0：空空     1：空地     2：地空     3：舰空
        num_units_qb_red = np.zeros((batch_size,14), dtype = int)           # 观测到的敌方或中立方units LX:18 存在但没有定义...
        num_rockets_opponent_red = np.zeros((batch_size,1), dtype = int)    # 观测到的敌方存活的导弹数量

        alive_airport_blue = np.zeros((batch_size,1), dtype = int )         # 机场存活情况 1：存活      2：死亡
        damage_airport_blue = np.zeros((batch_size,1), dtype = int )        # 机场损毁程度：0 - 100
        num_units_self_blue = np.zeros((batch_size,14), dtype = int )       # 己方的units 共13种类型units LX:18 存在但没有定义...
        num_missile_self_blue = np.zeros((batch_size,4), dtype = int )      # 共4种导弹 0：空空     1：空地     2：地空     3：舰空
        num_units_qb_blue = np.zeros((batch_size,14), dtype = int)          # 观测到的敌方或中立方units LX:18 存在但没有定义...
        num_rockets_opponent_blue = np.zeros((batch_size,1), dtype = int)   # 观测到的敌方存活的导弹数量

'''