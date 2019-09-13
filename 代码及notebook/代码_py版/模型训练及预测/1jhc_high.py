#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import datetime
import math
import gc
import copy
import json
import time
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMRegressor
#optional but advised
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/kaggle/input')
os.listdir()


# In[2]:


#读取各数据集，各数据集的特征 giba_distance_all_cols,basic_qm9_all_cols,bond_all_cols， 方法


# In[3]:


#读取各数据集
'''
#giba_distance
pd.read_csv('get-giba-distance-fea/giba_distance_test.csv.gz',compression='gzip')
#basic_qm9
pd.read_csv('basic-qm9/base_qm9_test.csv.gz',compression='gzip')
#bond
pd.read_csv('get-bond-feas/bond_train.csv.gz',compression='gzip')
'''

#各数据集的特征 giba_distance_all_cols,basic_qm9_all_cols,bond_all_cols

giba_distance_all_cols=['id',
'atom_2',
'atom_3',
'atom_4',
'atom_5',
'atom_6',
'atom_7',
'atom_8',
'atom_9',
'd_1_0',
'd_2_0',
'd_2_1',
'd_3_0',
'd_3_1',
'd_3_2',
'd_4_0',
'd_4_1',
'd_4_2',
'd_4_3',
'd_5_0',
'd_5_1',
'd_5_2',
'd_5_3',
'd_6_0',
'd_6_1',
'd_6_2',
'd_6_3',
'd_7_0',
'd_7_1',
'd_7_2',
'd_7_3',
'd_8_0',
'd_8_1',
'd_8_2',
'd_8_3',
'd_9_0',
'd_9_1',
'd_9_2',
'd_9_3',
'inv_dist0',
'inv_dist1',
'inv_distP',
'linkM0',
'linkM1',
'inv_dist0R',
'inv_dist1R',
'inv_distPR',
'inv_dist0E',
'inv_dist1E',
'inv_distPE',
'min_molecule_atom_0_dist_xyz',
'mean_molecule_atom_0_dist_xyz',
'max_molecule_atom_0_dist_xyz',
'sd_molecule_atom_0_dist_xyz',
'min_molecule_atom_1_dist_xyz',
'mean_molecule_atom_1_dist_xyz',
'max_molecule_atom_1_dist_xyz',
'sd_molecule_atom_1_dist_xyz',
'coulomb_C.x',
'coulomb_F.x',
'coulomb_H.x',
'coulomb_N.x',
'coulomb_O.x',
'yukawa_C.x',
'yukawa_F.x',
'yukawa_H.x',
'yukawa_N.x',
'yukawa_O.x',
'vander_C.x',
'vander_F.x',
'vander_H.x',
'vander_N.x',
'vander_O.x',
'coulomb_C.y',
'coulomb_F.y',
'coulomb_H.y',
'coulomb_N.y',
'coulomb_O.y',
'yukawa_C.y',
'yukawa_F.y',
'yukawa_H.y',
'yukawa_N.y',
'yukawa_O.y',
'vander_C.y',
'vander_F.y',
'vander_H.y',
'vander_N.y',
'vander_O.y',
'distC0',
'distH0',
'distN0',
'distC1',
'distH1',
'distN1',
'adH1',
'adH2',
'adH3',
'adH4',
'adC1',
'adC2',
'adC3',
'adC4',
'adN1',
'adN2',
'adN3',
'adN4',
'NC',
'NH',
'NN',
'NF',
'NO',]

basic_qm9_all_cols=['id',
'rc_A',
'rc_B',
'rc_C',
'mu',
'alpha',
'homo',
'lumo',
'gap',
'r2',
'zpve',
'U0',
'U',
'H',
'G',
'Cv',
'freqs_min',
'freqs_max',
'freqs_mean',
'linear',
'mulliken_min',
'mulliken_max',
'mulliken_mean',
'mulliken_atom_0',
'mulliken_atom_1',
'atom_index_x',
'x_0',
'y_0',
'z_0',
'x_mean_diff_0',
'y_mean_diff_0',
'z_mean_diff_0',
'x_std_diff_0',
'y_std_diff_0',
'z_std_diff_0',
'x_min_diff_0',
'y_min_diff_0',
'z_min_diff_0',
'x_max_diff_0',
'y_max_diff_0',
'z_max_diff_0',
'atom_index_y',
'atom_1',
'x_1',
'y_1',
'z_1',
'ele_neg_1',
'rad_1',
'x_mean_diff_1',
'y_mean_diff_1',
'z_mean_diff_1',
'x_std_diff_1',
'y_std_diff_1',
'z_std_diff_1',
'x_min_diff_1',
'y_min_diff_1',
'z_min_diff_1',
'x_max_diff_1',
'y_max_diff_1',
'z_max_diff_1',
'x_mean',
'y_mean',
'z_mean',
'x_std',
'y_std',
'z_std',
'x_min',
'y_min',
'z_min',
'x_max',
'y_max',
'z_max',
'dist',
'dist_x',
'dist_y',
'dist_z',
'type_0',
'type_1',
'atom_index_closest_0',
'distance_closest_0',
'x_closest_0',
'y_closest_0',
'z_closest_0',
'max_distance_x',
'atom_index_closest_1',
'distance_closest_1',
'x_closest_1',
'y_closest_1',
'z_closest_1',
'max_distance_y',
'atom_index_farthest_0',
'distance_farthest_0',
'x_farthest_0',
'y_farthest_0',
'z_farthest_0',
'min_distance_x',
'atom_index_farthest_1',
'distance_farthest_1',
'x_farthest_1',
'y_farthest_1',
'z_farthest_1',
'min_distance_y',
'distance_center0',
'distance_center1',
'distance_c0',
'distance_c1',
'distance_f0',
'distance_f1',
'cos_c0_c1',
'cos_f0_f1',
'cos_c0_f0',
'cos_c1_f1',
'cos_center0_center1',
'cos_c0',
'cos_c1',
'cos_f0',
'cos_f1',
'cos_center0',
'cos_center1',
'molecule_couples',
'atom_0_couples_count',
'atom_1_couples_count',
'molecule_dist_mean',
'molecule_dist_mean_diff',
'molecule_dist_std',
'molecule_dist_std_diff',
'molecule_dist_mean_min',
'molecule_dist_mean_min_diff',
'molecule_type_dist_max',
'molecule_type_dist_max_diff',
'molecule_type_dist_mean',
'molecule_type0_dist_mean',
'molecule_type1_dist_mean',
'molecule_type_dist_mean_diff',
'molecule_type0_dist_mean_diff',
'molecule_type1_dist_mean_diff',
'molecule_type_dist_std',
'molecule_type0_dist_std',
'molecule_type1_dist_std',
'molecule_type_dist_std_diff',
'molecule_type0_dist_std_diff',
'molecule_type1_dist_std_diff',
'molecule_type_dist_min',
'molecule_type0_dist_min',
'molecule_type1_dist_min',
'molecule_type_dist_min_diff',
'molecule_type0_dist_min_diff',
'molecule_type1_dist_min_diff',
'molecule_type0_dist_max',
'molecule_type1_dist_max',
'molecule_type0_dist_max_diff',
'molecule_type1_dist_max_diff',
'molecule_atom_1_dist_mean',
'molecule_atom_1_dist_mean_diff',
'molecule_atom_1_dist_std',
'molecule_atom_1_dist_std_diff',
'molecule_atom_1_dist_min',
'molecule_atom_1_dist_min_diff',
'molecule_atom_1_dist_max',
'molecule_atom_1_dist_max_diff',
'molecule_atom_index_0_x_1_mean',
'molecule_atom_index_0_x_1_mean_diff',
'molecule_atom_index_0_x_1_max',
'molecule_atom_index_0_x_1_max_diff',
'molecule_atom_index_0_x_1_min',
'molecule_atom_index_0_x_1_min_diff',
'molecule_atom_index_0_x_1_std',
'molecule_atom_index_0_x_1_std_diff',
'molecule_atom_index_0_y_1_mean',
'molecule_atom_index_0_y_1_mean_diff',
'molecule_atom_index_0_y_1_max',
'molecule_atom_index_0_y_1_max_diff',
'molecule_atom_index_0_y_1_min',
'molecule_atom_index_0_y_1_min_diff',
'molecule_atom_index_0_y_1_std',
'molecule_atom_index_0_y_1_std_diff',
'molecule_atom_index_0_z_1_mean',
'molecule_atom_index_0_z_1_mean_diff',
'molecule_atom_index_0_z_1_max',
'molecule_atom_index_0_z_1_max_diff',
'molecule_atom_index_0_z_1_min',
'molecule_atom_index_0_z_1_min_diff',
'molecule_atom_index_0_z_1_std',
'molecule_atom_index_0_z_1_std_diff',
'molecule_atom_index_1_x_0_mean',
'molecule_atom_index_1_x_0_mean_diff',
'molecule_atom_index_1_x_0_max',
'molecule_atom_index_1_x_0_max_diff',
'molecule_atom_index_1_x_0_min',
'molecule_atom_index_1_x_0_min_diff',
'molecule_atom_index_1_x_0_std',
'molecule_atom_index_1_x_0_std_diff',
'molecule_atom_index_1_y_0_mean',
'molecule_atom_index_1_y_0_mean_diff',
'molecule_atom_index_1_y_0_max',
'molecule_atom_index_1_y_0_max_diff',
'molecule_atom_index_1_y_0_min',
'molecule_atom_index_1_y_0_min_diff',
'molecule_atom_index_1_y_0_std',
'molecule_atom_index_1_y_0_std_diff',
'molecule_atom_index_1_z_0_mean',
'molecule_atom_index_1_z_0_mean_diff',
'molecule_atom_index_1_z_0_max',
'molecule_atom_index_1_z_0_max_diff',
'molecule_atom_index_1_z_0_min',
'molecule_atom_index_1_z_0_min_diff',
'molecule_atom_index_1_z_0_std',
'molecule_atom_index_1_z_0_std_diff',
'molecule_atom_index_0_dist_mean',
'molecule_atom_index_0_dist_mean_diff',
'molecule_atom_index_0_dist_std',
'molecule_atom_index_0_dist_std_diff',
'molecule_atom_index_0_dist_min',
'molecule_atom_index_0_dist_min_diff',
'molecule_atom_index_0_dist_max',
'molecule_atom_index_0_dist_max_diff',
'molecule_atom_index_1_dist_mean',
'molecule_atom_index_1_dist_mean_diff',
'molecule_atom_index_1_dist_std',
'molecule_atom_index_1_dist_std_diff',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_1_dist_min_diff',
'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_max_diff',]

bond_all_cols=['id',
'n_bonds_x',
'n_no_bonds_x',
'dist_mean_x',
'dist_median_x',
'dist_std_bond_x',
'dist_mean_bond_x',
'dist_median_bond_x',
'dist_mean_no_bond_x',
'dist_std_no_bond_x',
'dist_median_no_bond_x',
'range_dist_x',
'dist_bond_min_x',
'dist_bond_max_x',
'range_dist_bond_x',
'dist_no_bond_min_x',
'dist_no_bond_max_x',
'range_dist_no_bond_x',
'n_diff_x',
'n_bonds_y',
'n_no_bonds_y',
'dist_median_y',
'dist_std_bond_y',
'dist_mean_bond_y',
'dist_median_bond_y',
'dist_mean_no_bond_y',
'dist_std_no_bond_y',
'dist_median_no_bond_y',
'range_dist_y',
'dist_bond_min_y',
'dist_bond_max_y',
'range_dist_bond_y',
'dist_no_bond_min_y',
'dist_no_bond_max_y',
'range_dist_no_bond_y',
'n_diff_y',
'C',
'F',
'H',
'N',
'O',
'x_dist_abs',
'y_dist_abs',
'z_dist_abs',
'inv_distance3',
'dimension_x',
'dimension_y',
'dimension_z',
'molecule_dist_mean_bond_x',
'molecule_dist_mean_bond_y',
'molecule_dist_range_x',
'molecule_dist_range_y',
'molecule_dist_mean_no_bond_x',
'molecule_dist_mean_no_bond_y',
'molecule_atom_index_0_dist_min_div',
'molecule_atom_index_0_dist_std_div',
'molecule_atom_index_1_dist_min_div',
'molecule_atom_index_1_dist_mean_div',]

def reduce_memory_usage(df, verbose=True):
    '''
    使用pandas进行数据处理，有时文件不大，但用pandas以DataFrame形式加载内存中的时会占用很高的内存
    减少df占用内存所用，为int或float的数据采用合适的数据类型
    pandas读取数据时，会默认用占用内存最大的类型，（eg:对于int，会默认int64，因为它代表的数字范围最大）
    :param df:
    :param verbose:为True时，打印此方法减少了多少内存的占用
    :return:
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #原先df占用的内存，df.memory_usage().sum()单位为B
    #start_momery单位为MB
    start_memory_usage = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        #df[col]它会以为是个df，而不是series,所以用dtypes，而不是dtype，df.dtypes返回series，所以用all()返回series中的值（同一个值）
        col_type = df[col].dtypes#该特征值的类型
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':#为int
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:#为float
               
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_memory_usage = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('memory usage from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(start_memory_usage,end_memory_usage, 100 * (start_memory_usage - end_memory_usage) / start_memory_usage))
    return df

def get_data(t,leixing,top_feas_num,use_train_and_test):
    '''
    t：类型
    leixing: 'zong','fenzong'
    top_feas_num：[]  [2]or[2,3]
    use_train_and_test：bool型，要train和test，还是只要train
    '''
    if leixing=='zong':
        top_feas_num=top_feas_num[0]
        feas_in_diff_types_df_t = pd.read_csv(f'choose-feas-for-different-type/fea_imp_{t}.csv')
        feas_in_diff_types_df_t.sort_values(by=['importance'],axis=0,ascending=False,inplace=True)
        feas_in_diff_types_df_t = feas_in_diff_types_df_t.head(top_feas_num)

        use_cols_gd = np.append(feas_in_diff_types_df_t[feas_in_diff_types_df_t.feature_name.isin(giba_distance_all_cols)].feature_name.values,'id')
        use_cols_bq = np.append(feas_in_diff_types_df_t[feas_in_diff_types_df_t.feature_name.isin(basic_qm9_all_cols)].feature_name.values,'id')
        use_cols_bd = np.append(feas_in_diff_types_df_t[feas_in_diff_types_df_t.feature_name.isin(bond_all_cols)].feature_name.values,'id')

        #train_set
        skip_rows_train_t = np.where(~(train_csv.type==t))[0]+1
        train_t = pd.read_csv('champs-scalar-coupling/train.csv',skiprows=skip_rows_train_t,usecols=['id','scalar_coupling_constant'])
        print('giba_distance')
        gd_train_t = pd.read_csv('get-giba-distance-fea/giba_distance_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_gd)
        gd_train_t = reduce_memory_usage(gd_train_t)
        print('basic_qm9')
        bq_train_t = pd.read_csv('basic-qm9/base_qm9_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_bq)
        bq_train_t = reduce_memory_usage(bq_train_t)
        print('bond')
        bd_train_t = pd.read_csv('get-bond-feas/bond_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_bd)
        bd_train_t = reduce_memory_usage(bd_train_t)

        print('merge')
        train_t = pd.merge(train_t,gd_train_t,how='left',on='id',copy=False)
        train_t = pd.merge(train_t,bq_train_t,how='left',on='id',copy=False)
        train_t = pd.merge(train_t,bd_train_t,how='left',on='id',copy=False)
        
        if use_train_and_test==True:
            #test_set
            skip_rows_test_t = np.where(~(test_csv.type==t))[0]+1
            test_t = pd.read_csv('champs-scalar-coupling/test.csv',skiprows=skip_rows_test_t,usecols=['id'])
            print('giba_distance')
            gd_test_t = pd.read_csv('get-giba-distance-fea/giba_distance_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_gd)
            gd_test_t = reduce_memory_usage(gd_test_t)
            print('basic_qm9')
            bq_test_t = pd.read_csv('basic-qm9/base_qm9_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_bq)
            bq_test_t = reduce_memory_usage(bq_test_t)
            print('bond')
            bd_test_t = pd.read_csv('get-bond-feas/bond_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_bd)
            bd_test_t = reduce_memory_usage(bd_test_t)

            print('merge')
            test_t = pd.merge(test_t,gd_test_t,how='left',on='id',copy=False)
            test_t = pd.merge(test_t,bq_test_t,how='left',on='id',copy=False)
            test_t = pd.merge(test_t,bd_test_t,how='left',on='id',copy=False)
        
            del skip_rows_test_t,gd_test_t,bq_test_t,bd_test_t
            gc.collect()
        del feas_in_diff_types_df_t,use_cols_gd,skip_rows_train_t,gd_train_t,bq_train_t,bd_train_t
        gc.collect()
    else:
        top_feas_giba_distance_num=top_feas_num[0]
        top_feas_basic_qm9_bond_num = top_feas_num[-1]
        
        feas_in_diff_types_gd_df_t = pd.read_csv(f'cal-giba-distance-best-feas/fea_imp_{t}.csv')
        feas_in_diff_types_gd_df_t.sort_values(by=['importance'],axis=0,ascending=False,inplace=True)
        feas_in_diff_types_gd_df_t = feas_in_diff_types_gd_df_t.head(top_feas_giba_distance_num)

        feas_in_diff_types_bqb_df_t = pd.read_csv(f'cal-basic-bond-qm9-best-feas/fea_imp_{t}.csv')
        feas_in_diff_types_bqb_df_t.sort_values(by=['importance'],axis=0,ascending=False,inplace=True)
        feas_in_diff_types_bqb_df_t=feas_in_diff_types_bqb_df_t.head(top_feas_basic_qm9_bond_num)

        #当使用所有的giba——distance特征时，有可能id特征在里面
        if 'id' in feas_in_diff_types_gd_df_t.feature_name.values:
            use_cols_gd = feas_in_diff_types_gd_df_t.feature_name.values
        else:
            use_cols_gd = np.append(feas_in_diff_types_gd_df_t.feature_name.values,'id')
        use_cols_bq = np.append(feas_in_diff_types_bqb_df_t[feas_in_diff_types_bqb_df_t['where']=='basic_qm9'].feature_name.values,'id')
        use_cols_bd = np.append(feas_in_diff_types_bqb_df_t[feas_in_diff_types_bqb_df_t['where']=='bond'].feature_name.values,'id')
        
        #train_set
        skip_rows_train_t = np.where(~(train_csv.type==t))[0]+1
        train_t = pd.read_csv('champs-scalar-coupling/train.csv',skiprows=skip_rows_train_t,usecols=['id','scalar_coupling_constant'])
        print('giba_distance')
        gd_train_t = pd.read_csv('get-giba-distance-fea/giba_distance_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_gd)
        gd_train_t = reduce_memory_usage(gd_train_t)
        print('basic_qm9')
        bq_train_t = pd.read_csv('basic-qm9/base_qm9_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_bq)
        bq_train_t = reduce_memory_usage(bq_train_t)
        print('bond')
        bd_train_t = pd.read_csv('get-bond-feas/bond_train.csv.gz',compression='gzip',skiprows=skip_rows_train_t,usecols=use_cols_bd)
        bd_train_t = reduce_memory_usage(bd_train_t)

        print('merge')
        train_t = pd.merge(train_t,gd_train_t,how='left',on='id',copy=False)
        train_t = pd.merge(train_t,bq_train_t,how='left',on='id',copy=False)
        train_t = pd.merge(train_t,bd_train_t,how='left',on='id',copy=False)

        if use_train_and_test==True:
            
            #test-set
            skip_rows_test_t = np.where(~(test_csv.type==t))[0]+1
            test_t = pd.read_csv('champs-scalar-coupling/test.csv',skiprows=skip_rows_test_t,usecols=['id'])
            print('giba_distance')
            gd_test_t = pd.read_csv('get-giba-distance-fea/giba_distance_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_gd)
            gd_test_t = reduce_memory_usage(gd_test_t)
            print('basic_qm9')
            bq_test_t = pd.read_csv('basic-qm9/base_qm9_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_bq)
            bq_test_t = reduce_memory_usage(bq_test_t)
            print('bond')
            bd_test_t = pd.read_csv('get-bond-feas/bond_test.csv.gz',compression='gzip',skiprows=skip_rows_test_t,usecols=use_cols_bd)
            bd_test_t = reduce_memory_usage(bd_test_t)

            print('merge')
            test_t = pd.merge(test_t,gd_test_t,how='left',on='id',copy=False)
            test_t = pd.merge(test_t,bq_test_t,how='left',on='id',copy=False)
            test_t = pd.merge(test_t,bd_test_t,how='left',on='id',copy=False)
            
            del skip_rows_test_t,gd_test_t,bq_test_t,bd_test_t
            gc.collect()
            
        

        del feas_in_diff_types_gd_df_t,feas_in_diff_types_bqb_df_t,use_cols_gd,skip_rows_train_t,gd_train_t,bq_train_t,bd_train_t
        gc.collect()
    if use_train_and_test==True:
        return train_t,test_t
    else:
        return train_t
        


# * #将train-set和test_set的八个类型变为9个

# In[4]:


#将train-set和test_set的八个类型变为9个
# use atomic numbers to recode atomic names
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 120)

train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
train_csv = pd.read_csv(f'champs-scalar-coupling/train.csv', index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
print('train_csv')
display(train_csv.head(10))

print('Shape: ', train_csv.shape)
print('Total: ', train_csv.memory_usage().sum())
print(train_csv.memory_usage())

submission_csv = pd.read_csv(f'champs-scalar-coupling/sample_submission.csv', index_col='id')
test_csv = pd.read_csv(f'champs-scalar-coupling/test.csv', index_col='id', dtype=train_dtypes)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')
test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]
print('test_csv')
display(test_csv.head(10))

structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
structures_csv = pd.read_csv(f'champs-scalar-coupling/structures.csv', dtype=structures_dtypes)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')
print('structures_csv')
display(structures_csv.head(10))

print('Shape: ', structures_csv.shape)
print('Total: ', structures_csv.memory_usage().sum())
structures_csv.memory_usage()

df = train_csv.copy()
df=pd.merge(df,structures_csv,how='left',left_on=['molecule_index','atom_index_0'],right_on=['molecule_index','atom_index'])
df.rename(columns={'x':'x_0','y':'y_0','z':'z_0'},inplace=True)
df.drop(columns=['atom_index'],inplace=True)
df=pd.merge(df,structures_csv,how='left',left_on=['molecule_index','atom_index_1'],right_on=['molecule_index','atom_index'])
df.rename(columns={'x':'x_1','y':'y_1','z':'z_1'},inplace=True)
df.drop(columns=['atom_index'],inplace=True)

atom_0_array=df[['x_0','y_0','z_0']].values
atom_1_array=df[['x_1','y_1','z_1']].values

df['dist']=np.linalg.norm(atom_0_array-atom_1_array,ord=2,axis=1)
df['type']=df['type'].cat.add_categories('1JHC_high')
df.loc[(df.type=='1JHC')&(df.dist<1.065),'type']='1JHC_high'
train_csv['type']=df['type'].values
print('新增type后的train_csv')
display(train_csv.head())

del df
gc.collect()

df = test_csv.copy()
df=pd.merge(df,structures_csv,how='left',left_on=['molecule_index','atom_index_0'],right_on=['molecule_index','atom_index'])
df.rename(columns={'x':'x_0','y':'y_0','z':'z_0'},inplace=True)
df.drop(columns=['atom_index'],inplace=True)
df=pd.merge(df,structures_csv,how='left',left_on=['molecule_index','atom_index_1'],right_on=['molecule_index','atom_index'])
df.rename(columns={'x':'x_1','y':'y_1','z':'z_1'},inplace=True)
df.drop(columns=['atom_index'],inplace=True)

atom_0_array=df[['x_0','y_0','z_0']].values
atom_1_array=df[['x_1','y_1','z_1']].values

df['dist']=np.linalg.norm(atom_0_array-atom_1_array,ord=2,axis=1)
df['type']=df['type'].cat.add_categories('1JHC_high')
df.loc[(df.type=='1JHC')&(df.dist<1.065),'type']='1JHC_high'
test_csv['type']=df['type'].values
print('新增type后的test_csv')
display(test_csv.head())


del df
gc.collect()


# In[5]:


types=train_csv.type.unique()
submission = submission_csv.copy()
cv_scores = {}
'''
params= {'num_leaves': 255,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          
          'objective': 'regression',
          'max_depth': 9,
 
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.8,
          'reg_lambda': 0.2,
          'colsample_bytree': 1.0
         }
'''
params={
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1455,
    'num_leaves': 129,
    'min_child_samples': 78,
    'max_depth': 11,
    'subsample_freq': 1,
    'subsample': 0.88,
    'bagging_seed': 15,
    'reg_alpha': 0.10107001,
    'reg_lambda': 0.300132,
    'colsample_bytree': 1.0
}


# In[6]:


n_folds=5
n_estimators=10000
verbose=100
early_stopping_rounds=200
leixing='fenzong'
top_feas_num=[50,5]#'zong'，意味着第二个是0，第一个是giba_distance,第二个是basic_qm9_bond
use_train_and_test=True


# In[7]:


types=['1JHC_high']


# In[8]:



for i in range(len(types)):
    t=types[i]
    print(f'{i} training for {t}')
    if use_train_and_test==True:
        cv_score = 0
        train_t,test_t = get_data(t=t,leixing=leixing,top_feas_num=top_feas_num,use_train_and_test=True)
        
        y_data = train_t.pop('scalar_coupling_constant').values
        X_data = train_t.values
        X_test = test_t.values
        y_pred = np.zeros(X_test.shape[0], dtype='float32')
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=124)
        for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
            if fold >= n_folds:
                break
            print(f'{t} Fold {fold} started at {time.ctime()}')
            X_train, X_val = X_data[train_index], X_data[val_index]
            y_train, y_val = y_data[train_index], y_data[val_index]

            model = LGBMRegressor(**params, n_estimators=n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
                verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_val_pred = model.predict(X_val)
            val_score = np.log(mean_absolute_error(y_val, y_val_pred))
            print(f'{t} Fold {fold}, logMAE: {val_score}')

            cv_score += val_score / n_folds
            y_pred += model.predict(X_test) / n_folds
        submission.loc[test_csv['type'] == t, 'scalar_coupling_constant'] = y_pred

        cv_scores[t] = cv_score
        del X_data,y_data,X_test,train_t,test_t
        gc.collect()
    else:
        train_t =  get_data(t=t,leixing=leixing,top_feas_num=top_feas_num,use_train_and_test=False)
        train_t_train=train_t.sample(frac=0.7,random_state=124)
        train_t_val = train_t[~(train_t.index.isin(train_t_train.index))]
        
        train_t_train_y = train_t_train.pop('scalar_coupling_constant')
        train_t_val_y = train_t_val.pop('scalar_coupling_constant')
        
        model = LGBMRegressor(**params, n_estimators=n_estimators, n_jobs = -1)
        model.fit(train_t_train, train_t_train_y, eval_set=[(train_t_train, train_t_train_y), (train_t_val, train_t_val_y)],
              verbose=verbose, early_stopping_rounds=early_stopping_rounds)
        
        train_t_val_pred = model.predict(train_t_val)
        val_score = np.log(mean_absolute_error(train_t_val_y,train_t_val_pred))
        cv_scores[t] = val_score
        del train_t,train_t_train,train_t_train_y,train_t_val,train_t_val_y,model
        gc.collect()


# In[9]:


display(pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())}))
print(np.mean(list(cv_scores.values())))
print(submission[submission['scalar_coupling_constant'] == 0].shape)
display(submission.head(10))


# In[10]:


cv_scores


# In[11]:


today = str(datetime.date.today())
submission.to_csv(f'/kaggle/working/submission_1JHC_high_{today}_{np.mean(list(cv_scores.values())):.3f}.csv')

