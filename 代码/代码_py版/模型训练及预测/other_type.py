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
import time
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
import os
os.chdir('/kaggle/input')


# In[2]:


def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
    return base, structures

def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df

def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df

def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
    return df

def add_center(df):#得到原子对的中心点坐标
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

def add_distance_to_center(df):#得到其他原子到该原子对中心点的距离
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) + 
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))

def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))
    
def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)
            
def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True) 

def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):
    #base 列名 ['id', 'molecule_index', 'atom_index_0', 'atom_index_1','scalar_coupling_constant']
    #structures 列名 ['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z'] atom为原子序数
    base, structures = build_type_dataframes(some_csv, structures_csv, coupling_type)
    
    base = add_coordinates(base, structures, 0)
    #base 列名 ['id', 'molecule_index', 'atom_index_0', 'atom_index_1','scalar_coupling_constant', 'atom_0', 'x_0', 'y_0', 'z_0', 'atom_1','x_1', 'y_1', 'z_1']
    base = add_coordinates(base, structures, 1)
    
    #base 列名 ['id', 'molecule_index', 'atom_index_0', 'atom_index_1','scalar_coupling_constant',  'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1']
    base = base.drop(['atom_0', 'atom_1'], axis=1)
    #atoms 列名 [ 'molecule_index', 'atom_index_0', 'atom_index_1','scalar_coupling_constant', 'x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1']
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)
    #atoms 列名 [ 'molecule_index', 'atom_index_0', 'atom_index_1' 'x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1']

    
    #为原子对添加中心点坐标
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1', 'x_c', 'y_c', 'z_c']
    add_center(atoms)
    
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'x_c', 'y_c', 'z_c']
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    #添加分子中的其他原子（除了该原子对的原子）
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'x_c', 'y_c', 'z_c','atom_index', 'atom', 'x', 'y', 'z']
    atoms = merge_all_atoms(atoms, structures)
    
    #d_c分子中其他原子到该原子对中心点的距离
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'x_c', 'y_c', 'z_c','atom_index', 'atom', 'x', 'y', 'z', 'd_c']
    add_distance_to_center(atoms)
    
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'atom', 'x', 'y', 'z', 'd_c']
    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    #对距原子对中心点的距离排序
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    
    #num为核心原子下标，原子对分别为0,1，其他原子按距离该原子对的距离排序，最近的为2，依次向下。
    atoms['num'] = atom_groups.cumcount() + 2
    #atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'atom', 'x', 'y', 'z','num']    
    atoms = atoms.drop(['d_c'], axis=1)
    #只保留距离该原子对中心点最近的n_atoms（除该原子对，最近的下标为2）个原子
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    
    '''
        atoms 列名 ['molecule_index', 'atom_index_0', 'atom_index_1', 'atom_2', 'atom_3',
       'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'x_2',
       'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'y_2', 'y_3', 'y_4',
       'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'z_2', 'z_3', 'z_4', 'z_5', 'z_6',
       'z_7', 'z_8', 'z_9'] 除了该原子对的两个原子外，距离该原子对中心点最近的atom,及其坐标x,y,z
    '''
    atoms = atoms.reset_index()
    
    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            #此时要求除了该原子对的原子，还要有8个原子，对于原子数少的分子，不够8个的就将其原子序数设为0
            atoms[col] = atoms[col].fillna(0).astype('int8')
            
    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')
    
    '''
    ['id', 'molecule_index', 'atom_index_0', 'atom_index_1',
       'scalar_coupling_constant', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1',
       'z_1', 'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7',
       'atom_8', 'atom_9', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7',
       'x_8', 'x_9', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8',
       'y_9', 'z_2', 'z_3', 'z_4', 'z_5', 'z_6', 'z_7', 'z_8', 'z_9']
    '''
    full = add_atoms(base, atoms)
    
    '''
    ['id', 'molecule_index', 'atom_index_0', 'atom_index_1',
       'scalar_coupling_constant', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1',
       'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8',
       'atom_9', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'y_2',
       'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'z_2', 'z_3', 'z_4',
       'z_5', 'z_6', 'z_7', 'z_8', 'z_9', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
       'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1',
       'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
       'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1',
       'd_9_2', 'd_9_3']
    '''
    add_distances(full)
    
    full.sort_values('id', inplace=True)
    
    return full

def take_n_atoms(df, n_atoms, four_start=4):
    labels = ['id']
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    return df[labels]
def build_x_y_data_mullian(some_csv, coupling_type, n_atoms):
    '''
    full 列名
        ['id', 'molecule_index', 'atom_index_0', 'atom_index_1',
       'scalar_coupling_constant', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1',
       'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8',
       'atom_9', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'y_2',
       'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'z_2', 'z_3', 'z_4',
       'z_5', 'z_6', 'z_7', 'z_8', 'z_9', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
       'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1',
       'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
       'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1',
       'd_9_2', 'd_9_3']
    '''
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)
    '''
    df 列名 ['id', atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8',
       'atom_9', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
       'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1',
       'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
       'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1',
       'd_9_2', 'd_9_3']
    '''
    df = take_n_atoms(full, n_atoms)
    
    df = df.fillna(0)
    print('不算giba和id特征，其余为:',df.columns)
    

    
    if 'scalar_coupling_constant' in df:#是train-set
        some_df = pd.read_csv('/kaggle/input/champs-scalar-coupling/train.csv')
        some_type_index = some_df['type']==coupling_type
        rows_to_exclude = np.where(some_type_index==False)[0]+1 # retain the header row
        giba_some_df = pd.read_csv(f'/kaggle/input/giba-rr/train.csv.gz',compression='gzip', skiprows=rows_to_exclude,usecols=giba_cols)
        qm9_df = pd.read_csv('/kaggle/input/basic-qm9/base_qm9_train.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=qm9_cols)
        df = pd.merge(df,giba_some_df,on=['id'],how='left',copy=False)
        giba_distance_col_num = df.shape[1]
        df = pd.merge(df,qm9_df,on=['id'],how='left',copy=False)
        #df = df.fillna(0)
        
        
        y_data = df.pop('scalar_coupling_constant').values.astype('float32')
        X_data = df.drop(columns=['id']).values.astype('float32')
        
    else:
        some_df = pd.read_csv('/kaggle/input/champs-scalar-coupling/test.csv')
        some_type_index = some_df['type']==coupling_type
        rows_to_exclude = np.where(some_type_index==False)[0]+1 # retain the header row
        
        giba_some_df = pd.read_csv(f'/kaggle/input/giba-rr/test.csv.gz',compression='gzip', skiprows=rows_to_exclude,usecols=giba_cols)
        qm9_df = pd.read_csv('/kaggle/input/basic-qm9/base_qm9_test.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=qm9_cols)
        df = pd.merge(df,giba_some_df,on=['id'],how='left',copy=False)
        
        giba_distance_col_num = df.shape[1]
        
        df = pd.merge(df,qm9_df,on=['id'],how='left',copy=False)
          
        #df = df.fillna(0)
        X_data = df.drop(columns=['id']).values.astype('float32')
        
        y_data = None
    
    print('特征数: giba_distance ',giba_distance_col_num,'最终：',df.shape[1])
    del some_df,giba_some_df,df,full,qm9_df
    gc.collect()
    
  
   
    return X_data, y_data
def build_x_y_data(some_csv, coupling_type, n_atoms,top_num):
    '''
    full 列名
        ['id', 'molecule_index', 'atom_index_0', 'atom_index_1',
       'scalar_coupling_constant', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1',
       'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8',
       'atom_9', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'y_2',
       'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'z_2', 'z_3', 'z_4',
       'z_5', 'z_6', 'z_7', 'z_8', 'z_9', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
       'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1',
       'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
       'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1',
       'd_9_2', 'd_9_3']
    '''
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)
    '''
    df 列名 ['id', atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8',
       'atom_9', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
       'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1',
       'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
       'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1',
       'd_9_2', 'd_9_3']
    '''
    df = take_n_atoms(full, n_atoms)
    
    df = df.fillna(0)
    print('不算giba和id特征，其余为:',df.columns)
    

    top_feas_basic_qm9_bond_num=top_num
    feas_in_diff_types_bqb_df_t = pd.read_csv(f'/kaggle/input/cal-basic-bond-qm9-best-feas/fea_imp_{coupling_type}.csv')
    feas_in_diff_types_bqb_df_t.sort_values(by=['importance'],axis=0,ascending=False,inplace=True)
    feas_in_diff_types_bqb_df_t=feas_in_diff_types_bqb_df_t.head(top_feas_basic_qm9_bond_num)


    use_cols_bq = np.append(feas_in_diff_types_bqb_df_t[feas_in_diff_types_bqb_df_t['where']=='basic_qm9'].feature_name.values,'id')
    use_cols_bd = np.append(feas_in_diff_types_bqb_df_t[feas_in_diff_types_bqb_df_t['where']=='bond'].feature_name.values,'id')
    if 'scalar_coupling_constant' in df:#是train-set
        some_df = pd.read_csv('/kaggle/input/champs-scalar-coupling/train.csv')
        some_type_index = some_df['type']==coupling_type
        rows_to_exclude = np.where(some_type_index==False)[0]+1 # retain the header row
        giba_some_df = pd.read_csv(f'/kaggle/input/giba-rr/train.csv.gz',compression='gzip', skiprows=rows_to_exclude,usecols=giba_cols)
        basic_qm9_df = pd.read_csv('/kaggle/input/basic-qm9/base_qm9_train.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=use_cols_bq)
        bond_df = pd.read_csv('/kaggle/input/get-bond-feas/bond_train.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=use_cols_bd)

        df = pd.merge(df,giba_some_df,on=['id'],how='left',copy=False)
        
        giba_distance_col_num = df.shape[1]
        
        df = pd.merge(df,basic_qm9_df,on=['id'],how='left',copy=False)
        df = pd.merge(df,bond_df,on=['id'],how='left',copy=False)
        #df = df.fillna(0)


        y_data = df.pop('scalar_coupling_constant').values.astype('float32')
        X_data = df.drop(columns=['id']).values.astype('float32')

    else:
        some_df = pd.read_csv('/kaggle/input/champs-scalar-coupling/test.csv')
        some_type_index = some_df['type']==coupling_type
        rows_to_exclude = np.where(some_type_index==False)[0]+1 # retain the header row

        giba_some_df = pd.read_csv(f'/kaggle/input/giba-rr/test.csv.gz',compression='gzip', skiprows=rows_to_exclude,usecols=giba_cols)
        basic_qm9_df = pd.read_csv('/kaggle/input/basic-qm9/base_qm9_test.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=use_cols_bq)
        bond_df = pd.read_csv('/kaggle/input/get-bond-feas/bond_test.csv.gz',compression='gzip',skiprows=rows_to_exclude,usecols=use_cols_bd)

        df = pd.merge(df,giba_some_df,on=['id'],how='left',copy=False)
        
        giba_distance_col_num = df.shape[1]
        
        df = pd.merge(df,basic_qm9_df,on=['id'],how='left',copy=False)
        df = pd.merge(df,bond_df,on=['id'],how='left',copy=False)

        #df = df.fillna(0)
        X_data = df.drop(columns=['id']).values.astype('float32')

        y_data = None
    
    print('特征数: giba_distance ',giba_distance_col_num,'最终：',df.shape[1])
    del some_df,giba_some_df,df,full,basic_qm9_df,bond_df
    gc.collect()
    
  
   
    return X_data, y_data

def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms, n_folds=5, n_splits=5, random_state=128,top_num=0):
    print(f'*** Training Model for {coupling_type} ***')
    if top_num==0:
        X_data, y_data = build_x_y_data_mullian(train_csv, coupling_type, n_atoms)
        X_test, _ = build_x_y_data_mullian(test_csv, coupling_type, n_atoms)
    
    else:
        X_data, y_data = build_x_y_data(train_csv, coupling_type, n_atoms,top_num)
        X_test, _ = build_x_y_data(test_csv, coupling_type, n_atoms,top_num)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    cv_score = 0
    
    if n_folds > n_splits:
        n_splits = n_folds
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    
    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_folds:
            break
        print(f'Fold {fold + 1} started at {time.ctime()}')
        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(**LGB_PARAMS, n_estimators=7000, n_jobs = -1)
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
            verbose=100, early_stopping_rounds=200)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')
        
        cv_score += val_score / n_folds
        y_pred += model.predict(X_test) / n_folds
        
        
    submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred
    return cv_score


# In[3]:


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


# In[4]:


train_csv.type.unique()


# In[5]:


giba_cols= ['id',
    'inv_dist0',
 'inv_dist1',
 'inv_distP',
 'inv_dist0R',
 'inv_dist1R',
 'inv_distPR',
 'inv_dist0E',
 'inv_dist1E',
 'inv_distPE',
 'linkM0',
 'linkM1',
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
 'NO']
qm9_cols=['id',
'mulliken_min',
'mulliken_max',
'mulliken_mean',
'mulliken_atom_0',
'mulliken_atom_1']
'''
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}




LGB_PARAMS = {
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

'''
LGB_PARAMS = {'num_leaves': 255,
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
type_num={
    '1JHN':0,
    '1JHC':5,
    '2JHH':15,
    '2JHN':5,
    '2JHC':10,
    '3JHH':8,
    '3JHC':15,
    '3JHN':8
}
model_params = {
    #'1JHN': 7,
    #'1JHC': 10,
    #'2JHH': 9,
    #'2JHN': 9,
    #'2JHC': 9,
    #'3JHH': 9,
    '3JHC': 10,
    #'3JHN': 10,
    #'1JHC_high': 10 由其他模型预测
}
N_FOLDS = 5
submission = submission_csv.copy()

cv_scores = {}
for coupling_type in model_params.keys():
    cv_score = train_and_predict_for_one_coupling_type(
        coupling_type, submission, n_atoms=model_params[coupling_type], n_folds=N_FOLDS,top_num=type_num[coupling_type])
    cv_scores[coupling_type] = cv_score


# In[6]:


display(pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())}))
print(np.mean(list(cv_scores.values())))
print(submission[submission['scalar_coupling_constant'] == 0].shape)
display(submission.head(10))


# In[7]:


cv_scores


# In[8]:


today = str(datetime.date.today())
submission.to_csv(f'/kaggle/working/submission_3JHC_{today}_{np.mean(list(cv_scores.values())):.3f}.csv')

