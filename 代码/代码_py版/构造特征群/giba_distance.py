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
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
#optional but advised
import warnings
warnings.filterwarnings('ignore')
os.chdir('/kaggle/input')
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


# # 前面已经有8个类型变成9个类型了

# In[2]:


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
def build_type_dataframes(base, structures):
    #base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
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

def build_couple_dataframe(some_csv, structures_csv, n_atoms=10):
    #base 列名 ['id', 'molecule_index', 'atom_index_0', 'atom_index_1','scalar_coupling_constant']
    #structures 列名 ['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z'] atom为原子序数
    base, structures = build_type_dataframes(some_csv, structures_csv)
    
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

def build_x_y_data(some_csv, n_atoms):
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
    full = build_couple_dataframe(some_csv, structures_csv, n_atoms=n_atoms)
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
    df=reduce_memory_usage(df)
    print('不算giba和id特征，其余为:',df.columns)
    

    
    if 'scalar_coupling_constant' in df:#是train-set
        df.drop(columns=['scalar_coupling_constant'],inplace=True)
        

    giba_some_df = pd.read_csv(f'giba-rr/train.csv.gz',compression='gzip')
    giba_some_df = reduce_memory_usage(giba_some_df)

    df = pd.merge(df,giba_some_df,on=['id'],how='left')
    print('特征数:',df.shape[1])
    
    del giba_some_df,full
    gc.collect()
    
  
    return df


# In[3]:


test_df = build_x_y_data(test_csv, 10)
train_df = build_x_y_data(train_csv, 10)


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


print(train_df.shape)
print(test_df.shape)
for i in train_df.columns:
    print(i)


# In[7]:


train_df.drop(columns=['molecule_name','atom_index_1','atom_index_0','scalar_coupling_constant'],inplace=True)
test_df.drop(columns=['molecule_name','atom_index_1','atom_index_0','scalar_coupling_constant'],inplace=True)
print(train_df.shape)
print(test_df.shape)


# 无molecule_name，atom_index_1，atom_index_0，scalar_coupling_constant
# 有：
# id
# atom_2
# atom_3
# atom_4
# atom_5
# atom_6
# atom_7
# atom_8
# atom_9
# d_1_0
# d_2_0
# d_2_1
# d_3_0
# d_3_1
# d_3_2
# d_4_0
# d_4_1
# d_4_2
# d_4_3
# d_5_0
# d_5_1
# d_5_2
# d_5_3
# d_6_0
# d_6_1
# d_6_2
# d_6_3
# d_7_0
# d_7_1
# d_7_2
# d_7_3
# d_8_0
# d_8_1
# d_8_2
# d_8_3
# d_9_0
# d_9_1
# d_9_2
# d_9_3
# inv_dist0
# inv_dist1
# inv_distP
# linkM0
# linkM1
# inv_dist0R
# inv_dist1R
# inv_distPR
# inv_dist0E
# inv_dist1E
# inv_distPE
# min_molecule_atom_0_dist_xyz
# mean_molecule_atom_0_dist_xyz
# max_molecule_atom_0_dist_xyz
# sd_molecule_atom_0_dist_xyz
# min_molecule_atom_1_dist_xyz
# mean_molecule_atom_1_dist_xyz
# max_molecule_atom_1_dist_xyz
# sd_molecule_atom_1_dist_xyz
# coulomb_C.x
# coulomb_F.x
# coulomb_H.x
# coulomb_N.x
# coulomb_O.x
# yukawa_C.x
# yukawa_F.x
# yukawa_H.x
# yukawa_N.x
# yukawa_O.x
# vander_C.x
# vander_F.x
# vander_H.x
# vander_N.x
# vander_O.x
# coulomb_C.y
# coulomb_F.y
# coulomb_H.y
# coulomb_N.y
# coulomb_O.y
# yukawa_C.y
# yukawa_F.y
# yukawa_H.y
# yukawa_N.y
# yukawa_O.y
# vander_C.y
# vander_F.y
# vander_H.y
# vander_N.y
# vander_O.y
# distC0
# distH0
# distN0
# distC1
# distH1
# distN1
# adH1
# adH2
# adH3
# adH4
# adC1
# adC2
# adC3
# adC4
# adN1
# adN2
# adN3
# adN4
# NC
# NH
# NN
# NF
# NO
# 
# 
# 
# 
# 
# 

# In[8]:


print(train_df.memory_usage().sum()/1024**2)
print(test_df.memory_usage().sum()/1024**2)


# In[9]:


train_df.to_csv('/kaggle/working/giba_distance_train.csv.gz',compression='gzip',index=False)
test_df.to_csv('/kaggle/working/giba_distance_test.csv.gz',compression='gzip',index=False)


# In[ ]:




