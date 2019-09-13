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
#optional but advised
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('/kaggle/input')
print(os.listdir())

# Any results you write to the current directory are saved as output.


# * 不要：
# 'molecule_name',
# 'atom_index_0',
# 'atom_index_1',
# 'type',
# 'scalar_coupling_constant', test_set的都为Nan, train_set的有值
# * 要：
# * 'id',
# * 'rc_A',分子转动的转动常数
# * 'rc_B',分子转动的转动常数
# * 'rc_C',分子转动的转动常数
# * 'mu',分子的偶极矩，dipole_moments.csv contains X,Y,Z values per molecule and sqrt(X^2+Y^2+Z^2)=mu
# * 'alpha',分子极化率
# * 'homo',分子中已占有分子的能级最高轨道的能级
# * 'lumo',分子中未占有电子的能级最低轨道的能级
# * 'gap',lumo-home，光环境下是否稳定（gap越小，分子越容易激发）
# * 'r2',分子的空间范围
# * 'zpve',分子在零点（绝对零度，0k）的振动能
# * 'U0',分子在0k时的内能
# * 'U',分子在298.15k(25℃）的内能 U=电子能量+振动能量+转动能量
# * 'H',分子在298.15k(25℃）的焓（H=U+PV）p为压强，v为体积
# * 'G',分子在298.15k(25℃）的自由能（吉布斯），（G=H-TS）t温度，s熵，特征的热力学过程中，系统减少的内能可转化为对外做功的部分
# * 'Cv',分子在298.15k(25℃）的热容量
# * 每个自由度下都有自己的振动频率
# * 'freqs_min',分子中所有自由度下最小的振动频率
# * 'freqs_max',分子中所有自由度下最大的振动频率
# * 'freqs_mean',分子中所有自由度下平均的振动频率
# * 'linear',分子是否为线性
# * 'mulliken_min',分子中原子最小的mulliken charge
# * 'mulliken_max',分子中原子最大的mulliken charge
# * 'mulliken_mean',分子中原子的平均mulliken charge
# * 'mulliken_atom_0',分子中的原子对的第一个原子的mulliken charge
# * 'mulliken_atom_1'分子中的原子对的第一个原子的mulliken charge
# 
# 

# In[2]:


#返回结果越小越好，loss
def group_mean_log_mae(y_true_series, y_pred_array, groups, floor=1e-9):
    '''
    y_true_series = test_df['scalar_coupling_constant']
    groups = train_df['type'] or train_df['type'].values
    y_pred：array 
    
    Series1-array2=Series3
    
    groupby(此时用Series或array来给索引分组)
    '''
    #获取每个类型的平均误差绝对值
    maes = (y_true_series-y_pred_array).abs().groupby(groups,axis=0).mean()#
    #y_true_series-y_pred_array得到Series
    #返回：log(所有类型的平均误差绝对值)
    return np.log(maes.map(lambda x: max(x, floor))).mean()
def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='group_mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=1000, early_stopping_rounds=200, n_estimators=10000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - k-折切分器
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    }

    result_dict = {}
    
    # out-of-fold predictions on train data，通过k折验证，每次用一部分作为验证集，验证集加起来时整个训练集，oof为折外预测得到的整个训练集的预测值
    oof = np.zeros(len(X))
    
    # averaged predictions on train data，通过k折验证，每次用一部分作训练集，以此得到test_data的预测结果，加起来，然后再平均（/折数）
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    best_iterations=[]
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            #eval_set=[(X_train, y_train), (X_valid, y_valid)],来看是否过拟合
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

           
            best_iterations.append(model.best_iteration_)

        if eval_metric == 'group_mae':
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))


        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            
    oof[valid_index] = y_pred_valid.reshape(-1,)
    prediction /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof#k折交叉验证时，
    result_dict['prediction'] = prediction#k折，每次得到测试集的结果，然后平均
    result_dict['scores'] = scores

    result_dict['best_iterations']=best_iterations
   
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance
            result_dict['good_features']=cols
    return result_dict

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

#添加组成原子对的原子所组成的原子对中的最近最远距离
def add_closest_farthest_atom(df):
    
    df_temp = df[["molecule_name",
                  "atom_index_0", "atom_index_1",
                  "dist", "x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                        'atom_index_1': 'atom_index_0',
                                        'x_0': 'x_1',
                                        'y_0': 'y_1',
                                        'z_0': 'z_1',
                                        'x_1': 'x_0',
                                        'y_1': 'y_0',
                                        'z_1': 'z_0'})
    df_temp_all = pd.concat((df_temp, df_temp_), axis=0)#此时atom_index_0为分子的原子对中原子的所有索引
    df_temp_all["min_distance"] = df_temp_all.groupby(['molecule_name',
                                                       'atom_index_0'])['dist'].transform('min')
    df_temp_all["max_distance"] = df_temp_all.groupby(['molecule_name',
                                                       'atom_index_0'])['dist'].transform('max')

    df_temp = df_temp_all[df_temp_all["min_distance"] == df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(columns=['x_0', 'y_0', 'z_0', 'min_distance'])
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                      'atom_index_1': 'atom_index_closest',
                                      'dist': 'distance_closest',
                                      'x_1': 'x_closest',
                                      'y_1': 'y_closest',
                                      'z_1': 'z_closest'})
    #只保留分子中每个原子(在分子中的索引）的最近距离原子对，eg:CH4,对于C，有4个距离最小的原子对
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_temp, atom_idx)
        #atom_idx=0时
        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',#距离原子对的一个原子的最近的原子索引
                                'distance_closest': f'distance_closest_{atom_idx}',#身为原子对的第一个原子时，构造的距离最近原子对的距离
                                'x_closest': f'x_closest_{atom_idx}',#身为原子对的第一个原子时，构造的距离最近原子对的另一个原子的x坐标
                                'y_closest': f'y_closest_{atom_idx}',
                                'z_closest': f'z_closest_{atom_idx}'})

    df_temp = df_temp_all[df_temp_all["max_distance"] == df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'max_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                      'atom_index_1': 'atom_index_farthest',
                                      'dist': 'distance_farthest',
                                      'x_1': 'x_farthest',
                                      'y_1': 'y_farthest',
                                      'z_1': 'z_farthest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_temp, atom_idx)
        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',
                                'distance_farthest': f'distance_farthest_{atom_idx}',
                                'x_farthest': f'x_farthest_{atom_idx}',
                                'y_farthest': f'y_farthest_{atom_idx}',
                                'z_farthest': f'z_farthest_{atom_idx}'})
    df = reduce_memory_usage(df)
    return df

def change_structures_df(structures_df):
    #添加原子半径和原子的电负性
    atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor
    '''
    #为什么加上经验系数0.05?
    fudge_factor = 0.05
    atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
    '''

    atom_electron_negativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}


    atom_types = structures_df['atom'].values
    atom_ele_negs = [atom_electron_negativity[x] for x in atom_types]
    atom_rads = [atomic_radius[x] for x in atom_types]

    structures_df['ele_neg'] = atom_ele_negs
    structures_df['rad'] = atom_rads

    structures_df['x_mean']=structures_df.groupby(by='molecule_name')['x'].transform('mean')
    structures_df['y_mean']=structures_df.groupby(by='molecule_name')['y'].transform('mean')
    structures_df['z_mean']=structures_df.groupby(by='molecule_name')['z'].transform('mean')
    structures_df['x_std']=structures_df.groupby(by='molecule_name')['x'].transform('std')
    structures_df['y_std']=structures_df.groupby(by='molecule_name')['y'].transform('std')
    structures_df['z_std']=structures_df.groupby(by='molecule_name')['z'].transform('std')
    structures_df['x_min']=structures_df.groupby(by='molecule_name')['x'].transform('min')
    structures_df['y_min']=structures_df.groupby(by='molecule_name')['y'].transform('min')
    structures_df['z_min']=structures_df.groupby(by='molecule_name')['z'].transform('min')
    structures_df['x_max']=structures_df.groupby(by='molecule_name')['x'].transform('max')
    structures_df['y_max']=structures_df.groupby(by='molecule_name')['y'].transform('max')
    structures_df['z_max']=structures_df.groupby(by='molecule_name')['z'].transform('max')
    
    
    structures_df['x_mean_diff']=structures_df['x_mean']-structures_df['x']
    structures_df['y_mean_diff']=structures_df['y_mean']-structures_df['y']
    structures_df['z_mean_diff']=structures_df['z_mean']-structures_df['z']
    structures_df['x_std_diff']=structures_df['x_std']-structures_df['x']
    structures_df['y_std_diff']=structures_df['y_std']-structures_df['y']
    structures_df['z_std_diff']=structures_df['z_std']-structures_df['z']
    structures_df['x_min_diff']=structures_df['x_min']-structures_df['x']
    structures_df['y_min_diff']=structures_df['y_min']-structures_df['y']
    structures_df['z_min_diff']=structures_df['z_min']-structures_df['z']
    structures_df['x_max_diff']=structures_df['x_max']-structures_df['x']
    structures_df['y_max_diff']=structures_df['y_max']-structures_df['y']
    structures_df['z_max_diff']=structures_df['z_max']-structures_df['z']
    #因为x_mean之类都是以分子为单位，
    structures_df=structures_df.drop(columns=['x_mean','x_std','x_min','x_max','y_mean','y_std','y_min','y_max','z_mean','z_std','z_min','z_max'])
    
    return reduce_memory_usage(structures_df)




def add_structures_df(df):#对于train_df,test_df,structures_df都是同一个分子出现好几次
    for atom_idx in [0,1]:#0,1分别为分子中原子对的两个原子的索引
        df = pd.merge(df, structures_df, how = 'left',
                          left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                          right_on = ['molecule_name',  'atom_index'])


        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                    'x': f'x_{atom_idx}',
                                    'y': f'y_{atom_idx}',
                                    'z': f'z_{atom_idx}',
                                    'ele_neg': f'ele_neg_{atom_idx}',
                                    'rad': f'rad_{atom_idx}',
                                    'x_mean_diff':f'x_mean_diff_{atom_idx}',
                                    'x_std_diff':f'x_std_diff_{atom_idx}',
                                     'x_min_diff':f'x_min_diff_{atom_idx}',
                                    'x_max_diff':f'x_max_diff_{atom_idx}',
                                    'y_mean_diff':f'y_mean_diff_{atom_idx}',
                                    'y_std_diff':f'y_std_diff_{atom_idx}',
                                     'y_min_diff':f'y_min_diff_{atom_idx}',
                                    'y_max_diff':f'y_max_diff_{atom_idx}',
                                    'z_mean_diff':f'z_mean_diff_{atom_idx}',
                                    'z_std_diff':f'z_std_diff_{atom_idx}',
                                     'z_min_diff':f'z_min_diff_{atom_idx}',
                                    'z_max_diff':f'z_max_diff_{atom_idx}',
                                   })
    struct_df=pd.DataFrame()
    struct_df['molecule_name']=structures_df['molecule_name'].unique()
        #分子中原子的平均坐标
    struct_df['x_mean']=structures_df.groupby(by='molecule_name')['x'].mean().values
    struct_df['y_mean']=structures_df.groupby(by='molecule_name')['y'].mean().values
    struct_df['z_mean']=structures_df.groupby(by='molecule_name')['z'].mean().values
    struct_df['x_std']=structures_df.groupby(by='molecule_name')['x'].std().values
    struct_df['y_std']=structures_df.groupby(by='molecule_name')['y'].std().values
    struct_df['z_std']=structures_df.groupby(by='molecule_name')['z'].std().values
    struct_df['x_min']=structures_df.groupby(by='molecule_name')['x'].min().values
    struct_df['y_min']=structures_df.groupby(by='molecule_name')['y'].min().values
    struct_df['z_min']=structures_df.groupby(by='molecule_name')['z'].min().values
    struct_df['x_max']=structures_df.groupby(by='molecule_name')['x'].max().values
    struct_df['y_max']=structures_df.groupby(by='molecule_name')['y'].max().values
    struct_df['z_max']=structures_df.groupby(by='molecule_name')['z'].max().values
    
    df=pd.merge(df,struct_df,on='molecule_name',how='left')
    return reduce_memory_usage(df)
def add_magnetic_features(df, atom_idx):
    df=pd.merge(df,magnetic_shielding_tensors_df,how='left',
               left_on=['molecule_name',f'atom_index_{atom_idx}'],
                right_on=['molecule_name','atom_index'])
    
    df = df.drop(columns='atom_index')
    df = df.rename(columns={'XX': f'XX_{atom_idx}',
                            'YX': f'YX_{atom_idx}',
                            'ZX': f'ZX_{atom_idx}',
                            'XY': f'XY_{atom_idx}',
                            'YY': f'YY_{atom_idx}',
                            'ZY': f'ZY_{atom_idx}',
                            'XZ': f'XZ_{atom_idx}',
                            'YZ': f'YZ_{atom_idx}',
                            'ZZ': f'ZZ_{atom_idx}'
                            })
    return df

def add_mulliken_charges(df, atom_idx):
    df=pd.merge(df,mulliken_charges_df,how='left',
               left_on=['molecule_name',f'atom_index_{atom_idx}'],
                right_on=['molecule_name','atom_index'])
    
    df = df.drop(columns='atom_index')
    df = df.rename(columns={'mulliken_charge': f'mulliken_charge_{atom_idx}'})
    return df

def add_cos_features(df):
    atom_0_array=df[['x_0','y_0','z_0']].values
    atom_1_array=df[['x_1','y_1','z_1']].values
    #该分子中原子的平均坐标
    atom_mean_array=df[['x_mean','y_mean','z_mean']].values
    #分子中，原子对的第一个原子的坐标和该分子中原子的平均坐标的距离
    df['distance_center0']=np.linalg.norm(atom_0_array-atom_mean_array,ord=2,axis=1)
    df['distance_center1'] = np.linalg.norm(atom_1_array-atom_mean_array,ord=2,axis=1)
    df['distance_c0'] = np.sqrt((df['x_0']-df['x_closest_0'])**2 +                                 (df['y_0']-df['y_closest_0'])**2 +                                 (df['z_0']-df['z_closest_0'])**2)
    df['distance_c1'] = np.sqrt((df['x_1']-df['x_closest_1'])**2 +                                 (df['y_1']-df['y_closest_1'])**2 +                                 (df['z_1']-df['z_closest_1'])**2)
    
    df["distance_f0"] = np.sqrt((df['x_0']-df['x_farthest_0'])**2 +                                 (df['y_0']-df['y_farthest_0'])**2 +                                 (df['z_0']-df['z_farthest_0'])**2)
    df["distance_f1"] = np.sqrt((df['x_1']-df['x_farthest_1'])**2 +                                 (df['y_1']-df['y_farthest_1'])**2 +                                 (df['z_1']-df['z_farthest_1'])**2)


    vec_center0_x = (df['x_0'] - df['x_mean']) / (df["distance_center0"] + 1e-10)
    vec_center0_y = (df['y_0'] - df['y_mean']) / (df["distance_center0"] + 1e-10)
    vec_center0_z = (df['z_0'] - df['z_mean']) / (df["distance_center0"] + 1e-10)

    vec_center1_x = (df['x_1'] - df['x_mean']) / (df["distance_center1"] + 1e-10)
    vec_center1_y = (df['y_1'] - df['y_mean']) / (df["distance_center1"] + 1e-10)
    vec_center1_z = (df['z_1'] - df['z_mean']) / (df["distance_center1"] + 1e-10)


    #分子中，构成原子的第一个原子构成的原子对中距离最短的距离 df['distance_closest_0']
    # 分子中，构成原子的第二个原子构成的原子对中距离最短的距离 df['distance_closest_1']
    # 分子中，构成原子的第一个原子构成的原子对中距离最长的距离 df['distance_farthest_0']
    # 分子中，构成原子的第一个原子构成的原子对中距离最长的距离 df['distance_farthest_1']

    vec_c0_x = (df['x_0'] - df['x_closest_0']) / (df['distance_closest_0'] + 1e-10)
    vec_c0_y = (df['y_0'] - df['y_closest_0']) / (df['distance_closest_0'] + 1e-10)
    vec_c0_z = (df['z_0'] - df['z_closest_0']) / (df['distance_closest_0'] + 1e-10)

    vec_c1_x = (df['x_1'] - df['x_closest_1']) / (df['distance_closest_1'] + 1e-10)
    vec_c1_y = (df['y_1'] - df['y_closest_1']) / (df['distance_closest_1'] + 1e-10)
    vec_c1_z = (df['z_1'] - df['z_closest_1']) / (df['distance_closest_1'] + 1e-10)

    vec_f0_x = (df['x_0'] - df['x_farthest_0']) / (df['distance_farthest_0'] + 1e-10)
    vec_f0_y = (df['y_0'] - df['y_farthest_0']) / (df['distance_farthest_0'] + 1e-10)
    vec_f0_z = (df['z_0'] - df['z_farthest_0']) / (df['distance_farthest_0'] + 1e-10)

    vec_f1_x = (df['x_1'] - df['x_farthest_1']) / (df['distance_farthest_1'] + 1e-10)
    vec_f1_y = (df['y_1'] - df['y_farthest_1']) / (df['distance_farthest_1'] + 1e-10)
    vec_f1_z = (df['z_1'] - df['z_farthest_1']) / (df['distance_farthest_1'] + 1e-10)

    vec_x = (df['x_1'] - df['x_0']) / df['dist']
    vec_y = (df['y_1'] - df['y_0']) / df['dist']
    vec_z = (df['z_1'] - df['z_0']) / df['dist']

    #此时cos=a*b/(|a|*|b|)
    #原子对的第一个原子到其构成的距离最近的原子对向量，和原子对的第二个原子到其构成的距离最近的原子对向量的cos
    df["cos_c0_c1"] = vec_c0_x * vec_c1_x + vec_c0_y * vec_c1_y + vec_c0_z * vec_c1_z
    df["cos_f0_f1"] = vec_f0_x * vec_f1_x + vec_f0_y * vec_f1_y + vec_f0_z * vec_f1_z

    df["cos_c0_f0"] = vec_c0_x * vec_f0_x + vec_c0_y * vec_f0_y + vec_c0_z * vec_f0_z
    df["cos_c1_f1"] = vec_c1_x * vec_f1_x + vec_c1_y * vec_f1_y + vec_c1_z * vec_f1_z

    #原子对的第一原子到分子中心点的向量，和原子对的第二个原子到中心点的向量的cos，当atom_1或atom_0的坐标为分子的平均坐标时，cos为nan
    df["cos_center0_center1"] = vec_center0_x * vec_center1_x                                 + vec_center0_y * vec_center1_y                                 + vec_center0_z * vec_center1_z
   #原子对的第一个原子到其构成的距离最近的原子对向量，和该原子对向量的cos
    df["cos_c0"] = vec_c0_x * vec_x + vec_c0_y * vec_y + vec_c0_z * vec_z
    df["cos_c1"] = vec_c1_x * vec_x + vec_c1_y * vec_y + vec_c1_z * vec_z

    df["cos_f0"] = vec_f0_x * vec_x + vec_f0_y * vec_y + vec_f0_z * vec_z
    df["cos_f1"] = vec_f1_x * vec_x + vec_f1_y * vec_y + vec_f1_z * vec_z

    # 原子对的第一原子到分子中心点的向量，和原子对的向量的cos,当atom_1的坐标为分子的平均坐标时，cos为nan
    df["cos_center0"] = vec_center0_x * vec_x + vec_center0_y * vec_y + vec_center0_z * vec_z
    df["cos_center1"] = vec_center1_x * vec_x + vec_center1_y * vec_y + vec_center1_z * vec_z

    return reduce_memory_usage(df)
def add_split_type(df):
     #将原子结合类型type分为type_0(数字)，type_1(英文字母)
    df['type_0'] = df['type'].apply(lambda x: x[0])
    df['type_1'] = df['type'].apply(lambda x: x[1:])
    
    return df

def add_dist(df):
    #两点的坐标
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values
    #添加两点间的距离
    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1,ord=2)
    
    #两点在x轴上的距离
    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
   
    #两点在y轴上的距离
    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
    #两点在z轴上的距离
    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2
    
    return reduce_memory_usage(df)


def add_others(df):
   
    #一个分子有几个原子对
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    

    #df['dist_mean'],df['dist_mean_diff'],df['dist_min'],df['dist_max']这些都无用
    
    #分子中不同原子对的平均距离
    df['molecule_dist_mean']=df.groupby(by='molecule_name')['dist'].transform('mean')
    df['molecule_dist_mean_diff']=df['molecule_dist_mean']-df['dist']
 
    df['molecule_dist_std']=df.groupby(by='molecule_name')['dist'].transform('std')
    df['molecule_dist_std_diff']=df['molecule_dist_std']-df['dist']

    df['molecule_dist_mean_min']=df.groupby(by='molecule_name')['dist'].transform('min')
    df['molecule_dist_mean_min_diff']=df['molecule_dist_mean_min']-df['dist']

    df['molecule_type_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['molecule_type_dist_max_diff']=df['molecule_type_dist_max']-df['dist']

    
    df=reduce_memory_usage(df)
    #分子的结合类型下的不同原子对的平均距离
    df['molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df['molecule_type0_dist_mean'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('mean')
    df['molecule_type1_dist_mean'] = df.groupby(['molecule_name', 'type_1'])['dist'].transform('mean')
    
    df['molecule_type_dist_mean_diff'] = df['molecule_type_dist_mean'] - df['dist']
    df['molecule_type0_dist_mean_diff'] = df['molecule_type0_dist_mean'] - df['dist']
    df['molecule_type1_dist_mean_diff'] = df['molecule_type1_dist_mean'] - df['dist']
    
    df['molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df['molecule_type0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df['molecule_type1_dist_std'] = df.groupby(['molecule_name', 'type_1'])['dist'].transform('std')
    
    df['molecule_type_dist_std_diff'] = df['molecule_type_dist_std']-df['dist']
    df['molecule_type0_dist_std_diff'] = df['molecule_type0_dist_std']-df['dist']
    df['molecule_type1_dist_std_diff'] = df['molecule_type1_dist_std']-df['dist']
    

    
    
    df['molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df['molecule_type0_dist_min'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('min')
    df['molecule_type1_dist_min'] = df.groupby(['molecule_name', 'type_1'])['dist'].transform('min')
    
    df['molecule_type_dist_min_diff'] = df['molecule_type_dist_min']-df['dist']
    df['molecule_type0_dist_min_diff'] = df['molecule_type0_dist_min']-df['dist']
    df['molecule_type1_dist_min_diff'] = df['molecule_type1_dist_min']-df['dist']
    

    
    df['molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df['molecule_type0_dist_max'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('max')
    df['molecule_type1_dist_max'] = df.groupby(['molecule_name', 'type_1'])['dist'].transform('max')
    
    df['molecule_type_dist_max_diff'] = df['molecule_type_dist_max']-df['dist']
    df['molecule_type0_dist_max_diff'] = df['molecule_type0_dist_max']-df['dist']
    df['molecule_type1_dist_max_diff'] = df['molecule_type1_dist_max']-df['dist']
    

    
    df=reduce_memory_usage(df)
        #atom_0都是H就不弄了
    df['molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df['molecule_atom_1_dist_mean_diff']= df['molecule_atom_1_dist_mean']-df['dist']

    df['molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df['molecule_atom_1_dist_std_diff'] = df['molecule_atom_1_dist_std'] - df['dist']

    df['molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df['molecule_atom_1_dist_min_diff'] = df['molecule_atom_1_dist_min'] - df['dist']

    df['molecule_atom_1_dist_max'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('max')
    df['molecule_atom_1_dist_max_diff'] = df['molecule_atom_1_dist_min'] - df['dist']

    
    df=reduce_memory_usage(df)
        #原子对：第一个原子索引下，不同的第二个原子的坐标信息
    df['molecule_atom_index_0_x_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('mean')
    df['molecule_atom_index_0_x_1_mean_diff'] = df['molecule_atom_index_0_x_1_mean'] - df['x_1']

    df['molecule_atom_index_0_x_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('max')
    df['molecule_atom_index_0_x_1_max_diff'] = df['molecule_atom_index_0_x_1_max'] - df['x_1']

    df['molecule_atom_index_0_x_1_min'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('min')
    df['molecule_atom_index_0_x_1_min_diff'] = df['molecule_atom_index_0_x_1_min'] - df['x_1']
 
    df['molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df['molecule_atom_index_0_x_1_std_diff'] = df['molecule_atom_index_0_x_1_std']-df['x_1']

    
    df['molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df['molecule_atom_index_0_y_1_mean_diff'] = df['molecule_atom_index_0_y_1_mean'] - df['y_1']

    df['molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df['molecule_atom_index_0_y_1_max_diff'] = df['molecule_atom_index_0_y_1_max'] - df['y_1']

    df['molecule_atom_index_0_y_1_min'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('min')
    df['molecule_atom_index_0_y_1_min_diff'] = df['molecule_atom_index_0_y_1_min'] - df['y_1']

    df['molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df['molecule_atom_index_0_y_1_std_diff'] = df['molecule_atom_index_0_y_1_std']-df['y_1']
 
    
    df['molecule_atom_index_0_z_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('mean')
    df['molecule_atom_index_0_z_1_mean_diff'] = df['molecule_atom_index_0_z_1_mean'] - df['z_1']

    df['molecule_atom_index_0_z_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('max')
    df['molecule_atom_index_0_z_1_max_diff'] = df['molecule_atom_index_0_z_1_max'] - df['z_1']

    df['molecule_atom_index_0_z_1_min'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('min')
    df['molecule_atom_index_0_z_1_min_diff'] = df['molecule_atom_index_0_z_1_min'] - df['z_1']

    df['molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df['molecule_atom_index_0_z_1_std_diff'] = df['molecule_atom_index_0_z_1_std']-df['z_1']

  #原子对：第二个原子索引下，不同的第一个原子的坐标信息
    df['molecule_atom_index_1_x_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('mean')
    df['molecule_atom_index_1_x_0_mean_diff'] = df['molecule_atom_index_1_x_0_mean'] - df['x_0']
  
    df['molecule_atom_index_1_x_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('max')
    df['molecule_atom_index_1_x_0_max_diff'] = df['molecule_atom_index_1_x_0_max'] - df['x_0']
  
    df['molecule_atom_index_1_x_0_min'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('min')
    df['molecule_atom_index_1_x_0_min_diff'] = df['molecule_atom_index_1_x_0_min'] - df['x_0']
 
    df['molecule_atom_index_1_x_0_std'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('std')
    df['molecule_atom_index_1_x_0_std_diff'] = df['molecule_atom_index_1_x_0_std']-df['x_0']
 
    
    df['molecule_atom_index_1_y_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('mean')
    df['molecule_atom_index_1_y_0_mean_diff'] = df['molecule_atom_index_1_y_0_mean'] - df['y_0']
 
    df['molecule_atom_index_1_y_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('max')
    df['molecule_atom_index_1_y_0_max_diff'] = df['molecule_atom_index_1_y_0_max'] - df['y_0']

    df['molecule_atom_index_1_y_0_min'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('min')
    df['molecule_atom_index_1_y_0_min_diff'] = df['molecule_atom_index_1_y_0_min'] - df['y_0']
 
    df['molecule_atom_index_1_y_0_std'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('std')
    df['molecule_atom_index_1_y_0_std_diff'] = df['molecule_atom_index_1_y_0_std']-df['y_0']
 
   
    df['molecule_atom_index_1_z_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('mean')
    df['molecule_atom_index_1_z_0_mean_diff'] = df['molecule_atom_index_1_z_0_mean'] - df['z_0']
 
    df['molecule_atom_index_1_z_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('max')
    df['molecule_atom_index_1_z_0_max_diff'] = df['molecule_atom_index_1_z_0_max'] - df['z_0']

    df['molecule_atom_index_1_z_0_min'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('min')
    df['molecule_atom_index_1_z_0_min_diff'] = df['molecule_atom_index_1_z_0_min'] - df['z_0']
   
    df['molecule_atom_index_1_z_0_std'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('std')
    df['molecule_atom_index_1_z_0_std_diff'] = df['molecule_atom_index_1_z_0_std']-df['z_0']

  
    df=reduce_memory_usage(df)
    # Andrew's features selected
    #分子的索引出来的的原子作为不同原子对的第一个原子时

    #原子对的平均距离
    df['molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
     #原子对的平均距离和原子对距离的差值
    df['molecule_atom_index_0_dist_mean_diff'] = df['molecule_atom_index_0_dist_mean'] - df['dist']
    # 原子对距离标准差
    df['molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df['molecule_atom_index_0_dist_std_diff'] = df['molecule_atom_index_0_dist_std']-df['dist']
    #原子对距离的最小值
    df['molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    #原子对距离的最小值和原子对距离的差值
    df['molecule_atom_index_0_dist_min_diff'] = df['molecule_atom_index_0_dist_min'] - df['dist']
    # 原子对距离的最大值
    df['molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    # 原子对距离的最大值和原子对距离的差值
    df['molecule_atom_index_0_dist_max_diff'] = df['molecule_atom_index_0_dist_max'] - df['dist']
    


    # 分子的索引出来的的原子作为不同原子对的第二个原子时
    df['molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df['molecule_atom_index_1_dist_mean_diff'] = df['molecule_atom_index_1_dist_mean'] - df['dist']
    df['molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df['molecule_atom_index_1_dist_std_diff'] = df['molecule_atom_index_1_dist_std']-df['dist']   

    df['molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df['molecule_atom_index_1_dist_min_diff'] = df['molecule_atom_index_1_dist_min'] - df['dist']
    df['molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df['molecule_atom_index_1_dist_max_diff'] = df['molecule_atom_index_1_dist_max'] - df['dist']
    
  
    
    
    return reduce_memory_usage(df)#减少df所占的内存
   




def map_atom_info(df_1,df_2, atom_idx):#在add_closest_atom方法中用
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    df = df.drop(columns=['atom_index'])

    return df


def get_features(df):

    
    df=add_dist(df)
    print('add dist 成功',df.shape)
    df=add_split_type(df)
    print('add split_types 成功',df.shape)
    df=add_closest_farthest_atom(df)
    print('add closest_farthest_atom 成功',df.shape)
    df=add_cos_features(df)
    print('add cos_features 成功',df.shape)
    df=add_others(df)
    print('add其他创造的属性成功',df.shape)

    
                
    return df


# In[3]:


get_ipython().run_cell_magic('time', '', "#加载文件\n\ntrain_dtypes = {\n    'molecule_name': 'category',\n    'atom_index_0': 'int8',\n    'atom_index_1': 'int8',\n    'type': 'category',\n    'scalar_coupling_constant': 'float32'\n}\n\ntrain_df=pd.read_csv('champs-scalar-coupling/train.csv',dtype=train_dtypes)\ntest_df=pd.read_csv('champs-scalar-coupling/test.csv',dtype=train_dtypes)\nsample_submission_df=pd.read_csv('champs-scalar-coupling/sample_submission.csv')\nstructures_df=pd.read_csv('champs-scalar-coupling/structures.csv')\nprint('train_set')\nprint(train_df.shape)\ndisplay(train_df.head())\nprint('================')\nprint('test_set')\nprint(test_df.shape)\ndisplay(test_df.head())\nprint('================')\nprint('structures')\nprint(structures_df.shape)\ndisplay(structures_df.head())\nprint('================')\nprint('sample_submission')\nprint(sample_submission_df.shape)\ndisplay(sample_submission_df.head())")


# * 将8个type变9个

# In[4]:


get_ipython().run_cell_magic('time', '', "\ndf = train_df.copy()\ndf=pd.merge(df,structures_df,how='left',left_on=['molecule_name','atom_index_0'],right_on=['molecule_name','atom_index'])\ndf.rename(columns={'x':'x_0','y':'y_0','z':'z_0'},inplace=True)\ndf.drop(columns=['atom_index'],inplace=True)\ndf=pd.merge(df,structures_df,how='left',left_on=['molecule_name','atom_index_1'],right_on=['molecule_name','atom_index'])\ndf.rename(columns={'x':'x_1','y':'y_1','z':'z_1'},inplace=True)\ndf.drop(columns=['atom_index'],inplace=True)\n\natom_0_array=df[['x_0','y_0','z_0']].values\natom_1_array=df[['x_1','y_1','z_1']].values\n\ndf['dist']=np.linalg.norm(atom_0_array-atom_1_array,ord=2,axis=1)\ndf['type']=df['type'].cat.add_categories('1JHC_high')\ndf.loc[(df.type=='1JHC')&(df.dist<1.065),'type']='1JHC_high'\ntrain_df['type']=df['type'].values\nprint('新增type后的train_csv')\ndisplay(train_df.head())\n\ndel df\ngc.collect()\n\ndf = test_df.copy()\ndf=pd.merge(df,structures_df,how='left',left_on=['molecule_name','atom_index_0'],right_on=['molecule_name','atom_index'])\ndf.rename(columns={'x':'x_0','y':'y_0','z':'z_0'},inplace=True)\ndf.drop(columns=['atom_index'],inplace=True)\ndf=pd.merge(df,structures_df,how='left',left_on=['molecule_name','atom_index_1'],right_on=['molecule_name','atom_index'])\ndf.rename(columns={'x':'x_1','y':'y_1','z':'z_1'},inplace=True)\ndf.drop(columns=['atom_index'],inplace=True)\n\natom_0_array=df[['x_0','y_0','z_0']].values\natom_1_array=df[['x_1','y_1','z_1']].values\n\ndf['dist']=np.linalg.norm(atom_0_array-atom_1_array,ord=2,axis=1)\ndf['type']=df['type'].cat.add_categories('1JHC_high')\ndf.loc[(df.type=='1JHC')&(df.dist<1.065),'type']='1JHC_high'\ntest_df['type']=df['type'].values\nprint('新增type后的test_csv')\ndisplay(test_df.head())\n\n\ndel df\ngc.collect()")


# * 是否缩小范围

# In[5]:


#train_df=train_df.head(100)
#test_df=test_df.head(100)


# * 添加qm9特征

# In[6]:


get_ipython().run_cell_magic('time', '', "qm9_df = pd.read_pickle('/kaggle/input/quantum-machine-9-qm9/data.covs.pickle')\nqm9_df.drop(columns=['molecule_name',\n               'atom_index_0', \n               'atom_index_1', \n               'type',\n               'scalar_coupling_constant'],inplace=True)\nprint('读取qm9文件成功')\nqm9_df = reduce_memory_usage(qm9_df)\nprint('qm9修改类型，减少内存完毕')\ntrain_df = pd.merge(train_df, qm9_df, how='left', on=['id'])\ntest_df = pd.merge(test_df, qm9_df, how='left', on=['id'])\n\ndisplay(train_df.head())\ndisplay(test_df.head())\nprint(train_df.shape)\nprint(test_df.shape)\ndel qm9_df\ngc.collect()")


# In[7]:


get_ipython().run_cell_magic('time', '', "structures_df=change_structures_df(structures_df)\ntrain_df=add_structures_df(train_df)\ntest_df=add_structures_df(test_df)\nprint('add structures_df 成功')\ndel structures_df")


# In[8]:


print(train_df.shape)
print(test_df.shape)


# In[9]:


get_ipython().run_cell_magic('time', '', 'train_df=get_features(train_df)\ntest_df=get_features(test_df)\nprint(train_df.shape,train_df.memory_usage().sum()/1024**2)\nprint(test_df.shape,test_df.memory_usage().sum()/1024**2)')


# * 把对所有样本有相同的特征给去掉，原子对的第一个原子都是H，所以把 atom_0,ele_neg_0,rad_0，molecule_name,type,atom_index_0,atom_index_1列去掉

# In[10]:


for col in train_df.columns:
    a=train_df[col].values
    v=a[0]
    if (a==[v]*len(a)).all():
        print(col)
#train_df比test_df多了一个scalar_coupling_constant
train_df.drop(columns=['atom_0','ele_neg_0','rad_0','molecule_name','type','scalar_coupling_constant','atom_index_0','atom_index_1'],inplace=True)
test_df.drop(columns=['atom_0','ele_neg_0','rad_0','molecule_name','type','atom_index_0','atom_index_1'],inplace=True)


# * #将标称属性（值为字符）进行编码（此时为：用整数表示）

# In[11]:


get_ipython().run_cell_magic('time', '', "from sklearn.preprocessing import LabelEncoder\nfor f in ['atom_1','type_0','type_1']:\n    le = LabelEncoder()\n    le.fit(list(train_df[f].values) + list(test_df[f].values))\n    train_df[f] = le.transform(list(train_df[f].values))\n    test_df[f] = le.transform(list(test_df[f].values))")


# In[12]:


train_df[train_df.molecule_atom_index_1_z_0_std_diff==np.nan]


# In[13]:


print(train_df.shape)
print(test_df.shape)


# In[14]:


train_df.head()


# In[15]:


train_df


# In[16]:


for i in train_df.columns:
    print(i)


# 有无缺失值

# In[17]:


a=train_df.isnull().sum()
for i in a.index:
    if a.loc[i] != 0:
        print(i,a.loc[i])


# In[18]:


train_df.to_csv('/kaggle/working/base_qm9_train.csv.gz',compression='gzip',index=False)
test_df.to_csv('/kaggle/working/base_qm9_test.csv.gz',compression='gzip',index=False)

