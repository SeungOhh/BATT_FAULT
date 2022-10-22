import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from parameters import PATH_DATA_RAW, PATH_DATA_PREP_TRAIN
from model import *



#%%
# load data
# csv, xlsx, pkl, db 서버등에서 데이터 읽어옴
def read_train_data( file_path ):
    df_raw = pd.read_excel( PATH_DATA_PREP_TRAIN + file)
    columns = df_raw.columns
    return df_raw

#%%
# 최종 feature selection
# 학습에 사용할 Feature만 모아 놓음
# 모델에 따라 다른 feature를 선택 할 수 있도록 
def feature_selection_train_fault( df ):
    df_x = df[['DCCurrent', 'CVavg', 'CVmax', 'CVmin',
                 'MTavg', 'MTmax', 'MTmin']]
    df_y = df[['Fault']]
    return df_x, df_y

def feature_selection_train_SOH( df ):
    df_x = df[['DCCurrent', 'CVavg', 'CVmax', 'CVmin',
                 'MTavg', 'MTmax', 'MTmin']]
    df_y = df[['SOH_avg']]
    return df_x, df_y


#%%
def create_LSTM_dataset( df_x, df_y, LSTM_size = 32 ):
    arr_x = np.array(df_x)
    arr_y = np.array(df_y)
    
    xs = []
    ys = []
    
    for i in range( len(df_x) - LSTM_size - 1):
        x_ = arr_x[i:i+LSTM_size,:]
        y_ = arr_y[i + LSTM_size]
        xs.append(x_)
        ys.append(y_)

    return np.array(xs), np.array(ys)




#%%
file_list = os.listdir(PATH_DATA_PREP_TRAIN)
file = file_list[0]

df_train = read_train_data( file )
df_x,df_y = feature_selection_train_SOH( df_train )
xs, ys = create_LSTM_dataset( df_x, df_y )




