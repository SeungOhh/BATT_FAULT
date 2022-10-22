import pandas as pd
import os
from matplotlib import pyplot as plt
from parameters import PATH_DATA_RAW, PATH_DATA_PREP_ALL




#%%
# load data
# csv, xlsx, pkl, db 서버등에서 데이터 읽어옴
def read_raw_data( file_path ):
    df_raw = pd.read_csv( PATH_DATA_RAW + file, encoding='cp949' )
    columns = df_raw.columns
    df_out = df_raw.iloc[::60]
    return df_out



# feature_selection
# 필요한 input_feature, output_feature만 선택
# 혹시라도 나중에 사용 가능성이 있는 feature는 모두 선택
def feature_selection( df ):
    df_out = df[['Date_Time',
                 'DCCurrent', 'CVavg', 'CVmax', 'CVmin',
                 'MTavg', 'MTmax', 'MTmin', 
                 'Fault',
                 'IR_Chr', 'IR_Dchr',
                 'SOH_C', 'SOH_Ah', 'SOH_Wh']]
    return df_out


# feature_aug
# 추가적으로 계산이 필요한 input feature, output feature 계산 
def feature_aug( df ):
    df_out = df.copy()
    df_out['CVmax_diff'] = df_out['CVmax'] - df_out['CVavg']
    df_out['CVmin_diff'] = df_out['CVavg'] - df_out['CVmin']
    df_out['MTmax_diff'] = df_out['MTmax'] - df_out['MTavg']
    df_out['MTmin_diff'] = df_out['MTavg'] - df_out['MTmin']
    df_out['SOH_avg'] = (df_out['SOH_C'] + df_out['SOH_Ah'] + df_out['SOH_Wh'])/3
    return df_out



# feature_normalize
# 데이터를 보고 개별적으로 결정
# 중요한 점은 데이터간의 range가 비슷해지면 됨
def feature_normalize( df ):
    df_out = df.copy()
    df_out['DCCurrent'] = df_out['DCCurrent'] / 100
    df_out['CVavg'] = (df_out['CVavg']-3.7) / 0.5
    df_out['CVmax'] = (df_out['CVmax']-3.7) / 0.5
    df_out['CVmin'] = (df_out['CVmin']-3.7) / 0.5
    df_out['CVmax_diff'] = df_out['CVmax_diff'] / 0.5
    df_out['CVmin_diff'] = df_out['CVmin_diff'] / 0.5
    
    df_out['MTavg'] = (df_out['MTavg']) / 30    
    df_out['MTmax'] = (df_out['MTmax']) / 30 
    df_out['MTmin'] = (df_out['MTmin']) / 30   
    df_out['MTmax_diff'] = df_out['MTmax_diff'] / 30
    df_out['MTmax_diff'] = df_out['MTmax_diff'] / 30  
    
    df_out['IR_Chr'] = df_out['IR_Chr'] / 1000
    df_out['IR_Dchr'] = df_out['IR_Dchr'] / 1000
    
    df_out['SOH_avg'] = df_out['SOH_avg']/100
    return df_out 

# normalize를 역산
def feature_normalize_inv( df ):
    
    df_out = df.copy()
    df_out['DCCurrent'] = df_out['DCCurrent'] * 100
    df_out['CVavg'] = (df_out['CVavg']) * 0.5 + 3.7
    df_out['CVmax'] = (df_out['CVmax']) * 0.5 + 3.7
    df_out['CVmin'] = (df_out['CVmin']) * 0.5 + 3.7
    df_out['CVmax_diff'] = df_out['CVmax_diff'] * 0.5
    df_out['CVmin_diff'] = df_out['CVmin_diff'] * 0.5
    
    df_out['MTavg'] = (df_out['MTavg']) * 30    
    df_out['MTmax'] = (df_out['MTmax']) * 30 
    df_out['MTmin'] = (df_out['MTmin']) * 30   
    df_out['MTmax_diff'] = df_out['MTmax_diff'] * 30
    df_out['MTmax_diff'] = df_out['MTmax_diff'] * 30  
    
    df_out['IR_Chr'] = df_out['IR_Chr'] * 1000
    df_out['IR_Dchr'] = df_out['IR_Dchr'] * 1000
    
    df_out['SOH_avg'] = df_out['SOH_avg'] * 100
    return df_out 


# 


#%%
# dataframe의 모든 데이터를 plotting
def plot_pandas( df ):
    columns = df.columns
    for idx, i in enumerate(columns[:]):
        print(idx, i)
        plt.figure()
        plt.plot( df['Date_Time'], df[i] )
        time_tmp = df['Date_Time'].loc[::240]
        plt.xticks(time_tmp)
        plt.title( str(idx) + '  ' + i )




#%%
file_list = os.listdir(PATH_DATA_RAW)
file = file_list[9]

# function oriented coding
df_raw = read_raw_data(file)
df_feature = feature_selection(df_raw)
df_feature_aug = feature_aug(df_feature)
df_normal = feature_normalize(df_feature_aug)
df_normal_inv = feature_normalize_inv(df_normal)















# plot if necessary
# plot_pandas(df_feature_aug)

# save the result
# df_feature_aug.to_excel(PATH_DATA_PREP_ALL + file[:-4] + '_feature.xlsx', index = False)
df_normal.to_excel(PATH_DATA_PREP_ALL + file[:-4] + '_feature_norm.xlsx', index = False)


#%%
# for idx, i in enumerate(file_list):
#     print(i)
#     file = i
    
#     # function oriented coding
#     df_raw = read_raw_data(file)
#     df_feature = feature_selection(df_raw)
#     df_feature_aug = feature_aug(df_feature)
#     df_normal = feature_normalize(df_feature_aug)
#     df_normal_inv = feature_normalize_inv(df_normal)
    
#     # plot if necessary
#     # plot_pandas(df_feature_aug)
    
#     # save the result
#     # df_feature_aug.to_excel(PATH_DATA_PREP_ALL + file[:-4] + '_feature.xlsx', index = False)
#     df_normal.to_excel(PATH_DATA_PREP_ALL + file[:-4] + '_feature_norm.xlsx', index = False)



