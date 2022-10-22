import os
import sys
import pandas as pd
import datetime

# system path 추가
if not (os.path.dirname(os.path.realpath(__file__)) + os.sep in sys.path):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep)

if not (os.path.dirname(os.path.realpath(__file__)) + os.sep + 'preprocessing' in sys.path):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'data_raw')

if not (os.path.dirname(os.path.realpath(__file__)) + os.sep + 'postprocessing' in sys.path):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'data_prep')


# Paths
PATH_CUR = os.path.realpath(__file__) # param.py가 있는 위치
PATH_BASE_ = os.path.dirname(PATH_CUR) # 현재위치
PATH_BASE = os.path.dirname(PATH_BASE_) # 상위위치

PATH_DATA_RAW  = os.path.join(PATH_BASE, 'data_raw') + os.sep # RAW 데이터 파일이 있는 위치
PATH_DATA_PREP  = os.path.join(PATH_BASE, 'data_prep') + os.sep # prep 데이터 파일이 있는 위치
PATH_DATA_PREP_ALL  = os.path.join(PATH_BASE, 'data_prep') + os.sep + 'all' + os.sep #
PATH_DATA_PREP_TRAIN  = os.path.join(PATH_BASE, 'data_prep') + os.sep + 'train' + os.sep #
PATH_DATA_PREP_VAL  = os.path.join(PATH_BASE, 'data_prep') + os.sep + 'val' + os.sep #
PATH_DATA_PREP_TEST  = os.path.join(PATH_BASE, 'data_prep') + os.sep + 'TEST' + os.sep #

# model parameters
LSTM_size = 32
