import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, concatenate, Concatenate, Softmax, Dropout
from tensorflow.keras.layers import Bidirectional
import numpy as np





#%% TEST MODEL
def DNN_test(X_data):
    tf.random.set_seed(2) # set_seed를 안하면 매번 다른 결과가 나옴: 실행할때마다 걸어줘야함
    # Combined model
    m_input_01_01 = Input(shape = (X_data.shape[1]))
    m_dense_01_01 = Dense(256)(m_input_01_01)
    m_dense_L2 = Dense(2)(m_dense_01_01)
    m_output = Softmax()(m_dense_L2) # classification
    
    model = tf.keras.Model(inputs = m_input_01_01, outputs = m_output)
    model.summary()
    return model




#%% DNN models
def DNN_softmax(X_data):
    # Combined model
    m_input_01_01 = Input(shape = (X_data.shape[1]))
    m_dense_01_01 = Dense(256)(m_input_01_01)
    m_drop_01_01 = Dropout(rate=0.5)(m_dense_01_01)
    m_dense_01_02 = Dense(256)(m_drop_01_01)
    m_drop_01_02 = Dropout(rate=0.5)(m_dense_01_02)
    m_dense_01_03 = Dense(256)(m_drop_01_02)
    m_drop_01_02 = Dropout(rate=0.5)(m_dense_01_03)
    
    m_dense_L2 = Dense(2)(m_drop_01_02)
    m_output = Softmax()(m_dense_L2) # classification
    
    model = tf.keras.Model(inputs = m_input_01_01, outputs = m_output)
    model.summary()
    return model






#%% Bidirectional LSTM models
def Bi_LSTM_predict_small(input_shape):
    # input_shape = (LSTM_size, feature_size)
    # Combined model
    tf.random.set_seed(2) # set_seed를 안하면 매번 다른 결과가 나옴: 실행할때마다 걸어줘야함
    m_input_01 = Input(shape = input_shape)
    m_LSTM_01 = Bidirectional(LSTM(32, return_sequences=True))(m_input_01)
    m_LSTM_02 = Bidirectional(LSTM(32))(m_LSTM_01)
    
    m_dense_L0 = Dense(32)(m_LSTM_02)
    m_dense_L1 = Dense(32)(m_dense_L0)
    m_output = Dense(1)(m_dense_L1) # predict
    
    model = tf.keras.Model(inputs = m_input_01_01, outputs = m_output)
    model.summary()
    return model

def Bi_LSTM_predict_large(input_shape):
    # input_shape = (LSTM_size, feature_size)
    # Combined model
    tf.random.set_seed(2) # set_seed를 안하면 매번 다른 결과가 나옴: 실행할때마다 걸어줘야함
    m_input_01 = Input(shape = input_shape)
    m_LSTM_01 = Bidirectional(LSTM(64, return_sequences=True))(m_input_01)
    m_LSTM_02 = Bidirectional(LSTM(64, return_sequences=True))(m_LSTM_01)
    m_LSTM_03 = Bidirectional(LSTM(64))(m_LSTM_02)
    
    m_dense_L0 = Dense(64)(m_LSTM_03)
    m_dense_L1 = Dense(64)(m_dense_L0)
    m_output = Dense(1)(m_dense_L1) # predic
    
    model = tf.keras.Model(inputs = m_input_01, outputs = m_output)
    model.summary()
    return model

def Bi_LSTM_softmax_small(input_shape):
    # input_shape = (LSTM_size, feature_size)
    # Combined model
    tf.random.set_seed(2) # set_seed를 안하면 매번 다른 결과가 나옴: 실행할때마다 걸어줘야함
    m_input_01 = Input(shape = input_shape)
    m_LSTM_01 = Bidirectional(LSTM(32, return_sequences=True))(m_input_01)
    m_LSTM_02 = Bidirectional(LSTM(32))(m_LSTM_01)
    
    m_dense_L0 = Dense(32)(m_LSTM_02)
    m_dense_L1 = Dense(32)(m_dense_L0)
    m_dense_L2 = Dense(2)(m_dense_L1)
    m_output = Softmax()(m_dense_L2) # classification
    
    model = tf.keras.Model(inputs = m_input_01, outputs = m_output)
    model.summary()
    return model

def Bi_LSTM_softmax_large(input_shape):
    # input_shape = (LSTM_size, feature_size)
    # Combined model
    tf.random.set_seed(2) # set_seed를 안하면 매번 다른 결과가 나옴: 실행할때마다 걸어줘야함
    m_input_01 = Input(shape = input_shape)
    m_LSTM_01 = Bidirectional(LSTM(64, return_sequences=True))(m_input_01)
    m_LSTM_02 = Bidirectional(LSTM(64, return_sequences=True))(m_LSTM_01)
    m_LSTM_03 = Bidirectional(LSTM(64))(m_LSTM_02)
    
    m_dense_L0 = Dense(64)(m_LSTM_03)
    m_dense_L1 = Dense(64)(m_dense_L0)
    m_dense_L2 = Dense(2)(m_dense_L1)
    m_output = Softmax()(m_dense_L2) # classification
    
    model = tf.keras.Model(inputs = m_input_01, outputs = m_output)
    model.summary()
    return model






#%%
if __name__ == '__main__' : 
    model = DNN_test(np.array([[1,2,3]]))
    model.predict(np.array([[1,2,3]]))
    
    LSTM_size = 64
    feature_size = 108
    input_shape = (LSTM_size, feature_size)
    model = Bi_LSTM_softmax_large(input_shape)
    # model_LSTM_pred_small.summary()
    
    
