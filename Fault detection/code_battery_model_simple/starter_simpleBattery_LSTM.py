import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input



#%%
## Load data
df10 = pd.read_excel('data10c.xlsx')
df03 = pd.read_excel('data03c.xlsx')
df01 = pd.read_excel('data01c.xlsx')


#%%
# normalize 데이터가 대략 -1~1 사이의 값을 갖도록
df10['Current (A)'] = df10['Current (A)']/4
df10['SOC'] = df10['SOC'] - 0.5
df10['Cell Voltage (V)'] = (df10['Cell Voltage (V)'] - 3)

df03['Current (A)'] = df03['Current (A)']/4
df03['SOC'] = df03['SOC'] - 0.5
df03['Cell Voltage (V)'] = (df03['Cell Voltage (V)'] - 3)

df01['Current (A)'] = df01['Current (A)']/4
df01['SOC'] = df01['SOC'] - 0.5
df01['Cell Voltage (V)'] = (df01['Cell Voltage (V)'] - 3)


#%%
# input data plot
# soc = np.concatenate([df10['SOC'], df03['SOC'], df01['SOC']])
# volt = np.concatenate([df10['Cell Voltage (V)'], df03['Cell Voltage (V)'], df01['Cell Voltage (V)']])
# cur = np.concatenate([df10['Current (A)'], df03['Current (A)'], df01['Current (A)']])

# plt.plot(soc)
# plt.plot(volt-3)
# plt.plot(cur)
# plt.show()

#%%
## Create dataset
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



x10 = np.array(df10[['Current (A)', 'SOC']])
y10 = np.array(df10['Cell Voltage (V)'])    # normalize output
x_data_10 , y_data_10 = create_LSTM_dataset(x10, y10, 64)

x03 = np.array(df03[['Current (A)', 'SOC']])
y03 = np.array(df03['Cell Voltage (V)'])     # normalize output
x_data_03 , y_data_03 = create_LSTM_dataset(x03, y03, 64)

x01 = np.array(df01[['Current (A)', 'SOC']])
y01 = np.array(df01['Cell Voltage (V)'])    # normalize output
x_data_01 , y_data_01 = create_LSTM_dataset(x01, y01, 64)



#%% whole dataset
x_data_ = np.vstack((x_data_10, x_data_03))
x_data = np.vstack((x_data_, x_data_01))
                   
y_data = np.concatenate([y_data_10, y_data_03, y_data_01])





#%% validation split
# split = 9000
# x_train = x_data[:split,:,:]
# y_train = y_data[:split]

# x_valid = x_data[split:,:,:]
# y_valid = y_data[split:]



# %% Train
# Generate LSTM network
model = tf.keras.Sequential()
model.add(Input(shape=(64,2)))
model.add(Bidirectional(LSTM(32, input_shape=(64, 2), return_sequences=True)))
model.add(Bidirectional(LSTM(32, input_shape=(64, 2), return_sequences=True)))
model.add(Bidirectional(LSTM(32, input_shape=(64, 2))))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))
model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# loss = tf.keras.losses.mae
loss = tf.keras.losses.mean_squared_error
model.compile(loss=loss, optimizer=optimizer)


# history=model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, batch_size=16, verbose=1)
history=model.fit(x_data, y_data, validation_split=0.1, epochs=10, batch_size=16, verbose=1)


# %%
# model = tf.keras.models.load_model("mymodel_battery_LSTM.h5")
res_model = model.predict(x_data)


#%% inverse normalizse
res_model = res_model + 3
res_data = y_data + 3


#%%
plt.figure(figsize=(13,5))
plt.plot(res_model, label = 'model')
plt.plot(res_data, '--', label = 'data', )
plt.legend()
plt.show()


#%%
# save model
model.save("mymodel_battery_LSTM.h5")




