%% lstm model
modelfile = 'mymodel_battery_LSTM.h5';
net = importKerasNetwork(modelfile);
plot(net)



%% Read data
df10 = readmatrix('data10c.xlsx','NumHeaderLines',1);
df03 = readmatrix('data03c.xlsx','NumHeaderLines',1);
df01 = readmatrix('data01c.xlsx','NumHeaderLines',1);

% normalize
df10_voltage = df10(:,1)-3; % voltage
df10_current = df10(:,2)/4; % current
df10_soc = df10(:,3)-0.5; % SOC

df03_voltage = df03(:,1)-3; % voltage
df03_current = df03(:,2)/4; % current
df03_soc = df03(:,3)-0.5; % SOC

df01_voltage = df01(:,1)-3; % voltage
df01_current = df01(:,2)/4; % current
df01_soc = df01(:,3)-0.5; % SOC

df_x_10 = [df10_current, df10_soc];
df_x_03 = [df03_current, df03_soc];
df_x_01 = [df01_current, df01_soc];

df_x = vertcat(df_x_10, df_x_03, df_x_01);
df_y = vertcat(df10_voltage, df03_voltage, df01_voltage);

% dataset 
[xs, ys] = create_LSTM_dataset(df_x, df_y, 64);




%%
% predict using the model 
res = [];
for i = 1:length(xs)
    xs_ = transpose(squeeze(xs(i,:,:)));
    res(i) = predict(net, xs_);
end



%%
% plot
figure()
hold on
plot(res + 3)
plot(ys(:,64) + 3)
legend('model', 'data')


%%
function [xs, ys] = create_LSTM_dataset( df_x, df_y, LSTM_size)
    xs = [];
    ys = [];


    for i = 1:length(df_x) - LSTM_size - 1
        x_ = df_x(i:i+LSTM_size - 1,:);
        y_ = df_y(i:i+LSTM_size - 1,:);

        xs(i,:,:) = x_;
        ys(i,:,:) = y_;
    end
end



