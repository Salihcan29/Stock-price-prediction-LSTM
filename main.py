# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:46:52 2020

@author: Salihcan
"""

from pandas_datareader import data as pdr

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler

#%%
n = 50 # LSTM input number

#%%
print("Stock price prediction application.")
print("'\033[93m'ATTENTION: Model estimates are based on historical graph data and can't make high accurate predictions. Do not use for investments.\n")

print("To use the application, you first need training data. For this, you need to enter the stock codes. After training, the program will restart. You must enter different stock codes and dates for the test. In this way, you can visualize to see model predictions.")
print("More training data provides stronger model.")

print("There is few example stocks: ")
print("# EXAMPLE STOCK CODES FOR TRAINING AND TESTING")
print("# --TRAIN-- || --TEST--")
print("")
print("# PETKM.IS  || # PGSUS.IS")
print("# TUPRS.IS  || # CCOLA.IS")
print("# THYAO.IS  || # TM")
print("# GOOGL     || # 005930.KS")
print("")
print("# MGROS.IS  || # HALKB.IS")
print("")
print("# VOD       || # SISE.IS")
print("# TKC       || # GM")
print("# ARCLK.IS  || # KCHOL.IS")
print("# GARAN.IS  || # TUKAS.IS")

print("")

print("Enter stock codes for train or test ai model(-1 for stop)")
stock_names = []
input_stock_code = input("Enter stock code: ")
while (input_stock_code != "-1"):
    stock_names.append(input_stock_code)
    input_stock_code = input("Enter stock code: ")

print("Date interval of training&test data shouldn't conflict. This causes the model to memorize graphs.")
start = input("Enter date interval start (format yyyy-mm-dd): ")
end = input("Enter date interval end (format yyyy-mm-dd): ")

x_trains = []
y_trains = []
np_df_scaleds = []
scalers = []
for stock_name in stock_names:

    dataset = pdr.get_data_yahoo(stock_name, start=start, end=end)
    
    new_df = dataset[['Close']]
    new_df["returns"] = new_df["Close"].pct_change()
    new_df["log_returns"] = np.log(1+new_df.returns)
    new_df.dropna(inplace=True)
    
    np_df = new_df[["Close","log_returns"]].values
    
    scaler = MinMaxScaler().fit(np_df)
    np_df_scaled = scaler.fit_transform(np_df)
    
    x_train = []
    y_train = []
    
    for i in range(len(np_df)-n):
        x_train.append(np_df_scaled[i:i+n])
        y_train.append([np_df_scaled[i+n,0]])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train).reshape(len(y_train),1)
    
    x_trains.append(x_train)
    y_trains.append(y_train)
    np_df_scaleds.append(np_df_scaled)
    scalers.append(scaler)
    
#%%
from keras.callbacks import History 
history = History()

train_loss = []
val_loss = []

model = Sequential()
model.add(LSTM(20,input_shape =(n,x_train.shape[2])))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(loss='mse', optimizer="adam")

for i in range(125):
    for index,x_train in enumerate(x_trains):
        model.fit(x_train,y_trains[index],epochs=1,batch_size=32,callbacks=[history],verbose=0)
        train_loss.append(history.history["loss"])
        val_loss.append(history.history["val_loss"])
    
    print("Train & Validation loss: ",train_loss[-1][-1],"",val_loss[-1][-1])
    print("Remaining:", 125-i)

plt.plot(np.array(train_loss).flatten())
plt.plot(np.array(val_loss).flatten())

#%%
real_ys = []
predicted_ys = []
for index,item in enumerate(stock_names):
    real_y = scalers[index].inverse_transform(np_df_scaleds[index])[:,0]
    predicted_y = np.concatenate((model.predict(x_trains[index]),np.zeros(len(x_trains[index])).reshape(-1,1)),axis=1)
    predicted_y = scalers[index].inverse_transform(predicted_y)[:,0]
    real_ys.append(real_y)
    predicted_ys.append(predicted_y)

subplot_n = int(np.sqrt(len(stock_names)))

for index,item in enumerate(stock_names):
    ax = plt.subplot(subplot_n,subplot_n,1+index)
    ax.set_title(item)
    ax.plot(real_ys[index])
    ax.scatter(range(len(real_ys[index])),real_ys[index],c="green",s=1)
    ax.scatter(range(n,len(real_ys[index])),predicted_ys[index],c="red",s=2)

#%%

for index, name in enumerate(stock_names):
    MONEY = 1000000
    OWNED_STOCKS = 0  

    money_timeline = []
    for i in range(len(real_ys[index])-n):
        money_timeline.append(MONEY+OWNED_STOCKS*real_ys[index][n-1+i])
        if(real_ys[index][n-1+i] < predicted_ys[index][i]):
            OWNED_STOCKS += MONEY/real_ys[index][n-1+i]
            MONEY = MONEY%real_ys[index][n-1+i]
        if(real_ys[index][n-1+i] > predicted_ys[index][i]):
            MONEY += OWNED_STOCKS*real_ys[index][n-1+i]
            OWNED_STOCKS = 0

    ax = plt.subplot(subplot_n,subplot_n,index+1)
    ax.set_title(name)
    ax.text(0.01,0.01,"Real diff: {}%".format(round(-(real_ys[index][49]-real_ys[index][-1])/real_ys[index][49]*100,2)),transform=ax.transAxes)
    ax.text(0.01,0.11,"Robot diff: {}%".format(round(-(money_timeline[0]-money_timeline[-1])/money_timeline[0]*100,2)),transform=ax.transAxes)
    ax.plot(money_timeline)
    ax.ticklabel_format(useOffset=False, style='plain')