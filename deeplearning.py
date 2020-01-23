# Created by kwanhoon on 23/01/2020
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.optimizers import Adam
from functions import func
from sklearn.model_selection import train_test_split
import pandas as pd
target = 'cpu'
x_col = ['packetsize','modified_bandwidth']
# x_col = ['packetsize', 'pps']

results_path = './results/200121_integrated_data/'
data_path = './data/more_data/integrated_data.csv'
plt_path = results_path + 'plt/bandwidth/'  '_'
pred_path = results_path + 'pred/bandwidth/' '_'

data = pd.read_csv(data_path, header=0, index_col=0)
x = data[x_col].values
y = data[target].values

func = func(x, y)
transformed_x, transformed_y = func.transform()

train_X, test_X, train_Y, test_Y = train_test_split(transformed_x, transformed_y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=train_X.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, ))
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=[metrics.mean_absolute_error])
# model.compile(loss='mse', optimizer='', metrics=[metrics.mse, metrics.mean_absolute_error])
history = model.fit(train_X, train_Y, epochs=500, batch_size=25, verbose=2, validation_split=0.1)

yhat = model.predict(test_X)
inverted_x, inverted_y, inverted_yhat = func.invert(test_X, test_Y, yhat)

print(func.calculate_rmse(inverted_y, inverted_yhat))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()