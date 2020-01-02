# Created by kwanhoon on 02/01/2020

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from functions import func

target = 'cpu'
model = 'svr'
# regr = LinearRegression()
# regr = RandomForestRegressor()
regr = SVR(kernel='rbf', C=500, gamma=100, epsilon=.001)
# regr = SVR(kernel='rbf', C=500, gamma=100, epsilon=.005)

x_col = ['packetsize', 'pps', 'modified_bandwidth']

results_path = './results/200102_integrated_data/'
data_path = './data/more_data/integrated_data.csv'
plt_path = results_path + 'plt/' + model  + '_'
pred_path = results_path + 'pred/'+ model + '_'

data = pd.read_csv(data_path, header=0, index_col=0)
x = data[x_col].values
y = data[target].values

func = func(x, y)
x_list, y_list, yhat_list = func.learn_predict_using_kfold(regr, 10)

for i in range(len(x_list)):
    inverted_x, inverted_y, inverted_yhat = func.invert(x_list[i], y_list[i], yhat_list[i])
    print(func.calculate_rmse(inverted_y, inverted_yhat))
    func.save_prediction(inverted_y, inverted_yhat, pred_path + str(i) + '.csv', inverted_x)
    func.make_plot(inverted_y, inverted_yhat, plt_path + str(i) + '.png' )