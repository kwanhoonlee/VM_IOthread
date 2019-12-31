import pandas as pd
from sklearn.linear_model import LinearRegression
from functions import learn_predict, make_plot, save_prediction

ps = '1024'
typ = 'pps'
target = 'cpu'
model = 'lr'
x_col = ['packetsize', typ]

results_path = './results/191231_more_data/'
data_path = './data/more_data/more_data_ps_' + ps +'.csv'
plt_path = results_path + 'plt/'+ typ + '/' + model + '_' + typ + '_'
pred_path = results_path + 'pred/'+ typ + '/' + model + '_' + typ + '_'

data = pd.read_csv(data_path, header=0, index_col=0)

x = data[x_col].values
y = data[target].values

regr = LinearRegression()

yhat = learn_predict(regr, x, y)
make_plot(y, yhat, plt_path + ps + '.png')
save_prediction(y, yhat, pred_path + ps + '.csv')