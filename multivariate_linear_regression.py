import pandas as pd
from sklearn.linear_model import LinearRegression
from functions import predict_each_row

typ = 'bandwidth'
model = 'lr'
x_col = ['packetsize', 'cpu']

results_path = './results/191223_test/'
data_path = './data/multivariate_' + typ +'.xlsx'
plt_path = results_path + 'plt/'+ typ + '/' + model + '_' + typ + '_'
pred_path = results_path + 'pred/'+ typ + '/' + model + '_' + typ + '_'

data = pd.read_excel(data_path, header=0, index_col=0)

x = data[x_col].values
y = data[typ].values

regr = LinearRegression()

results = predict_each_row(regr, x, y, 5, plt_path, pred_path)
