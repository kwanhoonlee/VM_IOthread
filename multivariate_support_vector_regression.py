import pandas as pd
from sklearn.svm import SVR
from functions import predict_each_row

typ = 'bandwidth'
target = 'cpu'
model = 'svr'
x_col = ['packetsize', typ]

results_path = './results/191226_svr/'
data_path = './data/multivariate_' + typ +'.xlsx'
plt_path = results_path + 'plt/'+ typ + '/' + model + '_' + typ + '_'
pred_path = results_path + 'pred/'+ typ + '/' + model + '_' + typ + '_'

data = pd.read_excel(data_path, header=0, index_col=0)

x = data[x_col].values
y = data[target].values

# regr = SVR(kernel='poly',C=100, gamma='auto', degree=3, epsilon=0.1, coef0=1)
regr = SVR(kernel='rbf', C=100, gamma='auto', epsilon=.1)

results = predict_each_row(regr, x, y, 5, plt_path, pred_path)
