import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

def calculate_rmse(y, yhat):
    return sqrt(mean_squared_error(y, yhat))

def make_plot(y, yhat, save_path):
    rmse = calculate_rmse(y, yhat)

    plt.title("RMSE: "+ str(rmse))
    plt.plot(y, label='y')
    plt.plot(yhat, label='pred')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.clf()
    plt.close()

def save_prediction(y, yhat, save_path):
    rmse = calculate_rmse(y, yhat)
    df_result = pd.DataFrame([y, yhat] ).T
    df_result.columns = ['y', 'yhat']
    to_add = pd.DataFrame(['rmse', rmse]).T
    to_add.columns = ['y', 'yhat']

    df_result = df_result.append(to_add)
    df_result.to_csv(save_path)

def learn_predict(model, x, y):
    model = model.fit(x, y)
    yhat = model.predict(x)

    return yhat

def predict_each_row(model, x, y, row_count, plt_path, pred_path):
    init_model = model
    results = []

    for i in range(0, len(x), row_count):
        xi = x[i:i+row_count]
        yi = y[i:i+row_count]
        yhat = learn_predict(init_model, xi, yi)
        init_model = model
        results.append(yhat)

        fname = str(xi[0][0])
        save_prediction(yi, yhat, pred_path + fname + '.csv')
        make_plot(yi, yhat, plt_path + fname + '.png')

    return results
