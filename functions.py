import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

class func :

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.Y = y.reshape(-1, 1)
        self.scaler_x, self.scaler_y = self.make_scalers()
        self.transformed_x, self.transformed_y = self.transform()

    @staticmethod
    def calculate_rmse(y, yhat):
        return sqrt(mean_squared_error(y, yhat))

    def make_plot(self, y, yhat, save_path):
        rmse = self.calculate_rmse(y, yhat)

        plt.title("RMSE: "+ str(rmse))
        plt.plot(y, label='y')
        plt.plot(yhat, label='pred')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=500)
        plt.clf()
        plt.close()

    def save_prediction(self,  y, yhat, save_path, x_col, x=None, ):
        rmse = self.calculate_rmse(y, yhat)
        df_result = pd.DataFrame([y, yhat]).T
        df_result.columns = ['y', 'yhat']

        if x is not None :
            df_x = pd.DataFrame(x, columns=x_col)
            df_result = pd.concat([df_result, df_x], axis=1)

        else :
            pass
        to_add = pd.DataFrame(['rmse', rmse]).T
        to_add.columns = ['y', 'yhat', ]

        df_result = df_result.append(to_add)
        df_result.to_csv(save_path)

    def learn_predict(self, model, x, y):
        model = model.fit(x, y)
        yhat = model.predict(x)

        return yhat

    # Deprecated function
    def predict_each_row(self, model, x, y, row_count, plt_path, pred_path):
        init_model = model
        results = []

        for i in range(0, len(x), row_count):
            xi = x[i:i+row_count]
            yi = y[i:i+row_count]
            yhat = self.learn_predict(init_model, xi, yi)
            init_model = model
            results.append(yhat)

            fname = str(xi[0][0])
            self.save_prediction(yi, yhat, pred_path + fname + '.csv')
            self.make_plot(yi, yhat, plt_path + fname + '.png')

        return results

    # Deprecated function
    def reshape_more_data(self, raw, ps, path):
        packetsize = []
        cpu_usage = []

        for i in range(1, 11):
            for j in range(10):
                packetsize.append(ps)
                cpu_usage.append(i*10)

        packetsize = pd.DataFrame(packetsize)
        cpu_usage = pd.DataFrame(cpu_usage)

        init = pd.DataFrame(raw.iloc[:, 0:2].values)
        for i in range(1, len(raw)):
            next = pd.DataFrame(raw.iloc[:, i*2:(i+1)*2].values)
            init = pd.concat([init, next], axis=0)

        init = init.reset_index(drop=True)
        init = pd.concat([init, packetsize, cpu_usage, ], axis=1)
        init.columns = ['bandwidth', 'pps', 'packetsize', 'cpu']

        init.to_csv(path + str(ps) + '.csv')
        return init

    @staticmethod
    def split_train_test(x, y, splitCount):
        kf = KFold(n_splits=splitCount, shuffle=True)
        train_X, test_X, train_Y, test_Y = [], [], [], []

        for train_index, test_index in kf.split(x):
            train_x, test_x = x[train_index], x[test_index]
            train_X.append(train_x), test_X.append(test_x)
            train_y, test_y = y[train_index], y[test_index]
            train_Y.append(train_y), test_Y.append(test_y)

        return train_X, test_X, train_Y, test_Y

    def learn_predict_using_kfold(self, model, splitCount):
        init_model = model
        Y = self.transformed_y.reshape(-1)
        train_X, test_X, train_Y, test_Y = self.split_train_test(self.transformed_x, Y, splitCount)

        yhat_list, y_list, x_list = [], [], []

        for index in range(splitCount):
            init_model.fit(train_X[index], train_Y[index])
            yhat = init_model.predict(test_X[index])
            init_model = model
            yhat_list.append(yhat), y_list.append(test_Y[index]), x_list.append(test_X[index])

        return x_list, y_list, yhat_list

    def make_scalers(self):
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        scaler_x.fit(self.x), scaler_y.fit(self.Y)

        return scaler_x, scaler_y

    def transform(self):
        transformed_x = self.scaler_x.transform(self.x)
        transformed_y = self.scaler_y.transform(self.Y)

        return transformed_x, transformed_y

    def invert(self, x, y, yhat):
        inverted_x = self.scaler_x.inverse_transform(x)

        Y, Yhat = y.reshape(-1, 1), yhat.reshape(-1, 1)
        inverted_y = self.scaler_y.inverse_transform(Y)
        inverted_yhat = self.scaler_y.inverse_transform(Yhat)

        inverted_y, inverted_yhat = inverted_y.reshape(-1), inverted_yhat.reshape(-1)

        return inverted_x, inverted_y, inverted_yhat

