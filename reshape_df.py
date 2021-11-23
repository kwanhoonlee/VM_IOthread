import pandas as pd
from functions import reshape_more_data

data_path = './data/more_data/more_data_raw_ps_'
reshape_path = './data/more_data/more_data_ps_'

ps = [64, 128, 256, 512, 1024]

for i in ps :
    ps_data_path = data_path + str(i) + '.xlsx'
    raw = pd.read_excel(ps_data_path, header=None, index_col=0)
    reshape_more_data(raw, i, reshape_path)



