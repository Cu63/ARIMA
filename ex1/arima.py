from pandas import read_csv, DataFrame
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import itertools
import numpy as np

def calc_d_coef(data):
    test = sm.tsa.adfuller(data)
    print ('adf: ', test[0])
    print ('p-value: ', test[1])
    print('Critical values: ', test[4])
    if (test[1] >= 0.05):
        print ('есть единичные корни, ряд не стационарен')
    else:
        print ('единичных корней нет, ряд стационарен')
        return 0
    d = 1 
    while True:
        print("d: " + str(d))
        data_diff = data.diff(periods=d).dropna()
        test = sm.tsa.adfuller(data_diff)
        print ('adf: ', test[0])
        print ('p-value: ', test[1])
        print('Critical values: ', test[4])
        if (test[1] >= 0.05):
            print ('есть единичные корни, ряд не стационарен')
            d += 1
        else:
            print ('единичных корней нет, ряд стационарен')
            break
    return d

def calc_q_p_coef(data, d):
    lags_count = 25 
    data_diff = data.diff(periods=d).dropna()
    fig = plt.figure()
    data_diff.plot(figsize=(12,6))
    ig = plt.figure(figsize=(12,8))
    ax1 = ig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data_diff.values.squeeze(), lags=lags_count, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data_diff, lags=lags_count, ax=ax2, method='ywm')

    acf = sm.tsa.acf(data_diff.values.squeeze(), nlags=lags_count)
    pacf = sm.tsa.pacf(data_diff.values.squeeze(), nlags=lags_count)
    p = sum(abs(x) >= 0.20 for x in acf)
    q = sum(abs(x) >= 0.20 for x in pacf)
    return q, p


def read_data_from_file(fileName):
    dataset = read_csv(fileName, sep = ',', index_col=['date'], parse_dates=['date'], dayfirst=True)
    data = dataset.data
    data = data.resample('15T').sum()
    create_plot(data)
    return data

def create_plot(d):
    fig = plt.figure()
    d.plot(figsize = (12, 6))

def model(data, p, d, q):
    end = len(data)
    dot = int(end * 0.95)
    src_data_model = data[:dot]
    model = sm.tsa.ARIMA(data, freq='15T', order=(p, d, q)).fit()

    pred = model.predict(dot, end + 10, typ='levels', dynamic = True)
    diff=np.subtract(data[dot:],pred[:len(data)])
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE)
    mape = np.mean(np.abs((data[dot:] - pred[:len(data)])/data[dot:]))*100
    print("ARIMA(%d, %d, %d) %1.2f, mape = %f" % (p, d, q, RMSE, mape))
    q_test = sm.tsa.stattools.acf(model.resid, qstat=True) #свойство resid, хранит остатки модели, qstat=True, означает что применяем указынный тест к коэф-ам
    print(DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]}))
    create_plot(data)
    pred.plot(style='r--', label = "ARIMA(%d, %d, %d)" % (p,d,q))
    plt.legend()
    return pred


#graph = False
graph = True
d = 0
q = 0
p = 0
predicts = []
data = read_data_from_file('96-hours.csv')
#data = read_data_from_file('24-hours.csv')
#data = read_data_from_file('48-hours.csv')
d = calc_d_coef(data)
q, p = calc_q_p_coef(data, d)
print("p = %d, d = %d, q = %d\n" % (p, d, q))
predicts.append(model(data, p, d, q))
predicts.append(model(data, 0, 1, 2))
labeles = ["ARIMA(%d, %d, %d)" % (p,d,q), "ARIMA(0, 1, 2)"]
create_plot(data)
mid_pred = []
for i in range(len(predicts)):
    predicts[i].plot(label = labeles[i])
plt.legend()

mean_pred = predicts[0]
for i in range(len(mean_pred)):
    mean_pred[i] = (mean_pred[i] + predicts[1][i]) / 2

create_plot(data)
mean_pred.plot(label = "mean predictions", style = 'r--')
plt.legend()

dot = int(len(data) / 2)
diff=np.subtract(data[dot:],mean_pred[:len(data)])
square=np.square(diff)
MSE=square.mean()
RMSE=np.sqrt(MSE)
mape = np.mean(np.abs((data[dot:] - mean_pred[:len(data)])/data[dot:]))*100
print("Mean %1.2f, mape = %f" % (RMSE, mape))

if (graph == True):
    plt.legend()
    plt.show()
"""
p = range(0,10)
d = q = range(0,3)
pdq = list(itertools.product(p, d, q))
best_pdq = (0,0,0)
best_aic = np.inf
for params in pdq:
  model_test = sm.tsa.ARIMA(data, order = params)
  result_test = model_test.fit()
  if result_test.aic < best_aic:
    best_pdq = params
    best_aic = result_test.aic
print(best_pdq, best_aic)
"""
