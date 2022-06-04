from pandas import read_csv, DataFrame
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def calc_d_coef(data):
    d = 1

    while True:
        print("d: " + str(d))
        data_diff = data.diff(periods=d).dropna()
        test = sm.tsa.adfuller(data_diff)
        print ('adf: ', test[0])
        print ('p-value: ', test[1])
        print('Critical values: ', test[4])
        if (test[0]> test[4]['5%']):
            print ('есть единичные корни, ряд не стационарен')
        else:
            print ('единичных корней нет, ряд стационарен')
            break
    return d

def calc_q_p_coef(data, d):
    data_diff = data.diff(periods=d).dropna()
    data_diff.plot(figsize=(12,6))
    ig = plt.figure(figsize=(12,8))
    ax1 = ig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data_diff.values.squeeze(), lags=45, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data_diff, lags=45, ax=ax2)

    acf = sm.tsa.acf(data_diff.values.squeeze(), nlags=45)
    pacf = sm.tsa.pacf(data_diff.values.squeeze(), nlags=45)
    q = sum(x == 1 for x in acf)
    p = sum(x == 1 for x in pacf)

    return q, p


def read_data_from_file(fileName):
    dataset = read_csv(fileName, sep = ',', index_col=['date'], parse_dates=['date'], dayfirst=True)
    data = dataset.data
    #resample data for weekends
    #data = data.resample('W').sum()
    return data

def create_plot(d):
    fig = plt.figure()
    d.plot(figsize = (12, 6))

def analys(data):
    itog = data.describe()
    fig = plt.figure()
    data.hist()
    #print(itog)
    #coefficient of variation
    print('V = %f' % (itog['std']/itog['mean']))
    row =  [u'JB', u'p-value', u'skew', u'kurtosis']
    jb_test = sm.stats.stattools.jarque_bera(data)
    a = np.vstack([jb_test])
    itog = SimpleTable(a, row)
    print(itog)

    data_diff = data.diff(periods=1).dropna()
    test = sm.tsa.adfuller(data_diff)
    print ('adf: ', test[0])
    print ('p-value: ', test[1])
    print('Critical values: ', test[4])
    if (test[0]> test[4]['5%']):
        print ('есть единичные корни, ряд не стационарен')
    else:
        print ('единичных корней нет, ряд стационарен')
    """
    #сравнения мат ожидания в для разных промежутков
    m = data_diff.index[int(len(data_diff.index)/2+1)]
    r1 = sm.stats.DescrStatsW(data_diff[m:])
    r2 = sm.stats.DescrStatsW(data_diff[:m])
    print('p-value: ', sm.stats.CompareMeans(r1,r2).ttest_ind()[1])
    create_plot(data_diff)
    """

def model(data, d, q, p):
    end = len(data)
    dot = int(end  * 0.9)
    src_data_model = data[:dot]
    print(len(src_data_model))
    #model = sm.tsa.ARIMA(src_data_model, order=(d, q, p)""", freq='W'""").fit()
    model = sm.tsa.ARIMA(src_data_model, order=(d, q, p)).fit()
    print(model.summary())
    pred = model.predict(dot, end, typ='levels')
    trn = data[dot:]
    print("pred len" + str(len(pred)))
    print("trn len" + str(len(trn)))
    r2 = r2_score(trn, pred[:len(trn)])
    print ('R^2: %1.2f' % r2)
    create_plot(data)
    pred.plot(style='r--')
    print(pred)


#graph = False
graph = True
fig_num = 1
d = 0
q = 0
p = 0
data = read_data_from_file('24-hours1.csv')
d = calc_d_coef(data)
q, p = calc_q_p_coef(data, d)
model(data, d, q, p)
model(data, d, q-1, p)
model(data, d, q-1, p-1)
if (graph == True):
    plt.show()
