import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
np.random.seed(42)

date_range = pd.date_range(start="2023-01-01",periods= 100, freq= "D")
trend_data = np.linspace(start =0, stop = 10, num = 100)
season_data = 5 * np.sin(np.linspace(0,2 * np.pi , 100))
noise_data = np.random.normal(0, 1, 100)

data = trend_data + season_data +noise_data    #just input

time_series = pd.DataFrame(data={"Date":date_range,"value":data})
print("-------------time_series-----------------")
print(time_series)

print("___________After Indexing_______________")
time_series.set_index("Date",inplace= True)
print(time_series)

"--------------------Data Decompose______________"

decomposition = seasonal_decompose(time_series,model = "additive")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
print("-----------------trend_______________")
print(trend)
print("----------------seasonal______________")
print(seasonal)

"""--------------------Visualise all these factors-----------------"""

plt.figure(figsize=(12,8))

plt.subplot(411) # 4 rows,1 is columns, 1 is position
plt.plot(time_series, label = "actual data")
plt.legend()

plt.subplot(412)
plt.plot(trend, label = "data trend")
plt.legend()

plt.subplot(413)
plt.plot(trend, label = "data season")
plt.legend()

plt.subplot(414)
plt.plot(trend, label = "data residue")
plt.legend()

plt.show()

"""--------------------Future Forecasting-------------------"""
model = ARIMA(time_series,order= [5,1,0])
fit = model.fit()

forecasting = fit.forecast(steps=10)
forecasting_dates = pd.date_range(start=time_series.index[-1],periods=10,freq="D")  # forecast and this line means next 10 days kulla thu predict panrathu

forecast_df = pd.DataFrame({"Date":forecasting_dates,"Forecast":forecasting})
forecast_df.set_index("Date",inplace=True)

plt.figure(figsize=(10,8))
plt.plot(time_series, label="original")
plt.plot(forecast_df, label = "forecasted data")   # next 10 days la vara data velue
plt.legend()
plt.show()














