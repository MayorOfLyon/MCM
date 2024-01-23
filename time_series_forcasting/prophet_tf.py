import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

data = pd.read_excel('./data/2023-C.xlsx', header=1, usecols=['Date', 'Number of  reported results'])
data = data.rename(columns={'Date': 'ds', 'Number of  reported results': 'y'})
print(data.head())
model = Prophet()
# seasonality_mode='multiplicative' 乘法模型
model.add_seasonality(name='weekly', period=7, fourier_order=3)

model.fit(data)

future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

fig = model.plot(forecast)
fig = model.plot_components(forecast)
plt.show()

