import pandas as pd
from prophet import Prophet
# from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import matplotlib.pyplot as plt


# Example DataFrame
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=100),
    'y': (np.sin(np.arange(100) / 20) * 50 + 50)  # Just an example time series
})


# Initialize the Model
model = Prophet()

# Fit the Model
model.fit(df)

# Create future DataFrame for 365 days into the future
future = model.make_future_dataframe(periods=365)

# Predict future values
forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.show()



plt.show()
