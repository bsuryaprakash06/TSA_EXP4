# Ex.No: 04 FIT ARMA MODEL FOR TIME SERIES

### Date: 14 - 11 - 2026

# AIM:
To fit and simulate Autoregressive Moving Average (ARMA) models for time series data in Python.

# DATASET:
- **Dataset Name**: Daily Minimum Temperatures in Melbourne (`daily-min-temperatures.csv`)
- **Synthetic Dataset**: Simulated ARMA(1, 1) and ARMA(2, 2) stochastic processes generated using `statsmodels.tsa.arima_process.ArmaProcess`

# ALGORITHM:
1. Import necessary libraries (pandas, numpy, matplotlib, statsmodels).
2. Load the Daily Minimum Temperatures dataset and visualize the raw observations.
3. Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to identify lag orders.
4. Fit ARMA(1, 1) and ARMA(2, 2) models using `statsmodels.tsa.arima.model.ARIMA`.
5. Extract AR and MA coefficients ($\phi, \theta$) and simulate matching stochastic processes using `ArmaProcess`.
6. Plot and evaluate the ACF and PACF for the simulated processes.

# PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

data = pd.read_csv(url)

# Declare required variables
X = data['Temp']
N = len(X)

plt.rcParams['figure.figsize'] = [12, 6]

# Visualize original data
plt.plot(X)
plt.title('Original Data')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.show()

# Plot ACF and PACF
plot_acf(X, lags=50)
plt.title('ACF of Original Data')
plt.show()

plot_pacf(X, lags=50)
plt.title('PACF of Original Data')
plt.show()

# Fitting ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(
    nsample=N*10,
    scale=np.sqrt(arma11_model.scale)
) + X.mean()

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1, lags=40)
plt.title('ARMA(1,1) ACF')
plt.show()

plot_pacf(ARMA_1, lags=40)
plt.title('ARMA(1,1) PACF')
plt.show()

# Fitting ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.arparams[0]
phi2_arma22 = arma22_model.arparams[1]

theta1_arma22 = arma22_model.maparams[0]
theta2_arma22 = arma22_model.maparams[1]

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(
    nsample=N*10,
    scale=np.sqrt(arma22_model.scale)
) + X.mean()

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2, lags=40)
plt.title('ARMA(2,2) ACF')
plt.show()

plot_pacf(ARMA_2, lags=40)
plt.title('ARMA(2,2) PACF')
plt.show()
```

# OUTPUT:

### Original Data
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/ff07dec6-bd09-4bb8-bbaf-727bc442b6b0" />

### Autocorrelation Function and Partial Autocorrelation Function
<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/03634049-c48c-4392-a78e-c9842f90f9a4" />

### Simulated ARMA(1, 1) Process
<img width="990" height="528" alt="image" src="https://github.com/user-attachments/assets/42dfdfd5-15d0-40fb-8cb3-c78bcbfefb78" />

### ARMA(1, 1) ACF
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/666e8fee-69e3-460d-b886-6292344c70cd" />

### ARMA(1, 1) PACF
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/18ead110-09f2-4f34-b7f1-d1e95dc70227" />

### Simulated ARMA(2, 2) Process
<img width="990" height="528" alt="image" src="https://github.com/user-attachments/assets/c741a2a9-9ba4-4bb5-b971-f8120f05650b" />

### ARMA(2, 2) ACF
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/0171e39f-6359-4c84-a661-b3e1ac2a20cf" />

### ARMA(2, 2) PACF
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/530d60c9-a22c-4b78-80fb-f3afba7e7108" />

# RESULT:
Thus, the Python program to fit and simulate ARMA models has been executed successfully.
