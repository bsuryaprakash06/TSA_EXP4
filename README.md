# ARMA and ARIMA Process Modeling and Lag Identification

An end-to-end implementation of Autoregressive Moving Average (ARMA) and Autoregressive Integrated Moving Average (ARIMA) time series modeling, combining diagnostic ACF/PACF identification with parameter estimation and synthetic stochastic simulation.

---

## Objective

Model stationary and non-stationary stochastic time series processes using the Box-Jenkins methodology. The workflow utilizes Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for lag identification, estimates model parameters via Statsmodels, and simulates matching ARMA processes to validate dynamic properties.

---

## Datasets Utilized

1. **Daily Minimum Temperatures in Melbourne (`daily-min-temperatures.csv`)**:
   - **Domain**: Climatology.
   - **Sample**: 3,650 daily observations (1981–1990).
   - **Role**: Empirical reference series used to fit baseline ARMA(1, 1) and ARMA(2, 2) estimators.

2. **Synthetic Stochastic ARMA Processes**:
   - Generated dynamically via `statsmodels.tsa.arima_process.ArmaProcess` with specified autoregressive lag polynomial $\Phi(B)$ and moving average lag polynomial $\Theta(B)$, demonstrating theoretical vs. empirical sampling variability.

---

## Mathematical Formulation

### 1. General ARMA($p, q$) Specification
A mixed autoregressive moving average process combines $p$ autoregressive lags and $q$ moving average shocks:
$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$
where $\epsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$ is Gaussian white noise.

### 2. Lag Order Identification Diagnostics
- **ACF**: Tails off exponentially or sinusoidally for pure AR processes; cuts off sharply after lag $q$ for pure MA processes.
- **PACF**: Cuts off sharply after lag $p$ for pure AR processes; tails off exponentially for pure MA processes.
- For mixed ARMA($p, q$), both ACF and PACF exhibit gradual geometric decay, requiring numerical grid evaluation.

---

## Results & Visualizations

### Empirical Temperature Series
<img width="996" height="547" alt="Original Daily Minimum Temperature Data" src="https://github.com/user-attachments/assets/ff07dec6-bd09-4bb8-bbaf-727bc442b6b0" />

### Autocorrelation & Partial Autocorrelation Diagnostics
<img width="1198" height="590" alt="ACF and PACF Diagnostics" src="https://github.com/user-attachments/assets/03634049-c48c-4392-a78e-c9842f90f9a4" />

### ARMA(1, 1) Simulation & Diagnostics
<img width="990" height="528" alt="Simulated ARMA(1, 1) Process" src="https://github.com/user-attachments/assets/42dfdfd5-15d0-40fb-8cb3-c78bcbfefb78" />

| ARMA(1, 1) ACF | ARMA(1, 1) PACF |
|:---:|:---:|
| <img width="490" alt="ARMA(1, 1) ACF" src="https://github.com/user-attachments/assets/666e8fee-69e3-460d-b886-6292344c70cd" /> | <img width="490" alt="ARMA(1, 1) PACF" src="https://github.com/user-attachments/assets/18ead110-09f2-4f34-b7f1-d1e95dc70227" /> |

### ARMA(2, 2) Simulation & Diagnostics
<img width="990" height="528" alt="Simulated ARMA(2, 2) Process" src="https://github.com/user-attachments/assets/c741a2a9-9ba4-4bb5-b971-f8120f05650b" />

| ARMA(2, 2) ACF | ARMA(2, 2) PACF |
|:---:|:---:|
| <img width="490" alt="ARMA(2, 2) ACF" src="https://github.com/user-attachments/assets/0171e39f-6359-4c84-a661-b3e1ac2a20cf" /> | <img width="490" alt="ARMA(2, 2) PACF" src="https://github.com/user-attachments/assets/530d60c9-a22c-4b78-80fb-f3afba7e7108" /> |

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
