from jax.config import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pymcmcstat.chain.ChainStatistics import geweke, integrated_autocorrelation_time
plt.style.use(['science', 'ieee'])
config.update("jax_enable_x64", True)

# Import the posterior samples
# Example here is for real world discharge
# Diagnostics
parameters4 = [
    r'$K_1$',
    r'$K_2$',
    r'$K_3$',
    r'$K_4$',
    r'$K_{1,2}$',
    r'$K_{2,3}$',
    r'$K_{3,4}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$\hat{V}_3$',
    r'$\hat{V}_4$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']

# Import posterior samples for M4
M4 = pd.read_csv('../realworld/post4bucsrea.csv', names=parameters4)

parameters3 = [
    r'$K_1$',
    r'$K_2$',
    r'$K_3$',
    r'$K_{1,2}$',
    r'$K_{2,3}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$\hat{V}_3$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']

# load posterior samples of model m3
M3 = pd.read_csv('../realworld/post_3reabucs.csv', names=parameters3)

parameters2 = [
    r'$K_1$',
    r'$K_2$',
    r'$K_{1,2}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']

# load data of posterior samples of model m2

M2 = pd.read_csv('../realworld/post_2bucsrea.csv', names=parameters2)

# M4
# If the values are less than N/50, we can confirm the convergence of the
# parameter.
a = np.round(integrated_autocorrelation_time(M4.iloc[1:, :].values), 3)
print(
    f'The integrated autocorrelation time for parameters of M4 is :\n\t{a[0]}')
b = np.round(geweke(M4.iloc[1:, :].values), 3)
# For Geweke, if the p-value is > 0.05, we can confirm convergence
print(f'The p-values of parameters for M4 is :\n\t{b[1]}')

# M3
# If the values are less than N/50, we can confirm the convergence of the
# parameter.
c = np.round(integrated_autocorrelation_time(M3.iloc[1:, :].values), 3)
print(
    f'The integrated autocorrelation time for parameters of M3 is :\n\t{c[0]}')
d = np.round(geweke(M3.iloc[1:, :].values), 3)
# For Geweke, if the p-value is > 0.05, we can confirm convergence
print(f'The p-values of parameters for M3 is :\n\t{d[1]}')

# M2
# If the values are less than N/50, we can confirm the convergence of the
# parameter.
e = np.round(integrated_autocorrelation_time(M2.iloc[1:, :].values), 3)
print(
    f'The integrated autocorrelation time for parameters of M2 is :\n\t{e[0]}')
f = np.round(geweke(M2.iloc[1:, :].values), 3)
# For Geweke, if the p-value is > 0.05, we can confirm convergence
print(f'The p-values of parameters for M2 is :\n\t{f[1]}')
