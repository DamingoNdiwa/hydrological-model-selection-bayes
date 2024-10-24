import tensorflow_probability.substrates.jax as tfp
from statsmodels.tsa.stattools import acf
from jax.config import config
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(['science', 'ieee'])
config.update("jax_enable_x64", True)

tf = tfp.tf2jax

# Load data
df = pd.read_pickle("../../data/megala_creek_australia.pkl.gz")

# Slice out first three months of 1980
df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-03-31')]
df['year'] = df['date'].dt.strftime('%d-%m-%Y')

assert not np.any(np.isnan(df['precipitation']))
assert not np.any(np.isnan(df['evapotranspiration']))

precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
discharge = jnp.array(df['observed_discharge'], dtype=jnp.float64)

## Import synthetic discharge for M3
y_obs = jnp.load('hpc_scripts/data_1980.npy', allow_pickle=True)

# load the posterior predictve discharge values
# For M2

M2 = jnp.load('hpc_scripts/ppcstudytwo2bucs.npy', allow_pickle=True)

# For M3
M3 = jnp.load('hpc_scripts/ppcstudytwo3bucs.npy', allow_pickle=True)

# For M4
M4 = jnp.load('hpc_scripts/ppcstudytwo4bucs.npy', allow_pickle=True)

# import prior predictive samples
prior_obs = jnp.load('hpc_scripts/prior_obsm3.npy', allow_pickle=True)

# get the mean of the posterior predictive discharge values

M22_mu = jnp.mean(M2, axis=0)
M32_mu = jnp.mean(M3, axis=0)
M42_mu = jnp.mean(M4, axis=0)

# Legend for plots

legend_elements = [
    Line2D(
        [0],
        [0],
        ls='-',
        color='r',
        lw=0.5,
        label='Synthetic  discharge',
        alpha=0.5),
    Line2D(
        [0],
        [0],
        color='b',
        lw=0.8,
        ls='--',
        label=r'$M_2$'),
    Line2D(
        [0],
        [0],
        color='k',
        lw=0.8,
        ls='-.',
        label=r'$M_3$',
        alpha=0.8),
    Line2D(
        [0],
        [0],
        ls=':',
        color='m',
        label=r'$M_4$',
        alpha=0.8)]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(y_obs[0], color='r', lw=0.5, alpha=0.5, ls='-')
ax1.plot(jnp.squeeze(M22_mu), color='b', lw=0.8, ls='--')
ax1.plot(jnp.squeeze(M32_mu), color='k', lw=0.8, alpha=0.8, ls='-.')
ax1.plot(jnp.squeeze(M42_mu), color='m', alpha=0.8, ls=':')
ax1.set_xlabel('time (day)', fontsize=10)
ax1.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=10)
ax1.legend(handles=legend_elements, loc='best', fontsize='small', frameon=True)
fig.set_figwidth(5)
plt.savefig("ppc_studytwo.pdf")

# Caculate the posterior predictive pppvalue for model M2
n = len(M3)
M2 = jnp.squeeze(M3)
names = ['a' + str(i) for i in range(n)]
for i in range(len(names)):
    names[i] = acf(
        M2[i],
        adjusted=False,
        nlags=n,
        qstat=False,
        fft=True,
        alpha=None,
        bartlett_confint=True,
        missing='none')
chains = np.column_stack((names))
pd.DataFrame(chains)
acf_yobs = acf(
    y_obs[0],
    adjusted=False,
    nlags=n,
    qstat=False,
    fft=True,
    alpha=None,
    bartlett_confint=True,
    missing='none')
ppc = np.count_nonzero(chains >= acf_yobs.reshape(91, 1)
                       ) / (M3.shape[0] * M3.shape[1])
print(f'the ppp-value for model M3 is: {ppc}')

# calculate the prior calibrated posterior predictive pvalue

n = len(prior_obs)
names = ['a' + str(i) for i in range(n)]
for i in range(len(names)):
    names[i] = acf(
        prior_obs[i],
        adjusted=False,
        nlags=n,
        qstat=False,
        fft=True,
        alpha=None,
        bartlett_confint=True,
        missing='none')
priorchains = np.column_stack((names))

# Get the prior_cppp
prior_ppc = np.count_nonzero((priorchains >= acf_yobs.reshape(91,
                                                              1) / M3.shape[0] * M3.shape[1]) <= (chains >= acf_yobs.reshape(91,
                                                                                                                             1)) / (M3.shape[0] * M3.shape[1])) / (M3.shape[0] * M3.shape[1])
print(f'the prior-ppp-value for model M2 is: {prior_ppc}')

# Plot of Autocor
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(jnp.mean(chains, 1), acf_yobs, '.')
ax.text(0.01, 0.75, 'posterior predictive $p$-value = 0.568', fontsize=7)
ax.set_ylabel('Autocorrelation of replicated discharge', fontsize=7)
ax.set_xlabel('Autocorrelation of synthetic discharge', fontsize=7)
xpoints = ypoints = plt.xlim()
ax.plot(
    xpoints,
    ypoints,
    linestyle='--',
    color='k',
    lw=0.5,
    scalex=False,
    scaley=False)
fig.set_figwidth(6)
ax.set_ylim(-0.2, 1.1)
ax.set_xlim(-0.2, 1.1)
fig.set_figwidth(5)
plt.savefig('./ppcm3auto.pdf')
