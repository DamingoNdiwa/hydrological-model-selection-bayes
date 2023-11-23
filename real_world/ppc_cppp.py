from jax.config import config
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import acf
import matplotlib.dates as mdates
from nse_kge import nash_sutcliffe, kling_gupta
plt.style.use(['science', 'ieee'])
config.update("jax_enable_x64", True)

# Import data for 1980 and select first five months

df = pd.read_pickle("../data/megala_creek_australia.pkl.gz")
df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-05-31')]
df['year'] = df['date'].dt.strftime('%d-%m-%Y')

assert not np.any(np.isnan(df['precipitation']))
assert not np.any(np.isnan(df['evapotranspiration']))

precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
discharge = jnp.array(df['observed_discharge'], dtype=jnp.float64)

# Import the posterior predictve discharge values for model M3

M3 = jnp.load('./ppc3buc06.npy', allow_pickle=True)

# Get the mean predicted values for model M3
mu = jnp.mean(M3, axis=0)

# Get 50% CI(25 & 75 percentiles)
pi = jnp.percentile(M3, jnp.array([25, 75]), axis=0)

# Caculate the posterior predictive pppvalue for model M3
n = len(M3)
M3 = jnp.squeeze(M3)
names = ['a' + str(i) for i in range(n)]
for i in range(len(names)):
    names[i] = acf(
        M3[i],
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
    discharge,
    adjusted=False,
    nlags=n,
    qstat=False,
    fft=True,
    alpha=None,
    bartlett_confint=True,
    missing='none')
ppc = np.count_nonzero(chains >= acf_yobs.reshape(152, 1)
                       ) / (M3.shape[0] * M3.shape[1])
print(f'the ppp-value for model M3 is: {ppc}')

# import data from prior predictive disrtibution
prior_obs = jnp.load('./prior_obs.npy',allow_pickle=True)

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
prior_ppc = np.count_nonzero((priorchains >= acf_yobs.reshape(152,
                                                              1) / M3.shape[0] * M3.shape[1]) <= (chains >= acf_yobs.reshape(152,
                                                                                                                             1)) / (M3.shape[0] * M3.shape[1])) / (M3.shape[0] * M3.shape[1])
print(f'the prior-ppp-value for model M3 is: {prior_ppc}')

# plot of predicted versus observed autocorrelations for M3

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(jnp.mean(chains, 1), acf_yobs, '.')
ax.text(0.01, 0.78, 'posterior predictive $p$-value = 0.533', fontsize=7)
ax.text(
    0.01,
    0.68,
    'prior calibrated posterior predictive $p$-value = 0.660',
    fontsize=7)
ax.set_ylabel('Autocorrelation of replicated discharge', fontsize=7)
ax.set_xlabel('Autocorrelation of observed discharge', fontsize=7)
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
ax.set_xlim(-0.15, 1.1)
fig.set_figwidth(5)
plt.savefig('./plots/ppcm3reauto.pdf')

# Get the nash-sutcliffe
# Get the kling-gupta
nse = nash_sutcliffe(mu[0], discharge)
kge = kling_gupta(mu[0], discharge)

# Make plot of predicted and observed discharge for model M3

legend_elements = [
    Line2D(
        [0],
        [0],
        ls='-',
        color='b',
        label='Obseved discharge',
        alpha=0.67),
    Line2D(
        [0],
        [0],
        ls='-',
        color='r',
        label=r'Model $M_3$',
        alpha=0.67)]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(df.year.values, mu[0], 'r', alpha=0.6)
ax1.plot(df.year.values, df.observed_discharge, 'b', ls='-', alpha=0.6)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=80))
ax1.set_xticks([0, 21, 43, 66, 87, 108, 129, 150])
ax1.set_xticklabels(['01-01',
                     '22-01',
                     '13-02',
                     '06-03',
                     '27-03',
                     '17-04',
                     '08-05',
                     '29-05'],
                    )
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=12)
ax = ax1.twinx()
ax.invert_yaxis()
ax.bar(df.year.values, precipitation, color='g', alpha=0.7)
ax.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=12)
ax1.legend(handles=legend_elements, loc='best', bbox_to_anchor=(
    0.6, -0.25, 0.35, 0.95), fontsize='medium', frameon=True)
ax1.fill_between(df.year.values, pi[0][0], pi[1][0],
                 color="k", alpha=0.4, interpolate=False, edgecolor='k')
ax1.text(118, 22, "NSE = 0.400")
ax1.text(118, 16, "KGE = 0.532")
fig.set_figwidth(5)
plt.savefig("./ppc3bucsrea.pdf")

