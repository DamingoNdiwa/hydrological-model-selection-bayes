from jax.config import config
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import acf


plt.style.use(['science', 'ieee'])


##########################################################
config.update("jax_enable_x64", True)

# load 1980 data and select first three months
Ausdata = pd.read_csv(
    '/Users/damian.ndiwago/Desktop/python/thesis-ideas/TImarginal/Auatraliandata.txt',
    header='infer', sep="\t")

Ausdata['date'] = pd.to_datetime(Ausdata['Year'], format='%d-%m-%Y')

# Selct 1980

data_1980 = Ausdata.loc[(Ausdata['date'] >= '1980-01-01')
                        & (Ausdata['date'] <= '1980-03-31')]


# load the posterior predictve discharge values for model M3

M3 = jnp.load(
    '/Users/damian.ndiwago/Dropbox/Mac/Desktop/python/thesis-ideas/tfponjax/ppcs3bucsrea.npy',
    allow_pickle=True)

# Get the mean predicted values for model M3

M3_mu = jnp.mean(M3, axis=0)


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
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(111)
ax1.plot(
    data_1980.Qobs.values,
    "-",
    color='b',
    label=r"Obseved discharge",
    alpha=0.67)
ax1.plot(
    data_1980.Qobs.values,
    "-",
    color='b',
    label=r"Obseved discharge",
    alpha=0.67)
ax1.plot(M3_mu, "k-", label=r"Model $M_3$", color='r', alpha=0.67)
ax1.set_xticks([0, 11, 22, 33, 44, 55, 66, 77, 91])
ax1.set_xticklabels(['01-01',
                     '11-01',
                     '22-01',
                     '02-02',
                     '13-02',
                     '24-02',
                     '06-03',
                     '17-03',
                     '31-03'],
                    )
ax1.set_ylabel(r'Discharge ($\mathrm{mmday^{-1})}$', fontsize=10)
ax1.set_xlabel('Date')
ax1.legend(handles=legend_elements, loc='best', fontsize='large', frameon=True)
fig.set_figwidth(6)
plt.savefig(
    '/Users/damian.ndiwago/Dropbox/Mac/Desktop/draftpaper/Copernicus/ppcm3.pdf')

# Caculate the posterior predictive pppvalue for model M3
n = len(M3)
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
    data_1980.Qobs.values,
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

# upload data from prior predictive disrtibution
prior_obs = jnp.load(
    '/Users/damian.ndiwago/Dropbox/Mac/Desktop/python/thesis-ideas/tfponjax/prior_obs.npy',
    allow_pickle=True)

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

# Get the prior _cppp
prior_ppc = np.count_nonzero((priorchains >= acf_yobs.reshape(91,
                                                              1) / M3.shape[0] * M3.shape[1]) <= (chains >= acf_yobs.reshape(91,
                                                                                                                             1)) / (M3.shape[0] * M3.shape[1])) / (M3.shape[0] * M3.shape[1])
print(f'the prior-ppp-value for model M3 is: {prior_ppc}')

# plot of predicted versus observed autocorrelations for M3

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.plot(jnp.mean(chains, 1), acf_yobs, '.')
ax.text(0.01, 0.75, 'posterior predictive $p$-value = 0.395', fontsize=7)
ax.text(
    0.01,
    0.68,
    'prior calibrated posterior predictive $p$-value = 0.511',
    fontsize=7)
ax.set_ylabel('Autocorrelation of replicated discharge', fontsize=7)
ax.set_xlabel('Autocorrelation of observed discharge', fontsize=7)
xpoints = ypoints = plt.xlim()
ax.plot(
    xpoints,
    ypoints,
    linestyle='--',
    color='k',
    lw=1,
    scalex=False,
    scaley=False)
fig.set_figwidth(6)
plt.savefig(
    '/Users/damian.ndiwago/Dropbox/Mac/Desktop/draftpaper/Copernicus/ppcm3auto.pdf')




