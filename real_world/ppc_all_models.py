import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from jax.config import config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import scienceplots
import jax.numpy as jnp
import numpy as np
import pandas as pd
from nse_kge import nash_sutcliffe, kling_gupta
# https://github.com/DamingoNdiwa/hydrological-model-selection-bayes/blob/main/real_world/ppc_cppp.py
config.update("jax_enable_x64", True)
plt.style.use(['science', 'ieee'])


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

M3 = jnp.load('hpc_scripts/ppc3buc_iris.npy', allow_pickle=True)

# Get the mean predicted values for model M3
mu = jnp.mean(M3, axis=0)

# Get CI (percentiles)
pi = jnp.percentile(M3, jnp.array([25, 75]), axis=0)

# Import the posterior predictve discharge values for model M3

M2 = jnp.load('hpc_scripts/ppc2bucsrea_iris.npy', allow_pickle=True)

# Get the mean predicted values for model M3
mu_M2 = jnp.mean(M2, axis=0)

# Get CI (percentiles)

pi2 = jnp.percentile(M2, jnp.array([25, 75]), axis=0)

M4 = jnp.load('hpc_scripts/ppc4buc06_iris.npy', allow_pickle=True)

# Get the mean predicted values for model M3
mu_M4 = jnp.mean(M4, axis=0)

# Get CI (percentiles)

pi4 = jnp.percentile(M4, jnp.array([25, 75]), axis=0)

# Get the nash-sutcliffe
# Get the kling-gupta

nse = nash_sutcliffe(mu[0], discharge)
kge = kling_gupta(mu[0], discharge)
# code to print the results from nse and kge
print("NSE M3 = ", nse)
print("KGE M3 = ", kge)

nse_M2 = nash_sutcliffe(mu_M2[0], discharge)
kge_M2 = kling_gupta(mu_M2[0], discharge)
print("NSE M2 = ", nse_M2)
print("KGE M2 = ", kge_M2)

nse_M4 = nash_sutcliffe(mu_M4[0], discharge)
kge_M4 = kling_gupta(mu_M4[0], discharge)
print("NSE M4 = ", nse_M4)
print("KGE M4 = ", kge_M4)


legend_elementsm3 = [
    Line2D(
        [0],
        [0],
        ls='-',
        color='b',
        label='Observed discharge',
        alpha=0.6),
    Line2D(
        [0],
        [0],
        ls='-',
        color='r',
        label=r'Model $M_3$',
        alpha=0.6)]


legend_elementsm2 = [
    Line2D(
        [0],
        [0],
        ls='-',
        color='b',
        label='Observed discharge',
        alpha=0.6),
    Line2D(
        [0],
        [0],
        ls='-',
        color='r',
        label=r'Model $M_2$',
        alpha=0.6)]


legend_elementsm4 = [
    Line2D(
        [0],
        [0],
        ls='-',
        color='b',
        label='Observed discharge',
        alpha=0.6),
    Line2D(
        [0],
        [0],
        ls='-',
        color='r',
        label=r'Model $M_4$',
        alpha=0.6)]

# Create a figure
fig = plt.figure()

# Create a GridSpec with 3 rows and 4 columns
gs = GridSpec(3, 4)
# Create the top plot in the middle, spanning half of (0, 1) and (0, 2)
ax2 = fig.add_subplot(gs[0, 1:3])
ax2.plot(df.year.values, mu_M2[0], 'r', alpha=0.6)
ax2.plot(df.year.values, df.observed_discharge, 'b', ls='-', alpha=0.6)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=80))
ax2.set_xticks([0, 21, 43, 66, 87, 108, 129, 150])
ax2.set_xticklabels(['01-01',
                     '22-01',
                     '13-02',
                     '06-03',
                     '27-03',
                     '17-04',
                     '08-05',
                     '29-05'],
                    )
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=12)
ax2.text(113, 22, "NSE = 0.526")
ax2.text(113, 16, "KGE = 0.705")
ax = ax2.twinx()
ax.invert_yaxis()
ax.bar(df.year.values, precipitation, color='g', alpha=0.7)
ax.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=12)
ax2.legend(handles=legend_elementsm2, loc='best', bbox_to_anchor=(
    0.6, -0.25, 0.35, 0.95), fontsize='medium', frameon=True)
ax2.fill_between(df.year.values, pi2[0][0], pi2[1][0],
                 color="k", alpha=0.5, interpolate=False, edgecolor='None')

# Create the bottom-left plot
ax1 = fig.add_subplot(gs[1, 0:2])
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
ax1.text(113, 22, "NSE = 0.526")
ax1.text(113, 16, "KGE = 0.691")
ax = ax1.twinx()
ax.invert_yaxis()
ax.bar(df.year.values, precipitation, color='g', alpha=0.7)
ax.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=12)
ax1.legend(handles=legend_elementsm3, loc='best', bbox_to_anchor=(
    0.6, -0.25, 0.35, 0.95), fontsize='medium', frameon=True)
ax1.fill_between(df.year.values, pi[0][0], pi[1][0],
                 color="k", alpha=0.5, interpolate=False, edgecolor="None")


# Create the bottom-right plot
ax3 = fig.add_subplot(gs[1, 2:4])
ax3.plot(df.year.values, mu_M4[0], 'r', alpha=0.6)
ax3.plot(df.year.values, df.observed_discharge, 'b', ls='-', alpha=0.6)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=80))
ax3.set_xticks([0, 21, 43, 66, 87, 108, 129, 150])
ax3.set_xticklabels(['01-01',
                     '22-01',
                     '13-02',
                     '06-03',
                     '27-03',
                     '17-04',
                     '08-05',
                     '29-05'],
                    )
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=12)
ax = ax3.twinx()
ax.invert_yaxis()
ax.bar(df.year.values, precipitation, color='g', alpha=0.7)
ax.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=12)
ax3.legend(handles=legend_elementsm4, loc='best', bbox_to_anchor=(
    0.6, -0.25, 0.35, 0.95), fontsize='medium', frameon=True)
ax3.fill_between(df.year.values, pi[0][0], pi[1][0],
                 color="k", alpha=0.5, interpolate=False, edgecolor='None')
ax3.text(113, 22, "NSE = 0.421")
ax3.text(113, 16, "KGE = 0.808")


# Adjust the layout
fig.subplots_adjust(wspace=1.39, hspace=0.35)
fig.set_figheight(8.0)
fig.set_figwidth(7.0)
plt.savefig("./ppc234bucsrea.pdf", dpi=100000)
