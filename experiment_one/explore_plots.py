import os
from random import SystemRandom
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import jax
import jax.numpy as jnp
import jax.random as random
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
from hbv import create_joint_posterior
plt.style.use(['science', 'ieee'])


tf = tfp.tf2jax
tfd = tfp.distributions
tfb = tfp.bijectors

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")


# Load data
df = pd.read_pickle("../data/megala_creek_australia.pkl.gz")


# Slice out 1980
df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-12-31')]
df['year'] = df['date'].dt.strftime('%d-%m-%Y')

assert not np.any(np.isnan(df['precipitation']))
assert not np.any(np.isnan(df['evapotranspiration']))

print("Head of dataset")
print(df.head())
print("Tail of dataset")
print(df.tail())

t_start = 0.0  # days
num_days = (df['date'].values[-1] - df['date'].values[0]
            ).astype('timedelta64[D]').astype(int) + 1

# Times to observe solution

T = jnp.float64(num_days)
t_obs = jnp.arange(0, num_days) + 0.5
precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
discharge = jnp.array(df['observed_discharge'], dtype=jnp.float64)


# crete the joit posterior
# For two buckets model

model_prior_params = {
    "n": 2,
    "k": {"loc": jnp.log(jnp.array([2.0, 0.6])),
          "scale": jnp.array([0.25, 0.25, ])},
    "k_int": {"loc": jnp.array([0.8]),
              "scale": jnp.array([0.25])},
    "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
               "scale": tf.cast(0.25, dtype=jnp.float64)},
    "v_max": {"loc": tf.cast(1.0, dtype=jnp.float64),
              "scale": tf.cast(0.25, dtype=jnp.float64)},
    "sigma": {"concentration": tf.cast(5.0, dtype=jnp.float64),
              "scale": tf.cast(0.1, dtype=jnp.float64)},
    "t_obs": t_obs,
    "precipitation": precipitation,
    "evapotranspiration": evapotranspiration
}


dist = create_joint_posterior(model_prior_params)

# Sample from the prior predictive distribution

seed = SystemRandom().randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
key = random.PRNGKey(seed)
key, subkey = jax.random.split(key)
*prior, prior_predictive2 = dist.sample(seed=subkey)

# Set up legend

legend_elements = [
    Line2D(
        [0],
        [0],
        color='r',
        label='mean synthetic discharge',
        alpha=0.5),
    Line2D(
        [0],
        [0],
        color='b',
        ls='-',
        label='Observed discharge (Actual data)'),
    Line2D(
        [0],
        [0],
        color='g',
        label='Precipitation (Actual data)',
        alpha=0.7)]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(df.year.values, prior_predictive2, color='r', alpha=0.5, ls='-')
ax1.plot(df.year.values, discharge, 'b')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=80))
ax1.set_xticks([0, 79, 159, 239, 319, 365])
ax1.set_xticklabels(['01-01', '20-03', '08-06',
                    '27-08', '15-11', '31-12'])
ax1.set_xlabel('Date', fontsize=9)
ax1.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=10)
ax = ax1.twinx()
ax.invert_yaxis()
ax.bar(df.year.values, precipitation, color='g', alpha=0.7)
ax.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=10)
ax1.legend(handles=legend_elements, loc='best', bbox_to_anchor=(
    0.45, -0.08, 0.35, 0.95), fontsize='small', frameon=True)
fig.set_figwidth(6)
plt.savefig("./f366.pdf")

# Slice out first three months of 1980
df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-05-29')]

t_start = 0.0  # days
num_days = (df['date'].values[-1] - df['date'].values[0]
            ).astype('timedelta64[D]').astype(int) + 1

# Times to observe solution
T = jnp.float64(num_days)
t_obs = jnp.arange(0, num_days) + 0.5
precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
discharge = jnp.array(df['observed_discharge'], dtype=jnp.float64)

# crete joint posterior

model_prior_params = {
    "n": 2,
    "k": {"loc": jnp.log(jnp.array([2.0, 0.6])),
          "scale": jnp.array([0.25, 0.25, ])},
    "k_int": {"loc": jnp.array([0.8]),
              "scale": jnp.array([0.25])},
    "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
               "scale": tf.cast(0.25, dtype=jnp.float64)},
    "v_max": {"loc": tf.cast(1.0, dtype=jnp.float64),
              "scale": tf.cast(0.25, dtype=jnp.float64)},
    "sigma": {"concentration": tf.cast(5.0, dtype=jnp.float64),
              "scale": tf.cast(0.1, dtype=jnp.float64)},
    "t_obs": t_obs,
    "precipitation": precipitation,
    "evapotranspiration": evapotranspiration
}


dist = create_joint_posterior(model_prior_params)
# Get samples from prior predictve distribution

*prior, prior_predictive2 = dist.sample(5000, seed=subkey)
mu = jnp.mean(prior_predictive2, axis=0)
pi = jnp.percentile(prior_predictive2, jnp.array([2.5, 97.5]), axis=0)

# for legend
legend_elements = [
    Line2D(
        [0],
        [0],
        color='g',
        lw=0.6,
        label='Precipitation (Actual data)',
        alpha=0.7),
    Line2D(
        [0],
        [0],
        color='b',
        lw=0.6,
        ls='-',
        label='Observed discharge (Actual data)'),
    Line2D(
        [0],
        [0],
        color='r',
        lw=0.6,
        label='Prior predictive mean discharge',
        alpha=0.7),
    Line2D(
        [0],
        [0],
        lw=2,
        color='k',
        alpha=0.5,
        label='95\\% pointwise credible intervals')]

# plot for the first three months

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(df.year.values, mu, color='r', lw=0.6, ls='-')
ax2.plot(df.year.values, discharge, 'b', lw=0.6)
ax2.fill_between(
    df.year.values,
    pi[0],
    pi[1],
    edgecolor="None",
    facecolor="k",
    alpha=0.5,
    label='95\\% pointwise credible intervals')
ax2.tick_params(axis='x')
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
ax2.set_ylabel(r'Discharge ($\mathrm{mmd^{-1}}$)', fontsize=10)
ax3 = ax2.twinx()
ax3.invert_yaxis()
ax3.bar(df.year.values, precipitation, width=0.6, color='g', alpha=0.7)
ax3.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=10)
ax2.set_xlabel('Date')
ax3.legend(
    handles=legend_elements,
    loc='center right',
    fontsize='small',
    borderaxespad=0.5,
    frameon=True)
fig.set_figwidth(5)
plt.savefig("./obsynpre.pdf")

# posterior mean and 95 percent pointwise CI

*prior, prior_predictive = dist.sample(5000, seed=subkey)
mu = jnp.mean(prior_predictive, axis=0)
pi = jnp.percentile(prior_predictive, jnp.array([2.5, 97.5]), axis=0)

legend_elements = [
    Line2D(
        [0],
        [0],
        lw=2,
        color='b',
        alpha=0.5,
        label='95\\% pointwise credible intervals'),
    Line2D(
        [0],
        [0],
        color='k',
        lw=0.6,
        ls='-',
        label='Prior predictive mean discharge'),
    Line2D(
        [0],
        [0],
        color='g',
        lw=0.6,
        label='Precipitation (Actual data)',
        alpha=0.7)]

# Codes for plot on prior prdictive check
# With 95% CI
fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.plot(df.year.values, mu, color='k', lw=0.6, ls='-')
ax3.fill_between(
    df.year.values,
    pi[0],
    pi[1],
    edgecolor="None",
    facecolor="b",
    alpha=0.5,
    label='95\\% pointwise credible intervals')
ax3.set_ylabel(r'Discharge ($\mathrm{mmd^{-1})}$', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.legend(
    handles=legend_elements,
    loc='center right',
    fontsize='medium',
    borderaxespad=0.5,
    frameon=True)
ax4 = ax3.twinx()
ax4.invert_yaxis()
ax4.bar(df.year.values, precipitation, width=0.6, color='g', alpha=0.7)
ax4.set_ylabel(r'Precipitation ($\mathrm{mmd^{-1}}$)', fontsize=10)
ax4.set_xlabel('Date')
ax4.set_xticks([0, 21, 43, 66, 87, 108, 129, 150])
ax4.set_xticklabels(['01-01',
                     '22-01',
                     '13-02',
                     '06-03',
                     '27-03',
                     '17-04',
                     '08-05',
                     '29-05'],
                    )
fig.set_figwidth(6)
plt.savefig('./prior_pc.pdf')
