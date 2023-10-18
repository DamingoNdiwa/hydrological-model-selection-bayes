import os
from random import SystemRandom
from matplotlib.lines import Line2D
import seaborn as sns
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp

from hbv import create_joint_posterior

tf = tfp.tf2jax
tfd = tfp.distributions
tfb = tfp.bijectors

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

plt.style.use(['science', 'ieee'])


# Load data
df = pd.read_pickle("./megala_creek_australia.pkl.gz")

# Slice out first three months of 1980

df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-03-31')]
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

# Studyone
# crete the joint posterior
# For two buckets model

model_prior_params = {
    "n": 2,
    "k": {"loc": jnp.log(jnp.array([1.0, 0.6])),
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
print(seed)
key = random.PRNGKey(seed)
key, subkey = jax.random.split(key)
*prior, prior_predictive2 = dist.sample(5000, seed=subkey)

# process prior samples for plotting

num_params = 5
names = ['a' + str(i) for i in range(num_params)]
for i in range(len(names)):
    names[i] = jnp.squeeze(prior[i])

chains = np.column_stack((names))

prior1 = pd.DataFrame(
    chains,
    columns=[
        r'$k_1$',
        r'$k_2$',
        r'$k_{1,2}$',
        r'$\hat{V}_1$',
        r'$\hat{V}_2$',
        r'$V_{\mathrm{max}}$',
        r'$\sigma^2$'])
prior1['Distribution'] = 'Prior'

# load posterior samples
postone_2bucs = pd.read_csv(
    '/Users/damian.ndiwago/Desktop/python/thesis-ideas/tfponjax/hpctranfers/studyone/post1_2bucs.csv',
    names=[
        r'$k_1$',
        r'$k_2$',
        r'$k_{1,2}$',
        r'$\hat{V}_1$',
        r'$\hat{V}_2$',
        r'$V_{\mathrm{max}}$',
        r'$\sigma^2$'])
a = postone_2bucs[1:]
a['Distribution'] = 'Posterior'

# combine the prior samples and posterior samples into one data set
a = pd.concat([a, prior1], ignore_index=True)

# Studyone
# Posterior and  prior distributions for two buckets model

fig = plt.figure()
sns.set_context("paper",
                rc={'font.size': 35,
                    'legend.title_fontsize': 20,
                    'axes.labelsize': 35,
                    'xtick.labelsize': 27,
                    'ytick.labelsize': 27,
                    'xtick.major.size': 0.0,
                    'ytick.major.size': 0.0,
                    'ytick.minor.size': 0.0,
                    'xtick.minor.size': 0.0})
g = sns.PairGrid(a[1::30],
                 hue='Distribution',
                 palette={'Posterior': 'r',
                          'Prior': 'blue'},
                 diag_sharey=False,
                 corner=True,
                 despine=True)
g.map_lower(sns.kdeplot, common_norm=False, fill=False, levels=10)
g.map_diag(sns.kdeplot, ls='-', color='r', lw=1, common_norm=False)
data_true = a[:1]
true = 1.45, 0.25, 3.23, 1.08, 0.81, 2.52, 0.014, 'Prior'
a.loc[0] = true
g.data = data_true
handles = [Line2D([], [], color='blue', ls='-', label='Prior'),
           Line2D([], [], color='red', ls='-', label='Posterior')]
g.add_legend(handles=handles, frameon=True, fontsize='small')
g.figure.set_figwidth(18)
g.figure.savefig('./studyone2bucs_priorposte.pdf', dpi=500)

# Studyone
# Posterior plots for two buckets model with true value
fig = plt.figure()
sns.set_context("paper",
                rc={'font.size': 35,
                    'legend.title_fontsize': 20,
                    'axes.labelsize': 35,
                    'xtick.labelsize': 27,
                    'ytick.labelsize': 27,
                    'xtick.major.size': 0.0,
                    'ytick.major.size': 0.0,
                    'ytick.minor.size': 0.0,
                    'xtick.minor.size': 0.0})
g = sns.PairGrid(postone_2bucs[1::90],
                 diag_sharey=False,
                 corner=True,
                 despine=True)
g.map_lower(sns.kdeplot, color='blue', common_norm=False, fill=True)
g.map_diag(sns.kdeplot, color='blue', ls='-', lw=1, common_norm=False)

data_true = postone_2bucs[:1]
true = 1.454, 0.248, 3.23, 1.08, 0.81, 2.52, 0.014
data_true.loc[0] = true
g.data = data_true
g.map_lower(sns.scatterplot, color='k', zorder=3, s=100)
handles = [Line2D([], [], color='k', ls='', marker='o', label='True value')]
g.add_legend(handles=handles, fontsize='medium', frameon=True, markerscale=1.5)
g.figure.set_figwidth(16)
g.fig.autofmt_xdate(rotation=45)
g.figure.savefig('./studyone2bucs.pdf', dpi=500)
