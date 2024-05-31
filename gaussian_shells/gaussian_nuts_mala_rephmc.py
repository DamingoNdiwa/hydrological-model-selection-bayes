
import logging
from tensorflow_probability.substrates.jax.internal import tensor_util
from tensorflow_probability.substrates.jax.internal import prefer_static as ps
from tensorflow_probability.substrates.jax.internal import parameter_properties
from tensorflow_probability.substrates.jax.internal import dtype_util
from tensorflow_probability.substrates.jax.distributions import distribution
import tensorflow_probability.substrates.jax as tfp
import numpy as np
from jax import config, jit, random
from jax import jit, random
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.numpy import log, sqrt, pi
from jax.random import PRNGKey
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])
config.update("jax_enable_x64", True)
tf = tfp.tf2jax
tfd = tfp.distributions

# imports to avoid warnings


logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


# Gaussian shell log-likelihood

def loglikelihood(theta1, theta2, c1=tf.constant(
        [-3.5, 0.]), c2=tf.constant([3.5, 0.])):
    theta = tf.stack([theta1, theta2], -1)

    def logcirc(theta, c, r=tf.constant([2.0]), w=tf.constant([0.1])):
        logll = -0.5 * (tf.linalg.norm(theta - c, axis=-1) - r)**2 / \
            w**2 - tf.math.log(tf.sqrt(2 * np.pi * w**2))
        return logll
    return tf.math.reduce_logsumexp([logcirc(theta, c1), logcirc(theta, c2)])


class GaussianShell(distribution.AutoCompositeTensorDistribution):

    def __init__(self,
                 theta1,
                 theta2,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='GaussianShell'):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype(
                [theta1, theta2], dtype_hint=tf.float32)
            self._theta1 = tensor_util.convert_nonref_to_tensor(
                theta1, dtype=dtype, name='theta1')
            self._theta2 = tensor_util.convert_nonref_to_tensor(
                theta2, dtype=dtype, name='theta2')
            super(GaussianShell, self).__init__(
                dtype=dtype,
                reparameterization_type=tfd.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            theta1=parameter_properties.ParameterProperties(),
            theta2=parameter_properties.ParameterProperties())

    @property
    def theta1(self):
        """Distribution parameter for the mean."""
        return self._theta1

    @property
    def theta2(self):
        """Distribution parameter for the mean."""
        return self._theta2

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        theta1 = tf.convert_to_tensor(self.theta1)
        theta2 = tf.convert_to_tensor(self.theta2)

        shape = ps.concat([[n], self._batch_shape_tensor(
            theta1=theta1, theta2=theta2)], axis=0)
        sampled = tf.zeros(shape=shape, dtype=self.dtype)

        return sampled

    def _log_prob(self, x):
        theta1 = tf.convert_to_tensor(self.theta1)
        theta2 = tf.convert_to_tensor(self.theta2)
        return loglikelihood(theta1, theta2)


def model():

    theta1 = yield tfd.Sample(tfd.Uniform(tf.cast(-6.0, dtype='float64'), tf.cast(6.0, dtype='float64')), sample_shape=(1), name='theta1')
    theta2 = yield tfd.Sample(tfd.Uniform(tf.cast(-6.0, dtype='float64'), tf.cast(6.0, dtype='float64')), sample_shape=(1), name='theta2')

    yield tfd.Sample(GaussianShell(theta1, theta2), name="GaussianShell")


dist = tfd.JointDistributionCoroutineAutoBatched(model)
dist.sample(seed=PRNGKey(0))

posterior = dist.experimental_pin(GaussianShell=1)


def make_power_log_prob_fn():
    @tf.function()
    def power_log_prob_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        log_prob = log_prob_parts.pinned[0] + \
            sum(list(log_prob_parts.unpinned))
        return log_prob

    return power_log_prob_fn


power_log_prob_fn = make_power_log_prob_fn()

num_results = int(5000)
num_burnin_steps = int(500)

bij = posterior.experimental_default_event_space_bijector()
pulled_back_shape = bij.inverse_event_shape(posterior.event_shape)
current_state = posterior.sample_unpinned(seed=PRNGKey(0))

# define kernel for the sampler
# NUTS

sampler = tfp.mcmc.TransformedTransitionKernel(
    tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=power_log_prob_fn,
        step_size=0.1),
    bijector=bij)

adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=sampler,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=0.75)


def run_chain():
    """ samples from posterior
    """
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=adaptive_sampler,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None,
        seed=PRNGKey(0)
        # num_steps_between_results=1,
    )


post = run_chain()

fig, ax = plt.subplots()
plt.plot(post.theta1[:, 0][::25], post.theta2[:, 0][::25], 'b.')
ax.set_xlabel(r'$\theta_2$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
fig.set_figwidth(4)
fig.savefig("./nutsshell.pdf")

# Replica excahnge hmc


def make_power_log_prob_fn():
    @jit
    def power_log_prob_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        log_prob = log_prob_parts.pinned[0] + \
            sum(list(log_prob_parts.unpinned))
        return log_prob

    return power_log_prob_fn


power_log_prob_fn = make_power_log_prob_fn()

# unormalized target logpdf


def make_power_log_prob_fn():
    @jit
    def power_log_prob_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        log_prob = log_prob_parts.pinned[0] + \
            sum(list(log_prob_parts.unpinned))
        return log_prob

    return power_log_prob_fn


power_log_prob_fn = make_power_log_prob_fn()

init_key, sample_key = random.split(PRNGKey(5))

current_state = posterior.sample_unpinned(1, seed=init_key)


# Get temperature values
k = 50  # 35  # k is the number of temperatures
num_betas = k
num_chains = 1
inverse_temperatures = jnp.flip(jnp.asarray(a=[pow((i / (k)), 5)
                                               for i in range(1, k + 1)]))

# sample from the prior as initial values
# any seed works
init_key, sample_key = random.split(PRNGKey(1255))
current_state = posterior.sample_unpinned(num_chains, seed=init_key)

# Initialize the HMC transition kernel.
num_results = int(5000)
num_burnin_steps = int(500)

# Run the chain (with burn-in).


def make_kernel_fn(target_log_prob_fn):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=30,
        step_size=0.005)
    return tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))


remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=power_log_prob_fn,
    inverse_temperatures=inverse_temperatures,
    make_kernel_fn=make_kernel_fn)


def trace_swaps(unused_state, results):
    """ results to trace """

    return (  # results.inverse_temperatures,
        results.post_swap_replica_states)
   # results.is_swap_accepted)


@jit
def run_chain(key):
    """ samples from posterior
    """
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=remc,
        num_burnin_steps=num_burnin_steps,
        trace_fn=trace_swaps,
        seed=key)


post1, post_all = run_chain(sample_key)

fig, ax = plt.subplots()
plt.plot(post1.theta1[:, 0][::25], post1.theta2[:, 0][::25], 'b.')
ax.set_xlabel(r'$\theta_2$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
fig.set_figwidth(4)
fig.savefig("./rehmc.pdf")

# MALA


def run_chain():
    """ samples from posterior
    """
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=power_log_prob_fn,
            step_size=0.01),
        num_burnin_steps=num_burnin_steps,
        trace_fn=None,
        seed=PRNGKey(0)
    )


post2 = run_chain()
fig, ax = plt.subplots()
plt.plot(post2.theta1[:, 0][::25], post2.theta2[:, 0][::25], 'b.')

ax.set_xlabel(r'$\theta_2$', fontsize=12)
ax.set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
fig.set_figwidth(4)
fig.savefig("./mala.pdf")


def loglikelihood(theta, c1=jnp.array([-3.5, 0.]), c2=jnp.array([3.5, 0.])):
    def logcirc(theta, c, r=jnp.array(2.0), w=jnp.array(0.1)):
        logll = -0.5 * (norm(theta - c, axis=-1) - r)**2 / \
            w**2 - log(sqrt(2 * pi * w**2))
        return logll
    return jnp.logaddexp(logcirc(theta, c1), logcirc(theta, c2))


# Plot PDF contours.
x = jnp.linspace(-6., 6., int(1e4), dtype=jnp.float32)


def meshgrid(x, y=x):
    [gx, gy] = jnp.meshgrid(x, y, indexing='ij')
    gx, gy = jnp.float32(gx), jnp.float32(gy)
    grid = jnp.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)


grid = meshgrid(jnp.linspace(-6, 6, 100, dtype=jnp.float32))

fig, ax = plt.subplots()
ab = ax.contourf(grid[..., 0], grid[..., 1], loglikelihood(grid))
fig.colorbar(ab, shrink=0.8)
ax.set_xlabel(r'$\theta_2$')
ax.set_ylabel(r'$\theta_1$', rotation='horizontal')

# subplots

fig, axs = plt.subplots(2, 2)

# Plot data in each subplot
axs[0, 0].plot(post.theta1[:, 0][::25], post.theta2[:, 0][::25], 'b.')
axs[0, 0].set_xlabel(r'$\theta_2$', fontsize=12)
axs[0, 0].set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
axs[0, 0].tick_params(axis='both', labelsize=12)
axs[0, 1].plot(post1.theta1[:, 0][::25], post1.theta2[:, 0][::25], 'b.')
axs[0, 1].set_xlabel(r'$\theta_2$', fontsize=12)
axs[0, 1].set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
axs[0, 1].tick_params(axis='both', labelsize=12)

axs[1, 0].plot(post2.theta1[:, 0][::25], post2.theta2[:, 0][::25], 'b.')
axs[1, 0].set_xlabel(r'$\theta_2$', fontsize=12)
axs[1, 0].set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
ax.tick_params(axis='both', labelsize=12)


ab = axs[1, 1].contourf(grid[..., 0], grid[..., 1], loglikelihood(grid))
axs[1, 1].set_xlabel(r'$\theta_2$', fontsize=12)
axs[1, 1].set_ylabel(r'$\theta_1$', rotation='horizontal', fontsize=12)
fig.colorbar(ab)

# Labels for the subplots
labels = ['(a)', '(b)', '(c)', '(d)']
positions = [(-0.2, 1.2), (-0.1, 1.2), (-0.2, 1.2), (-0.1, 1.2)]

# Add the labels to the subplots in bold
for ax, label, pos in zip(axs.flat, labels, positions):
    ax.text(
        pos[0],
        pos[1],
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='right')

# Adjust layout to make space for the labels
fig.subplots_adjust(wspace=0.5, hspace=0.8)
fig.set_figheight(4)
fig.set_figwidth(5)
fig.savefig("./rehmc_nuts_mala.pdf")

