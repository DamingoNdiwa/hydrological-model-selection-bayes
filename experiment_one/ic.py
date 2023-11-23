import jax
import jax.numpy as jnp
from jax import grad, vmap
from jax.random import PRNGKey, normal
#from hbv import create_joint_posterior
import tensorflow_probability.substrates.jax as tfp


# Calculate DIC using method 2
def calculate_DIC_2(Dbar, PD_2):
    """
    Calculate DIC using method 2.

    Parameters:
    Dbar (float): Posterior mean of the deviance.
    PD_2 (float): 1/2*(Deviance of the posterior variance) using method 2.

    Returns:
    float: DIC using method 2.
    """
    DIC_2 = Dbar + PD_2
    return DIC_2


# Calculate the effective number of parameters using method 1
def calculate_PD_1(Dbar, Dhat):
    """
    Calculate the effective number of parameters using method 1.

    Parameters:
    Dbar (float): Posterior mean of the deviance.
    Dhat (float): Deviance of the posterior mean.

    Returns:
    float: Effective number of parameters.
    """
    PD_1 = Dbar - Dhat
    return PD_1

# Calculate DIC using method 1


def calculate_DIC_1(Dhat, PD_1):
    """
    Calculate DIC using method 1.

    Parameters:
    Dhat (float): Deviance of the posterior mean.
    PD_1 (float): Effective number of parameters using method 1.

    Returns:
    float: DIC using method 1.
    """
    DIC_1 = Dhat + 2 * PD_1
    return DIC_1

# %%%


def log_pw_pred_density(plpd):
    density = tfp.math.reduce_logmeanexp(plpd)
    return density

# penalty term 1


def PWAIC_1(plpd):
    lppd = log_pw_pred_density(plpd)
    plpd = jnp.mean(lppd)
    diff = lppd - plpd
    samples4 = 2 * jnp.sum(diff)
    return samples4

# penalty term 2


def PWAIC_2(plpd):
    PWAIC2 = jnp.var(plpd, axis=0)
    return jnp.sum(PWAIC2)


def WAIC_1(plpd):
    A = -2 * log_pw_pred_density(plpd)
    B = 2 * PWAIC_1(plpd)
    WAIC1 = A + B
    return WAIC1

# WAIC based on the second penalty term


def WAIC_2(plpd):
    A = -2 * log_pw_pred_density(plpd)
    B = 2 * PWAIC_2(plpd)
    WAIC2 = A + B
    return WAIC2
