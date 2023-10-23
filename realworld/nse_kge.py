import numpy as np


def nash_sutcliffe(simulated, observed):
    """Calculate the Nash-Sutcliffe efficiency.

    Args:
        simulated: A list of simulated values.
        observed: A list of observed values.

    Returns:
        The Nash-Sutcliffe efficiency, a float between 0 and 1.
    """

    if len(simulated) != len(observed):
        raise ValueError("Simulated and observed arrays must have the same length.")

    mean_observed = sum(observed) / len(observed)
    numerator = sum([(sim - obs)**2 for sim, obs in zip(simulated, observed)])
    denominator = sum([(obs - mean_observed)**2 for obs in observed])

    # Avoid division by zero.
    if denominator == 0:
        return 1

    return 1 - numerator / denominator


def kling_gupta(simulated, observed):
    """Calculate the Kling-Gupta efficiency.

    Args:
        simulated: A NumPy array of simulated values.
        observed: A NumPy array of observed values.

    Returns:
        The Kling-Gupta efficiency, a float between 0 and 1.
    """

    if len(simulated) != len(observed):
        raise ValueError("Simulated and observed arrays must have the same length.")

    sim_mean = np.mean(simulated)
    obs_mean = np.mean(observed)
    sim_std = np.std(simulated)
    obs_std = np.std(observed)
    corr = np.corrcoef(simulated, observed)[0, 1]

    # Avoid division by zero.
    if obs_std == 0:
        return 1

    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean

    return 1 - np.sqrt((alpha - 1)**2 + (beta - 1)**2 - 2 * (alpha - 1) * (beta - 1) * (corr - 1))
