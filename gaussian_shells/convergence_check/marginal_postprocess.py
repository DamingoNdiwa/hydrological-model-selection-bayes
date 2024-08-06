import numpy as np
import os
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee'])
from scipy.interpolate import make_interp_spline


betas = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
n = len(betas)
marginal_likelihoods = np.zeros(n)

# Iterate only through the available directories
for run in range(n):
    directory = f"output4_{run + 5}"
    results_file = os.path.join(directory, "results.npz")

    # Load the results and extract the marginal likelihood
    with np.load(results_file) as results:
        marginal_likelihoods[run] = results["marginal_likelihood"]
        print(f"beta_value {betas[run]}:  {marginal_likelihoods[run]}")

# Create a DataFrame and save to CSV
mll2 = pd.DataFrame({
    'marginal_likelihood': marginal_likelihoods,
    'beta': betas})

print(mll2)

# Interpolation

mll2.to_csv("marginal_likelihoods2.csv", index=False)
mll = pd.read_csv('./marginal_likelihoods2.csv' )
x_new = np.linspace(betas.min(), betas.max(), 300)
spl = make_interp_spline(betas,  mll['marginal_likelihood'], k=1)
y_smooth = spl(x_new)
fig = plt.figure()
#plt.plot(mll['beta'][:8], mll['marginal_likelihood'][:8], marker='.', linestyle='-', color='b',)
plt.axhline(y=-1.75, color='red', linestyle='--', label='True value')
plt.text(x=-0.8, y=-1.75, s='-1.75', color='red', verticalalignment='bottom')
plt.plot(mll['beta'], mll['marginal_likelihood'], marker='.',linestyle='-',  color='blue',)
plt.plot(mll2['beta'], mll2['marginal_likelihood'], marker='.',linestyle='-',  color='k',)
plt.legend(loc='center')
# Plot the smooth curve
plt.plot(x_new, y_smooth, label='Smooth Curve', color='blue')


# Adding labels and title
plt.xlabel('Number of temperatures ($\\beta$)')
plt.ylabel('Log Marginal Likelihood')
fig.set_figwidth(5.0)
plt.savefig('gauss_converge.pdf', dpi=100000)
