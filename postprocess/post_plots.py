import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use(['science', 'ieee'])


# parameter names to used for plots in model m4

parameters4 = [
    r'$k_1$',
    r'$k_2$',
    r'$k_3$',
    r'$k_4$',
    r'$k_{1,2}$',
    r'$k_{2,3}$',
    r'$k_{3,4}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$\hat{V}_3$',
    r'$\hat{V}_4$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']

# load data of posterior samples for model M4
postsamples4 = pd.read_csv('../realworld/post4bucsrea.csv', names=parameters4)

fig = plt.figure()


def pairplot(postdata):
    """function to make pairplots for model m4 and save

    Args:
        postdata (data): posterior samples in a pandas datarame

    Returns:
        pair plots
    """
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
    g = sns.PairGrid(postdata[1:], diag_sharey=False,
                     corner=True, despine=True)
    g.map_lower(sns.kdeplot, color='blue', common_norm=False, fill=True)
    g.map_diag(sns.kdeplot, color='blue', ls='-', lw=1, common_norm=False)
    #g.set_figwidth(20)
    g.figure.set_figwidth(21)
    g.fig.autofmt_xdate(rotation=45)  # Rotate x-tick labels
    pairplot = g.savefig('./4pairplots4bucs.pdf', dpi=500)
    plt.show()
    return pairplot


# correlation plot
# call function to give pair plot
pairplot(postsamples4[1::3])
postsamples4[1::3]

def corplot(postdata):
    """fucntion to make  heat map showing corellation between parameters and save

    Args:
        postdata (_type_): posterior samples in a pandas dataframe

     Returns:
        heat map
    """
    sns.set_context("paper",
                    rc={'font.size': 30,
                        'legend.title_fontsize': 20,
                        'axes.labelsize': 40,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'xtick.major.size': 0.0,
                        'ytick.major.size': 0.0,
                        'ytick.minor.size': 0.0,
                        'xtick.minor.size': 0.0})
    rho = postdata.corr(method="spearman")
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(rho, dtype=bool))
    np.fill_diagonal(mask, False)
    heatmap = sns.heatmap(postdata.corr(), mask=mask, vmin=-1,
                          vmax=1, annot=False)
    plt.savefig('./corplots4bucs.pdf')
    return(heatmap)


corplot(postsamples4[1::3])

# Parameters names of model m3

parameters3 = [
    r'$k_1$',
    r'$k_2$',
    r'$k_3$',
    r'$k_{1,2}$',
    r'$k_{2,3}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$\hat{V}_3$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']


postsamples3 = pd.read_csv('../realworld/post_3reabucs.csv', names=parameters3)

# Make pair plots of model m3
fig = plt.figure()


def pairplot(postdata):
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
    g = sns.PairGrid(postdata[1:], diag_sharey=False,
                     corner=True, despine=True)
    g.map_lower(sns.kdeplot, color='blue', common_norm=False, fill=True)
    g.map_diag(sns.kdeplot, color='blue', ls='-', lw=1, common_norm=False)
    g.figure.set_figwidth(20)
    g.fig.autofmt_xdate(rotation=45)  # Rotate x-tick labels
    pairplot = g.figure.savefig('./pairplots3bucs.pdf')
    return pairplot

# call function to make pair plots of model m3
pairplot(postsamples3[1::3])

# function to make heat map of correlations of model m3
def corplot(postdata):
    sns.set_context("paper",
                    rc={'font.size': 30,
                        'legend.title_fontsize': 20,
                        'axes.labelsize': 40,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'xtick.major.size': 0.0,
                        'ytick.major.size': 0.0,
                        'ytick.minor.size': 0.0,
                        'xtick.minor.size': 0.0})
    rho = postdata.corr(method="spearman")
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(rho, dtype=bool))
    np.fill_diagonal(mask, False)
    heatmap = sns.heatmap(postdata.corr(), mask=mask, vmin=-1,
                          vmax=1, annot=False)
    plt.savefig('./corplots3bucs.pdf')
    return(heatmap)


corplot(postsamples3[1::3])

# parmeters of model m2
parameters2 = [
    r'$K_1$',
    r'$K_2$',
    r'$K_{1,2}$',
    r'$\hat{V}_1$',
    r'$\hat{V}_2$',
    r'$V_{\mathrm{max}}$',
    r'$\sigma^2$']


# load data of posterior samples of model m2
postsamples2 = pd.read_csv('../realworld/post_2bucsrea.csv', names=parameters2)


# function to make pairplots of posterior parameters of model m2
fig = plt.figure(figsize=(10, 10))


def pairplot(postdata):
    sns.set_context("paper",
                    rc={'font.size': 30,
                        'legend.title_fontsize': 20,
                        'axes.labelsize': 30,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'xtick.major.size': 0.0,
                        'ytick.major.size': 0.0,
                        'ytick.minor.size': 0.0,
                        'xtick.minor.size': 0.0})
    g = sns.PairGrid(postdata[1:], diag_sharey=False,
                     corner=True, despine=True)
    g.map_lower(sns.kdeplot, color='blue', common_norm=False, shade=True)
    g.map_diag(sns.kdeplot, color='blue', ls='-', lw=1, common_norm=False)
    g.figure.set_figwidth(20)
    pairplot = g.figure.savefig('./pairplots2bucs.pdf')
    return pairplot


# call function to make pair plots of model m2
pairplot(postsamples2[1::3])

# function to make heat map of correlations of model m2

def corplot(postdata):
    sns.set_context("paper",
                    rc={'font.size': 30,
                        'legend.title_fontsize': 20,
                        'axes.labelsize': 40,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'xtick.major.size': 0.0,
                        'ytick.major.size': 0.0,
                        'ytick.minor.size': 0.0,
                        'xtick.minor.size': 0.0})

    rho = postdata.corr(method="spearman")
    mask = np.triu(np.ones_like(rho, dtype=bool))
    np.fill_diagonal(mask, False)
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(postdata.corr(), mask=mask, vmin=-1,
                          vmax=1, annot=False)
    plt.savefig('./corplots2bucs.pdf')
    return(heatmap)


corplot(postsamples2[1::3])

