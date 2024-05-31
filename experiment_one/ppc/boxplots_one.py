import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import scienceplots
from jax.config import config
config.update("jax_enable_x64", True)
plt.style.use(['science', 'ieee'])

# Studyone
# The results were obtained by running each model many times in parallel with different initial values
# The shell files provided can modified to run the models any number of
# times on the HPC cluster, and the results post-processed

M2 = jnp.array([217.11942891,
                211.54192398,
                220.38517574,
                217.8559804,
                219.40133844,
                218.32246743,
                221.87289074,
                220.33747949,
                212.26055039,
                218.20161733,
                218.97039584,
                214.75599933,
                220.30097001,
                220.22356919])
M3 = jnp.array([203.90088079,
                197.49796082,
                200.71137814,
                205.23254613,
                205.01385101,
                197.65966946,
                203.1943469,
                205.04807065,
                205.41267808,
                206.58096952,
                204.89830478,
                203.13865718,
                207.01581276,
                202.06097827])
M4 = jnp.array([156.41658202,
                154.75506303,
                152.28116095,
                150.93625674,
                154.18756919,
                162.98726646,
                153.17271283,
                152.56751221,
                161.20255632,
                150.15877972,
                153.77402571,
                159.90696698,
                146.62290123,
                155.33278217,
                157.22441298])

print(f'the mean log-marginal likelihood for M2 is : {round(jnp.mean(M2),3)}')
print(f'the mean log-marginal likelihood for M3 is : {round(jnp.mean(M3),3)}')
print(f'the mean log-marginal likelihood for M4 is : {round(jnp.mean(M4),3)}')

fig = plt.figure()
ax1 = fig.add_subplot(111)
columns = [r'$M_2, M_3, M_4$']
ax1.boxplot([M2, M3, M4], patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax1.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax1.set_ylabel('Log marginal likelihood', fontsize=10)
ax1.set_xlabel('Models', fontsize=10)
fig.set_figwidth(4)
plt.savefig("./logmastudy_one.pdf")

# The DIC and WAIC for the models
# Obtained by running the models many times with different initial values
# randomly initialised

ex1_2bucs_Dic = [-521.4168531064525, -519.8907041537145, -522.1232007253536, -520.7779066970872, -522.3434528437772, -
                 521.4885209246132, -521.522425525855, -521.5712812826637, -519.9841813612745] # -520.0262001216167, -520.6250593086656 ]
ex1_2bucs_waic = [-516.9418449820848, -512.417513798562, -515.1644932755582, -512.6830242385932, -512.81456493386, -
                  516.3736679602113, -515.0415880797789, -513.2524941107545, -514.4930345335346] # -512.1583592848139, -514.5198176232221  ]
ex1_3buc_waic = [-484.29136345088835, -507.2081663240031, -500.63822951104504, -482.6559787682421, -
                 507.44156245897176, -507.12586056656284, -507.4204013870319, -507.21527556118326, -511.18165104016646]
ex1_3buc_dic = [-511.7011742568445, -516.7724687534817, -516.9705562563923, -505.258648580398, - \
    509.9255382538677, -511.20442320520516, -512.2913083497572, -512.2909071979564, -517.3579821868952]
ex1_4buc_waic = [-438.49343604184133, -431.69282943047705, -428.4478385770345, -451.6535721750593, -457.20106757917586, -436.6761318868941, -
                 453.22407953378104, -451.4841009720122, -458.22325101026524] # , -446.3692186115176, -471.65477980018903, -446.15288226277755, -461.7219878679857 ]
ex1_3buc_dic = [-441.505773419289, -434.77449451787777, -440.3930609105001, -455.9745866974446, -459.18602277958564, -438.5722013618144, -
                454.04686040454317, -453.4085041176336, -462.9393414074259] # , -451.7351996722965, -475.8914189614111, -453.0240221360621, -465.7271286341008  ]
wAicex1 = pd.DataFrame(
    dict(
        M2=ex1_2bucs_waic,
        M3=ex1_3buc_waic,
        M4=ex1_4buc_waic))
wAicex1 = wAicex1.rename(
    columns={
        'M2': r'$M_2$',
        'M3': r'$M_3$',
        'M4': r'$M_4$'})
dicex1 = pd.DataFrame(dict(M2=ex1_2bucs_Dic, M3=ex1_3buc_dic, M4=ex1_3buc_dic))
dicex1 = dicex1.rename(
    columns={
        'M2': r'$M_2$',
        'M3': r'$M_3$',
        'M4': r'$M_4$'})


# First subplot
fig = plt.figure()

# Add the first subplot (large one on the left)
ax1 = fig.add_subplot(1, 2, 1)
ax1.boxplot([M2, M3, M4], patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax1.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax1.set_ylabel('Log marginal likelihood', fontsize=10)
ax1.set_xlabel('Models', fontsize=10)
ax1.text(-0.25, 1.1, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')


# Second subplot
ax2 = fig.add_subplot(2, 2, 2)
ax2.boxplot(dicex1, patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax2.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax2.set_ylabel('DIC')
ax2.text(-0.2, 1.2, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')


# third subplot
ax3 = fig.add_subplot(2, 2, 4)
ax3.boxplot(wAicex1, patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax3.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax3.set_xlabel("Models", fontsize=10)
ax3.set_ylabel("WAIC")
ax3.text(-0.2, 1.2, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')

fig.subplots_adjust(wspace=0.4, hspace=0.3)
fig.set_figheight(2.5)
fig.set_figwidth(5)
plt.savefig("./ic_ex1.pdf")
