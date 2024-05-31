import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax.config import config
import scienceplots
config.update("jax_enable_x64", True)
plt.style.use(['science', 'ieee'])

# Studyone
# The results were obtained by running each model many times in parallel with different initial values
# The shell files provided can modified to run the models any number of
# times on the HPC cluster, and the results post-processed

M2 = jnp.array([162.33330603,
                160.65340229,
                162.13155279,
                163.22579496,
                163.19500443,
                161.30983552,
                162.27907378,
                160.12726685,
                163.70623022,
                158.96132323,
                161.6713138,
                160.02142916,
                162.75224699,
                162.03856076,
                159.37880723])
M3 = jnp.array([175.38756715,
                172.51896784,
                175.55742437,
                174.39873683,
                173.67282319,
                174.54476284,
                175.0716626,
                173.51980885,
                174.19445704,
                173.90953245,
                176.60535209,
                174.85196102,
                168.35110554,
                172.01482422,
                173.07743515])
M4 = jnp.array([146.76705408,
                147.20628314,
                148.83795516,
                148.01173018,
                148.76830863,
                148.6846524,
                148.31805913,
                146.53414979,
                146.12002402,
                148.21255901,
                147.42079548,
                148.47407528,
                149.21503881,
                149.05161122,
                149.27308698])

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

ex2_2bucs_Dic = [-401.13021947435124, -400.6291806006183, -401.0611516691965, -404.9035726216561, -400.75893795787664, -403.1747084218352, - \
    400.8043213048324, -399.8032065217425, -402.7227780059838, -400.569108474599, -402.17791978534456] # , -400.5091498903982,  -401.192010175437]
ex2_2bucs_waic = [-394.70650786821875, -395.32579039969744, -395.63821426779276, -394.4105023099678, -394.5865906442441, -394.363403840405, - \
    392.70046172157066, -393.17409292166775, -394.87059008574045, -394.52595774172863, -392.4112975815236] # , -393.59402949049553, -389.95720850801246]
ex2_3buc_waic = [-419.8113319562443, -420.68372229550084, -417.32993824927166, -420.299002596208, -421.48943833803, -422.03039499463165, -
                 419.02522596087897, -424.4427525682583, -418.3063557682755, -421.0925206938771, -419.66628792660606]  # , -417.41206639093747]
ex2_3buc_dic = [-428.0300465923006, -427.68109396829374, -427.59958561245173, -428.16086653429556, -429.1195714041543, -427.399946206066, -
                428.51663285690506, -428.52507511780226, -429.846786779024, -426.1559829012299, -426.0106505338563]  # , -427.037885861247]
ex2_4buc_waic = [-421.7878128277034, -413.67510578130555, -409.52481101386263, -415.1021833354367, -421.3396599778833, - \
    412.6682582877571, -416.248341510543, -425.7577565338513, -421.6792559674022, -410.5571713027781, -420.56920885968833]
ex2_4buc_dic = [-427.71782963079454, -425.4888450003781, -423.14668214812974, -424.803118152705, -426.1365610182779, - \
    424.7825603537255, -427.20324876777505, -429.8009890928533, -425.01095503934704, -425.32326678437204, -427.98217335445014]
wAicex2 = pd.DataFrame(
    dict(
        M2=ex2_2bucs_waic,
        M3=ex2_3buc_waic,
        M4=ex2_4buc_waic))
wAicex2 = wAicex2.rename(
    columns={
        'M2': r'$M_2$',
        'M3': r'$M_3$',
        'M4': r'$M_4$'})
dicex2 = pd.DataFrame(dict(M2=ex2_2bucs_Dic, M3=ex2_3buc_dic, M4=ex2_4buc_dic))
dicex2 = dicex2.rename(
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
ax1.text(-0.25, 1.13, '(a)', transform=ax1.transAxes, fontsize=14,
         fontweight='bold', verticalalignment='top', horizontalalignment='left')

# Second subplot
ax2 = fig.add_subplot(2, 2, 2)
ax2.boxplot(dicex2, patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax2.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax2.set_ylabel('DIC')
ax2.text(-0.2, 1.3, '(b)', transform=ax2.transAxes, fontsize=14,
         fontweight='bold', verticalalignment='top', horizontalalignment='left')


# third subplot
ax3 = fig.add_subplot(2, 2, 4)
ax3.boxplot(wAicex2, patch_artist=True, flierprops=dict(
    marker='.', ms=5, markerfacecolor='k'))
ax3.set_xticks([1, 2, 3], [r'$M_2$', r'$M_3$', r'$M_4$'])
ax3.set_xlabel("Models", fontsize=10)
ax3.set_ylabel("WAIC")
ax3.text(-0.2, 1.2, '(c)', transform=ax3.transAxes, fontsize=14,
         fontweight='bold', verticalalignment='top', horizontalalignment='left')

fig.subplots_adjust(wspace=0.4, hspace=0.3)
fig.set_figheight(2.5)
fig.set_figwidth(5)
plt.savefig("./ic_ex2.pdf")
