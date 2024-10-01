# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot Fig. 2


import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec


# mpl.use('Qt5Agg')
plt.close('all')
one_clm = 3.46
two_clm = 7.08

# induce some R function
rread = ro.r['load']
ls = ro.r('ls()')
levels = ro.r('levels')

model_input = '/work/a07/qiang/Infor_Loss/02_model/'
fig_dir = '/work/a07/qiang/Infor_Loss/04_plot/fig_test/'
input_dir = '/work/a07/qiang/Aging_Japan/climate_data/'

rread(model_input + 'CLM_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')

# All
all_prediction = np.array(ro.r('all_prediction'))  # (5, 10, 47, 1220)  'at', 'rh', 'ws', 'rad'， CLM

with np.load(model_input + 'model_input.npz') as file:
    date_array_summer = file['date_array_summer']  # (1220, 5)
    # 'Newborn', 'Baby', 'Teenager', 'Adult', 'Elderly', 'Unclear'
    all_htk = file['all_htk']  # (47, 1220)

#####################################################################################################################

pref_47 = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu',
           'Shizuoka', 'Aichi', 'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo',
           'Nara', 'Wakayama', 'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi', 'Fukuoka', 'Saga', 'Nagasaki',
           'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa']

color_hub_clm = ['k', 'tab:blue', 'tab:orange', 'tab:green']

######################################################################################################################
# Time series
fig1 = plt.figure(1, figsize=(two_clm, 5), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=2,
                       ncols=2,
                       height_ratios=[1,
                                      1],
                       width_ratios=[1, 1])

lw = 0.9
vali_period = 8
ct_sum = np.array([12, 22, 26, 39])  # original
# ct_sum = np.array([28, 0, 46, 18])  # test
ct_name_1 = ['Tokyo', 'Aichi', 'Osaka', 'Fukuoka']
no_sum_1 = ['a', 'b', 'c', 'd']
grey_it = 0.7
###############################################

for ii in range(4):
    ax = fig1.add_subplot(gs[ii // 2, ii % 2])
    plt.title(ct_name_1[ii], fontsize=8.6)

    plt.bar(np.linspace(0, 121, 122),
            all_htk[ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
            facecolor=[grey_it, grey_it, grey_it], label='OBS', edgecolor=[grey_it, grey_it, grey_it])
    plt.plot(np.linspace(0, 121, 122),
             all_prediction[0, vali_period, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color=color_hub_clm[0], label='T', linewidth=lw)
    plt.plot(np.linspace(0, 121, 122),
             all_prediction[1, vali_period, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color=color_hub_clm[1], label='T-RH', linewidth=lw)
    plt.plot(np.linspace(0, 121, 122),
             all_prediction[2, vali_period, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color=color_hub_clm[2], label='T-RH-W', linewidth=lw)
    plt.plot(np.linspace(0, 121, 122),
             all_prediction[3, vali_period, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
             color=color_hub_clm[3], label='T-RH-W-SR', linewidth=lw)
    # plt.plot(np.linspace(0, 121, 122),
    #          all_prediction[4, vali_period, ct_sum[ii], vali_period * 122: (vali_period + 1) * 122],
    #          color='red', label='CLM', linewidth=lw)

    if ii == 0:
        plt.axis([0, 122, -10, 500])
        plt.text(4, 492, no_sum_1[ii], fontsize=8, fontweight='bold')
        plt.yticks([0, 200, 400], [0, 200, 400], fontsize=8)
    elif ii == 3:
        plt.axis([0, 122, -5, 200])
        plt.text(4, 196.8, no_sum_1[ii], fontsize=8, fontweight='bold')
        plt.yticks([0, 100, 200], [0, 100, 200], fontsize=8)
    else:
        plt.axis([0, 122, -10, 450])
        plt.text(4, 442.8, no_sum_1[ii], fontsize=8, fontweight='bold')
        plt.yticks([0, 200, 400], [0, 200, 400], fontsize=8)

    plt.xticks([0, 30, 61, 92], ['Jun', 'Jul', 'Aug', 'Sep'], fontsize=8)
    plt.ylabel('Daily HT-EADs', fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.2)
    # legend
    if ii == 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.legend(handles=handles,
                         fontsize='small', ncol=1,
                         loc='upper right',
                         borderpad=0.4, labelspacing=0.1, columnspacing=0.3, handletextpad=0.2)

plt.show()

fig1.savefig(fig_dir + 'clm_time_series.svg',
             dpi=1200,
             format='svg')

print('All Finished.')






