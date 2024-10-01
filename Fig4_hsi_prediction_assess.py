# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot Fig. 4

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from matplotlib import gridspec
from matplotlib import cm
from sklearn.metrics import r2_score

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

rread(model_input + 'HSI_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')

# All
# (9, 10, 47, 1220) 'at', 'tw', 'ts', 'wbgt', 'swbgt', 'hx', 'apt', 'utci', 'hi'
all_prediction_org = np.array(ro.r('all_prediction'))
all_prediction = np.zeros((9, 10, 47, 1220)) * np.NaN
all_prediction[0, :, :, :] = all_prediction_org[0, :, :, :]
all_prediction[1:, :, :, :] = all_prediction_org[1:, :, :, :][::-1, :, :, :]
# at, hi, utci, apt, hx, swbgt, wbgt, ts, tw

with np.load(model_input + 'model_input.npz') as file:
    date_array_summer = file['date_array_summer']  # (1220, 5)
    all_htk = file['all_htk']  # (47, 1220)


#######################################################################################################################
# MAE and R2 calculation

# MAE
all_mae_cali = np.zeros((9, 10, 47)) * np.NaN  # var, yy, pf
all_mae_vali = np.zeros((9, 10, 47)) * np.NaN

# R2
all_R2_cali = np.zeros((9, 10, 47)) * np.NaN
all_R2_vali = np.zeros((9, 10, 47)) * np.NaN

for yy in range(10):
    for pf in range(47):
        for var in range(9):

            vali_idx = np.where(date_array_summer[:, 0] == yy + 2010)[0]
            cali_idx = np.where(date_array_summer[:, 0] != yy + 2010)[0]

            # Old vali
            nan_idx = (~np.isnan(all_htk[pf, vali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, vali_idx]))

            all_mae_vali[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, vali_idx] - all_prediction[var, yy, pf, vali_idx]))

            all_R2_vali[var, yy, pf] = r2_score((all_htk[pf, vali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, vali_idx])[nan_idx])

            # Old cali
            nan_idx = (~np.isnan(all_htk[pf, cali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, cali_idx]))

            all_mae_cali[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, cali_idx] - all_prediction[var, yy, pf, cali_idx]))

            all_R2_cali[var, yy, pf] = r2_score((all_htk[pf, cali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, cali_idx])[nan_idx])

print('R2, MAE calculation finished.')

#####################################################################################################################

pref_47 = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu',
           'Shizuoka', 'Aichi', 'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo',
           'Nara', 'Wakayama', 'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi', 'Fukuoka', 'Saga', 'Nagasaki',
           'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa']

hsi_tick_humi_weight = np.array(['T',
                                 'HI', 'UTCI', 'APT', r'H$_{\mathrm{x}}$',
                                 r'T$_{\mathrm{sWBG}}$', r'T$_{\mathrm{WBG}}$',
                                 r'T$_{\mathrm{s}}$', r'T$_{\mathrm{w}}$'])

color_hub_hsi = ['k', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:pink',
                 'tab:brown', 'tab:purple', 'tab:red', 'lightcoral']

lws = 0.9
markers = 'o'
mksizes = 2

# ##################################################################################################
# # Radar plot of R2

fig2 = plt.figure(2, figsize=(two_clm, 5.5), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig2,
                       nrows=3,
                       ncols=2,
                       height_ratios=[1,
                                      0.05,
                                      0.5],
                       width_ratios=[1, 1])

# All Cali
ax_pl = fig2.add_subplot(gs[0, 0], projection='polar')
ax_pl.grid(linestyle='--')
for ii in range(9):
    ax_pl.plot(np.linspace(0, 47, 48) / 47 * np.pi * 2, np.append(np.mean(all_R2_cali[ii, :, :], axis=0),
                                                                  np.mean(all_R2_cali[ii, :, 0])),
               c=color_hub_hsi[ii], label=hsi_tick_humi_weight[ii], linewidth=lws, marker=markers, markersize=mksizes)
plt.axis([0, 2 * np.pi, 0, 1])
ax_pl.set_xticks(np.linspace(0, 46, 47) / 47 * np.pi * 2)
ax_pl.set_xticklabels(np.linspace(1, 47, 47, dtype=int), fontsize=8)
ax_pl.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_pl.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=8)
ax_pl.set_rlabel_position(12.5)
ax_pl.text(0.764 * np.pi, 1.4, 'a', fontsize=9, fontweight='bold')
ax_pl.set_title('Calibration', fontsize=9)

# All Vali
ax_pl = fig2.add_subplot(gs[0, 1], projection='polar')
ax_pl.grid(linestyle='--')
for ii in range(9):
    ax_pl.plot(np.linspace(0, 47, 48) / 47 * np.pi * 2, np.append(np.mean(all_R2_vali[ii, :, :], axis=0),
                                                                  np.mean(all_R2_vali[ii, :, 0])),
               c=color_hub_hsi[ii], label=hsi_tick_humi_weight[ii], linewidth=lws, marker=markers, markersize=mksizes)
plt.axis([0, 2 * np.pi, 0, 1])
ax_pl.set_xticks(np.linspace(0, 46, 47) / 47 * np.pi * 2)
ax_pl.set_xticklabels(np.linspace(1, 47, 47, dtype=int), fontsize=8)
ax_pl.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_pl.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=8)
ax_pl.set_rlabel_position(12.5)
ax_pl.text(0.764 * np.pi, 1.4, 'b', fontsize=9, fontweight='bold')
ax_pl.set_title('Validation', fontsize=9)

# legend
handles, labels = plt.gca().get_legend_handles_labels()
axlgd = fig2.add_subplot(gs[1, :])
lgd = plt.legend(handles=handles,
                 fontsize='small', ncol=10,
                 loc='lower center',
                 borderpad=0.4, labelspacing=0.1, columnspacing=0.3, handletextpad=0.2)
axlgd.set_frame_on(False)
axlgd.set_xticks([])
axlgd.set_yticks([])

# cali box
ax = fig2.add_subplot(gs[2, 0])
var_box = [
    np.mean(all_R2_cali[0, :, :], axis=0),
    np.mean(all_R2_cali[1, :, :], axis=0),
    np.mean(all_R2_cali[2, :, :], axis=0),
    np.mean(all_R2_cali[3, :, :], axis=0),
    np.mean(all_R2_cali[4, :, :], axis=0),
    np.mean(all_R2_cali[5, :, :], axis=0),
    np.mean(all_R2_cali[6, :, :], axis=0),
    np.mean(all_R2_cali[7, :, :], axis=0),
    np.mean(all_R2_cali[8, :, :], axis=0)
]
bplot = plt.boxplot(var_box,
                    medianprops={'color': 'red', 'linewidth': 1},
                    patch_artist=True,
                    showfliers=False)
# fill with colors
for patch, color in zip(bplot['boxes'], color_hub_hsi):
    patch.set_facecolor(color)

ax.axhline(y=np.median(np.mean(all_R2_cali[0, :, :], axis=0)), c='r', linestyle='--', lw=1)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], hsi_tick_humi_weight, fontsize=8)
plt.ylabel(r'R$^{\mathrm{2}}$', fontsize=8)
plt.axis([0.5, 9.5, 0.3, 1])
ax.axhline(y=0.4, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=0.6, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=0.8, c='grey', linestyle='--', lw=0.6)
plt.text(0.5, 1.01, 'c', fontsize=8, fontweight='bold')
plt.yticks([0.4, 0.6, 0.8, 1.0],
           [0.4, 0.6, 0.8, 1.0],
           fontsize=8)


# vali box
ax = fig2.add_subplot(gs[2, 1])
var_box = [
    np.mean(all_R2_vali[0, :, :], axis=0),
    np.mean(all_R2_vali[1, :, :], axis=0),
    np.mean(all_R2_vali[2, :, :], axis=0),
    np.mean(all_R2_vali[3, :, :], axis=0),
    np.mean(all_R2_vali[4, :, :], axis=0),
    np.mean(all_R2_vali[5, :, :], axis=0),
    np.mean(all_R2_vali[6, :, :], axis=0),
    np.mean(all_R2_vali[7, :, :], axis=0),
    np.mean(all_R2_vali[8, :, :], axis=0)
]
bplot = plt.boxplot(var_box,
                    medianprops={'color': 'red', 'linewidth': 1},
                    patch_artist=True,
                    showfliers=False)
# fill with colors
for patch, color in zip(bplot['boxes'], color_hub_hsi):
    patch.set_facecolor(color)

ax.axhline(y=np.median(np.mean(all_R2_vali[0, :, :], axis=0)), c='r', linestyle='--', lw=1)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], hsi_tick_humi_weight, fontsize=8)
plt.ylabel(r'R$^{\mathrm{2}}$', fontsize=8)
plt.axis([0.5, 9.5, 0.3, 1])
ax.axhline(y=0.4, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=0.6, c='grey', linestyle='--', lw=0.6)
ax.axhline(y=0.8, c='grey', linestyle='--', lw=0.6)
plt.text(0.5, 1.01, 'd', fontsize=8, fontweight='bold')
plt.yticks([0.4, 0.6, 0.8, 1.0],
           [0.4, 0.6, 0.8, 1.0],
           fontsize=8)

# plt.show()

fig2.savefig(fig_dir + 'hsi_R2.svg',
             dpi=1200,
             format='svg')

print('All Finished.')



