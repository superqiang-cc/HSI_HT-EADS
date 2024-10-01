# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot Fig. 5

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

with np.load(model_input + 'model_input.npz') as file:
    date_array_summer = file['date_array_summer']  # (1220, 5)
    all_htk = file['all_htk']  # (47, 1220)

#########################################################################################################
# HSI
rread(model_input + 'HSI_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')
# (9, 10, 47, 1220) 'at', 'tw', 'ts', 'wbgt', 'swbgt', 'hx', 'apt', 'utci', 'hi'
all_prediction_org = np.array(ro.r('all_prediction'))
all_prediction = np.zeros((9, 10, 47, 1220)) * np.NaN
all_prediction[0, :, :, :] = all_prediction_org[0, :, :, :]
all_prediction[1:, :, :, :] = all_prediction_org[1:, :, :, :][::-1, :, :, :]

# MAE and R2 calculation

# MAE
all_mae_cali_hsi = np.zeros((9, 10, 47)) * np.NaN  # var, yy, pf
all_mae_vali_hsi = np.zeros((9, 10, 47)) * np.NaN

# R2
all_R2_cali_hsi = np.zeros((9, 10, 47)) * np.NaN
all_R2_vali_hsi = np.zeros((9, 10, 47)) * np.NaN

for yy in range(10):
    for pf in range(47):
        for var in range(9):

            vali_idx = np.where(date_array_summer[:, 0] == yy + 2010)[0]
            cali_idx = np.where(date_array_summer[:, 0] != yy + 2010)[0]

            # Old vali
            nan_idx = (~np.isnan(all_htk[pf, vali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, vali_idx]))

            all_mae_vali_hsi[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, vali_idx] - all_prediction[var, yy, pf, vali_idx]))

            all_R2_vali_hsi[var, yy, pf] = r2_score((all_htk[pf, vali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, vali_idx])[nan_idx])

            # Old cali
            nan_idx = (~np.isnan(all_htk[pf, cali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, cali_idx]))

            all_mae_cali_hsi[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, cali_idx] - all_prediction[var, yy, pf, cali_idx]))

            all_R2_cali_hsi[var, yy, pf] = r2_score((all_htk[pf, cali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, cali_idx])[nan_idx])

print('HSI: R2, MAE calculation finished.')


#####################################################################################################################
# CLM
rread(model_input + 'CLM_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata')
all_prediction = np.array(ro.r('all_prediction'))  # (5, 10, 47, 1220) 'at', 'rh', 'ws', 'rad'， CLM

# MAE and R2 calculation

# MAE
all_mae_cali_clm = np.zeros((5, 10, 47)) * np.NaN  # var, yy, pf
all_mae_vali_clm = np.zeros((5, 10, 47)) * np.NaN

# R2
all_R2_cali_clm = np.zeros((5, 10, 47)) * np.NaN
all_R2_vali_clm = np.zeros((5, 10, 47)) * np.NaN

for yy in range(10):
    for pf in range(47):
        for var in range(5):

            vali_idx = np.where(date_array_summer[:, 0] == yy + 2010)[0]
            cali_idx = np.where(date_array_summer[:, 0] != yy + 2010)[0]

            # Old vali
            nan_idx = (~np.isnan(all_htk[pf, vali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, vali_idx]))

            all_mae_vali_clm[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, vali_idx] - all_prediction[var, yy, pf, vali_idx]))

            all_R2_vali_clm[var, yy, pf] = r2_score((all_htk[pf, vali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, vali_idx])[nan_idx])

            # Old cali
            nan_idx = (~np.isnan(all_htk[pf, cali_idx])
                       & ~np.isnan(all_prediction[var, yy, pf, cali_idx]))

            all_mae_cali_clm[var, yy, pf] = np.nanmean(
                np.abs(all_htk[pf, cali_idx] - all_prediction[var, yy, pf, cali_idx]))

            all_R2_cali_clm[var, yy, pf] = r2_score((all_htk[pf, cali_idx])[nan_idx],
                                                (all_prediction[var, yy, pf, cali_idx])[nan_idx])

print('CLM: R2, MAE calculation finished.')
#####################################################################################################

pref_47 = ['Hokkaido',  # Hokkaido 1
           'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',  # Tohoku 6
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',  # Kanto 7
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu', 'Shizuoka', 'Aichi',  # Chubu 9
           'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo', 'Nara', 'Wakayama',  # Kansai 7
           'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',  # Chugoku 5
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi',  # Shikoku 4
           'Fukuoka', 'Saga', 'Nagasaki', 'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa']  # Kyushu & Okinawa 8

Region_8_nm = ['Hokkaido (1)', 'Tohoku (6)', 'Kanto (7)', 'Chubu (9)',
               'Kansai (7)', 'Chugoku (5)', 'Shikoku (4)', 'Kyushu\n& Okinawa (8)']

Region_8_nb = [[0],  # Hokkaido 1
               [1, 2, 3, 4, 5, 6],  # Tohoku 6
               [7, 8, 9, 10, 11, 12, 13],  # Kanto 7
               [14, 15, 16, 17, 18, 19, 20, 21, 22],  # Chubu 9
               [23, 24, 25, 26, 27, 28, 29],  # Kansai 7
               [30, 31, 32, 33, 34],  # Chugoku 5
               [35, 36, 37, 38],  # Shikoku 4
               [39, 40, 41, 42, 43, 44, 45, 46]]  # Kyushu & Okinawa 8


hsi_tick_humi_weight = np.array(['T',
                                 'HI', 'UTCI', 'APT', r'H$_{\mathrm{x}}$',
                                 r'T$_{\mathrm{sWBG}}$', r'T$_{\mathrm{WBG}}$',
                                 r'T$_{\mathrm{s}}$', r'T$_{\mathrm{w}}$'])

var_tick_sum = ['T', 'T-RH', 'T-RH-W', 'T-RH-\nW-SR', 'CLM']

color_hub_hsi = ['k', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:pink',
                 'tab:brown', 'tab:purple', 'tab:red', 'lightcoral']

color_hub_clm = ['k', 'tab:blue', 'tab:orange', 'tab:green']

var_2 = np.array([1, 4, 5, 7, 8])
var_3 = np.array([2, 3])
var_4 = np.array([6])

# Obtian the R2 different relative to Tair
R2_cali_clm_diff = np.zeros((3, 47)) * np.NaN
R2_vali_clm_diff = np.zeros((3, 47)) * np.NaN

R2_cali_hsi_diff = np.zeros((8, 47)) * np.NaN
R2_vali_hsi_diff = np.zeros((8, 47)) * np.NaN

R2_var2_diff = np.zeros((2, 6, 47)) * np.NaN
R2_var3_diff = np.zeros((2, 3, 47)) * np.NaN
R2_var4_diff = np.zeros((2, 2, 47)) * np.NaN

for ii in range(3):

    R2_cali_clm_diff[ii, :] = np.nanmean((all_R2_cali_clm[ii + 1, :, :] - all_R2_cali_clm[0, :, :]), axis=0)
    R2_vali_clm_diff[ii, :] = np.nanmean((all_R2_vali_clm[ii + 1, :, :] - all_R2_vali_clm[0, :, :]), axis=0)

for ii in range(8):

    R2_cali_hsi_diff[ii, :] = np.nanmean((all_R2_cali_hsi[ii + 1, :, :] - all_R2_cali_hsi[0, :, :]), axis=0)
    R2_vali_hsi_diff[ii, :] = np.nanmean((all_R2_vali_hsi[ii + 1, :, :] - all_R2_vali_hsi[0, :, :]), axis=0)

R2_var2_diff[0, 0, :] = R2_cali_clm_diff[0, :]
R2_var2_diff[0, 1:, :] = R2_cali_hsi_diff[var_2 - 1, :]
R2_var2_diff[1, 0, :] = R2_vali_clm_diff[0, :]
R2_var2_diff[1, 1:, :] = R2_vali_hsi_diff[var_2 - 1, :]

R2_var3_diff[0, 0, :] = R2_cali_clm_diff[1, :]
R2_var3_diff[0, 1:, :] = R2_cali_hsi_diff[var_3 - 1, :]
R2_var3_diff[1, 0, :] = R2_vali_clm_diff[1, :]
R2_var3_diff[1, 1:, :] = R2_vali_hsi_diff[var_3 - 1, :]

R2_var4_diff[0, 0, :] = R2_cali_clm_diff[2, :]
R2_var4_diff[0, 1:, :] = R2_cali_hsi_diff[var_4 - 1, :]
R2_var4_diff[1, 0, :] = R2_vali_clm_diff[2, :]
R2_var4_diff[1, 1:, :] = R2_vali_hsi_diff[var_4 - 1, :]

# divide into var 2, 3, 4, and 8 regions
R2_var2_diff_rg = np.zeros((2, 6, 8)) * np.NaN
R2_var3_diff_rg = np.zeros((2, 3, 8)) * np.NaN
R2_var4_diff_rg = np.zeros((2, 2, 8)) * np.NaN

for rg in range(8):
    R2_var2_diff_rg[:, :, rg] = np.nanmean(R2_var2_diff[:, :, Region_8_nb[rg]], axis=2)
    R2_var3_diff_rg[:, :, rg] = np.nanmean(R2_var3_diff[:, :, Region_8_nb[rg]], axis=2)
    R2_var4_diff_rg[:, :, rg] = np.nanmean(R2_var4_diff[:, :, Region_8_nb[rg]], axis=2)


# # ##################################################################################################
# # # Bar plot of R2


fig1 = plt.figure(1, figsize=(two_clm, 5.8), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=3,
                       ncols=2,
                       height_ratios=[1,
                                      1,
                                      1],
                       width_ratios=[1, 1])
hsi_tick_humi_weight = np.array(['T',
                                 'HI', 'UTCI', 'APT', r'H$_{\mathrm{x}}$',
                                 r'T$_{\mathrm{sWBG}}$', r'T$_{\mathrm{WBG}}$',
                                 r'T$_{\mathrm{s}}$', r'T$_{\mathrm{w}}$'])

width = 0.12  # the width of the bars
# cali
##########################################################################################################
cvali = 0
# var 2
ax = fig1.add_subplot(gs[0, 0])
var2_bar = {
    'T-RH': R2_var2_diff_rg[cvali, 0, :],
    'HI': R2_var2_diff_rg[cvali, 1, :],
    r'H$_{\mathrm{x}}$': R2_var2_diff_rg[cvali, 2, :],
    r'T$_{\mathrm{sWBG}}$': R2_var2_diff_rg[cvali, 3, :],
    r'T$_{\mathrm{s}}$': R2_var2_diff_rg[cvali, 4, :],
    r'T$_{\mathrm{w}}$': R2_var2_diff_rg[cvali, 5, :],
}

var2_cl = ['tab:blue', 'tab:cyan', 'tab:pink', 'tab:brown', 'tab:red', 'lightcoral']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var2_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var2_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks([])
plt.yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], fontsize=8)
plt.axis([-0.2, 7.8, -0.3, 0.3])
ax.legend(loc='upper right', ncol=3, fontsize='x-small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.26, 'a', fontsize=8, fontweight='bold')

# var 3
ax = fig1.add_subplot(gs[1, 0])
var3_bar = {
    'T-RH-W': R2_var3_diff_rg[cvali, 0, :],
    'UTCI': R2_var3_diff_rg[cvali, 1, :],
    'APT': R2_var3_diff_rg[cvali, 2, :],
}

var3_cl = ['tab:orange', 'tab:olive', 'tab:grey']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var3_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var3_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks([])
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], [-0.2, -0.1, 0, 0.1, 0.2], fontsize=8)
plt.axis([-0.2, 7.8, -0.2, 0.2])
ax.legend(loc='upper right', ncol=3, fontsize='x-small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.17, 'b', fontsize=8, fontweight='bold')

# var 4
ax = fig1.add_subplot(gs[2, 0])
var4_bar = {
    'T-RH-W-SR': R2_var4_diff_rg[cvali, 0, :],
    r'T$_{\mathrm{WBG}}$': R2_var4_diff_rg[cvali, 1, :],
}

var4_cl = ['tab:green', 'tab:purple']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var4_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var4_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks(x + width * 0.5, Region_8_nm, fontsize=8, rotation=-45)
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], [-0.2, -0.1, 0, 0.1, 0.2], fontsize=8)
plt.axis([-0.2, 7.8, -0.2, 0.2])
ax.legend(loc='upper right', ncol=2, fontsize='x-small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.17, 'c', fontsize=8, fontweight='bold')

# vali
##########################################################################################################
cvali = 1
# var 2
ax = fig1.add_subplot(gs[0, 1])
var2_bar = {
    'T-RH': R2_var2_diff_rg[cvali, 0, :],
    'HI': R2_var2_diff_rg[cvali, 1, :],
    r'H$_{\mathrm{x}}$': R2_var2_diff_rg[cvali, 2, :],
    r'T$_{\mathrm{sWBG}}$': R2_var2_diff_rg[cvali, 3, :],
    r'T$_{\mathrm{s}}$': R2_var2_diff_rg[cvali, 4, :],
    r'T$_{\mathrm{w}}$': R2_var2_diff_rg[cvali, 5, :],
}

var2_cl = ['tab:blue', 'tab:cyan', 'tab:pink', 'tab:brown', 'tab:red', 'lightcoral']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var2_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var2_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks([])
plt.yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], fontsize=8)
plt.axis([-0.2, 7.8, -0.3, 0.3])
# ax.legend(loc='upper right', ncol=3, fontsize='small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.26, 'd', fontsize=8, fontweight='bold')

# var 3
ax = fig1.add_subplot(gs[1, 1])
var3_bar = {
    'T-RH-W': R2_var3_diff_rg[cvali, 0, :],
    'UTCI': R2_var3_diff_rg[cvali, 1, :],
    'APT': R2_var3_diff_rg[cvali, 2, :],
}

var3_cl = ['tab:orange', 'tab:olive', 'tab:grey']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var3_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var3_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks([])
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], [-0.2, -0.1, 0, 0.1, 0.2], fontsize=8)
plt.axis([-0.2, 7.8, -0.2, 0.2])
# ax.legend(loc='upper right', ncol=3, fontsize='small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.17, 'e', fontsize=8, fontweight='bold')

# var 4
ax = fig1.add_subplot(gs[2, 1])
var4_bar = {
    'T-RH-W-SR': R2_var4_diff_rg[cvali, 0, :],
    r'T$_{\mathrm{WBG}}$': R2_var4_diff_rg[cvali, 1, :],
}

var4_cl = ['tab:green', 'tab:purple']

x = np.arange(len(Region_8_nm))  # the label locations
multiplier = 0
for attribute, measurement in var4_bar.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,
                   facecolor=var4_cl[multiplier], edgecolor='k', lw=0.5)
    multiplier += 1

ax.set_ylabel(r'Difference in R$^{\mathrm{2}}$', fontsize=8)
plt.xticks(x + width * 0.5, Region_8_nm, fontsize=8, rotation=-45)
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], [-0.2, -0.1, 0, 0.1, 0.2], fontsize=8)
plt.axis([-0.2, 7.8, -0.2, 0.2])
# ax.legend(loc='upper right', ncol=2, fontsize='small')
ax.axhline(y=0, c='grey', linestyle='-', lw=0.6)
plt.text(-0.15, 0.17, 'f', fontsize=8, fontweight='bold')

plt.show()

fig1.savefig(fig_dir + 'R2_INFO_region.svg',
             dpi=1200,
             format='svg')

print('All Finished.')



