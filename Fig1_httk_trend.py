# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to plot Fig. 1


import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from matplotlib import gridspec
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rpy2.robjects as ro

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
plot_dir = '/work/a07/qiang/Aging_Japan/04_plot/'
jp_hsi_dir = '/work/a07/qiang/HSI_Japan/Japan_Data/HSIs_1980_2019/'

pref_47 = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 'Yamagata', 'Fukushima',
           'Ibaraki', 'Tochigi', 'Gunma', 'Saitama', 'Chiba', 'Tokyo', 'Kanagawa',
           'Niigata', 'Toyama', 'Ishikawa', 'Fukui', 'Yamanashi', 'Nagano', 'Gifu',
           'Shizuoka', 'Aichi', 'Mie', 'Shiga', 'Kyoto', 'Osaka', 'Hyogo',
           'Nara', 'Wakayama', 'Tottori', 'Shimane', 'Okayama', 'Hiroshima', 'Yamaguchi',
           'Tokushima', 'Kagawa', 'Ehime', 'Kochi', 'Fukuoka', 'Saga', 'Nagasaki',
           'Kumamoto', 'Oita', 'Miyazaki', 'Kagoshima', 'Okinawa']

# Load heat stroke data
with np.load(model_input + 'model_input.npz') as file:
    date_array_summer = file['date_array_summer']  # (1220, 5)
    all_htk = file['all_htk']  # (47, 1220)

htk_each_yr = np.zeros((47, 10)) * np.NaN
for ii in range(10):
    htk_each_yr[:, ii] = np.nansum(all_htk[:, ii * 122: (ii + 1) * 122], axis=1)

# annual for 2010-2019
all_htk_yr = np.nansum(all_htk, axis=1) / 10

# load Japan shp
jp_shp_file = plot_dir + '/JP_shp/gadm41_JPN_1.shp'
jp_shp = gpd.read_file(jp_shp_file)
jp_shp['geometry'] = jp_shp['geometry'].simplify(0.01)
jp_shp['NAME_1'][12] = 'Hyogo'
jp_shp['NAME_1'][26] = 'Nagasaki'

with np.load(jp_hsi_dir + 'jp_hsi.npz') as file:
    jp_lon = file['jp_lon']  # 47
    jp_lat = file['jp_lat']  # 47
    jp_city = file['jp_city']  # 47

shp_order_toshow_all = np.zeros(47)
shp_order_toshow_young = np.zeros(47)
shp_order_toshow_old = np.zeros(47)
for ct in range(47):
    shp_order = np.where(jp_city == np.array(jp_shp['NAME_1'], dtype='str')[ct])[0]
    shp_order_toshow_all[ct] = all_htk_yr[shp_order]

jp_shp['ALL_STK'] = shp_order_toshow_all

######################################################################################################
# # # Figure 1


def ts_pref(ct, od):
    plt.plot(np.linspace(2010, 2019, 10, dtype=int), htk_each_yr[ct, :],
             'o', ls='-', ms=2,
             lw=0.5, c='k', label='Observations')
    plt.axis([2009.7, 2019.3, 0, 8000])
    plt.xticks([2010, 2012, 2014, 2016, 2018],
               [2010, 2012, 2014, 2016, 2018],
               fontsize=8)
    plt.yticks([2000, 4000, 6000],
               [2000, 4000, 6000],
               fontsize=8)

    ax.axhline(y=2000, c='grey', linestyle='--', lw=0.6)
    ax.axhline(y=4000, c='grey', linestyle='--', lw=0.6)
    ax.axhline(y=6000, c='grey', linestyle='--', lw=0.6)
    plt.xlabel('Year', fontsize=8)
    plt.ylabel('Annual HT-EADs', fontsize=8)
    plt.title(jp_city[ct], fontsize=8)
    plt.text(2009.6, 8100, od, fontsize=8, fontweight='bold')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


fig1 = plt.figure(1, figsize=(two_clm, 5.5), constrained_layout=True)
gs = gridspec.GridSpec(figure=fig1,
                       nrows=4,
                       ncols=2,
                       height_ratios=[1,
                                      1,
                                      1,
                                      1],
                       width_ratios=[1, 0.3])

# Japan map
ax = fig1.add_subplot(gs[0:4, 0], projection=ccrs.LambertConformal())
ax.set_position([-0.01, 0.03, 0.7, 1.00])
ax.set_frame_on(False)

cmap_ovlp = plt.cm.get_cmap('OrRd', 10)
bounds_ovlp = np.linspace(0, 5000, 11)
norm_ovlp = mpl.colors.BoundaryNorm(bounds_ovlp, cmap_ovlp.N)

# Main part
jp46_shp = jp_shp[jp_shp['NAME_1'] != 'Okinawa']
jp46_shp.plot(ax=ax, linewidth=0.3, column='ALL_STK', cmap=cmap_ovlp, norm=norm_ovlp)
plt.axis([127, 147, 29.7, 46])
ax.set_xticks([])
ax.set_yticks([])
ax.plot([128, 131], [38.5, 38.5], c='k', linewidth=0.5)
ax.plot([137, 137], [45, 42], c='k', linewidth=0.5)
ax.plot([131, 137], [38.5, 42], c='k', linewidth=0.5)
plt.text(128, 45, 'a', fontsize=8, fontweight='bold')
# Tokyo
plt.scatter(jp_lon[12], jp_lat[12], c='k', s=2)
plt.text(jp_lon[12] + 2, jp_lat[12] + 2, jp_city[12], fontsize='8')
plt.plot([jp_lon[12], jp_lon[12] + 2], [jp_lat[12], jp_lat[12] + 2], c='k', linewidth=0.5)

# Aichi
plt.scatter(jp_lon[22], jp_lat[22], c='k', s=2)
plt.text(jp_lon[22] + 2, jp_lat[22] - 2, jp_city[22], fontsize='8')
plt.plot([jp_lon[22], jp_lon[22] + 1.8], [jp_lat[22], jp_lat[22] - 1.5], c='k', linewidth=0.5)

# Osaka
plt.scatter(jp_lon[26], jp_lat[26], c='k', s=2)
plt.text(jp_lon[26], jp_lat[26] - 2.5, jp_city[26], fontsize='8', horizontalalignment='center')
plt.plot([jp_lon[26], jp_lon[26]], [jp_lat[26], jp_lat[26] - 2.2], c='k', linewidth=0.5)

# Fukuoka
plt.scatter(jp_lon[39], jp_lat[39], c='k', s=2)
plt.text(jp_lon[39], jp_lat[39] + 2.5, jp_city[39], fontsize='8', horizontalalignment='center')
plt.plot([jp_lon[39], jp_lon[39]], [jp_lat[39], jp_lat[39] + 2.2], c='k', linewidth=0.5)

# Okinawa
okinawa_shp = jp_shp[jp_shp['NAME_1'] == 'Okinawa']
ax_oknw = ax.inset_axes([0.01, 0.5, 0.6, 0.6])
okinawa_shp.plot(ax=ax_oknw, linewidth=0.3, column='ALL_STK', cmap=cmap_ovlp, norm=norm_ovlp)
ax_oknw.set_xlim(123, 130)
ax_oknw.set_ylim(24, 28)
ax_oknw.set_xticks([])
ax_oknw.set_yticks([])
ax_oknw.set_frame_on(False)
plt.text(129, 43, 'Okinawa', fontsize=8)

# Colorbar
ax_cb = ax.inset_axes([0.80, 0.02, 0.04, 0.3])
cb = colorbar.ColorbarBase(ax_cb,
                           cmap=cmap_ovlp,
                           norm=norm_ovlp,
                           orientation='vertical',
                           ticks=list(np.linspace(0, 5000, 6)))
cb.ax.set_yticklabels(list(np.linspace(0, 5000, 6, dtype=int)), fontsize=7)
# cb.set_label('Averaged annual HT-EADs', fontsize=8)

ax_cb.set_frame_on(False)

# Time series Tokyo
ax = fig1.add_subplot(gs[0, 1])
ts_pref(12, 'b')

# Time series Osaka
ax = fig1.add_subplot(gs[1, 1])
ts_pref(22, 'c')

# Time series Aichi
ax = fig1.add_subplot(gs[2, 1])
ts_pref(26, 'd')

# Time series Fukuoka
ax = fig1.add_subplot(gs[3, 1])
ts_pref(39, 'e')

plt.show()

fig1.savefig(fig_dir + 'httk_info.svg',
             format='svg',
             dpi=1200)

print('All Finished.')




















