# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

plt.rcParams['svg.fonttype'] = 'none'


nc = "../../draw_data/Land Ice/AIS/rslpismflat_ais_quants.nc"
excel = "../../draw_data/Land Ice/nature_draw_GT.xlsx"
ex_data = pd.read_excel(excel)
year = ex_data['Year'].tolist()
ais_low = ex_data['Antarctic Ice Sheet [lower]']
ais_high = ex_data['Antarctic Ice Sheet [upper]']
ais_mean = ex_data['Antarctic Ice Sheet [mean]']
ais_mean1 = (ex_data['Antarctic Ice Sheet [mean]']).tolist()  # committee

data = xr.open_dataset(nc)

value = data['esl']
time = np.array(data['age']).tolist()
# [11.7 --- 0.1] Ho
# [0.1 --- 0.05]  pre
# [7 --- 5]     Mid
median0 = value[:,2] * -0.3625
value_66up0 = value[:,3] * -0.3625
value_66down0 = value[:,1] * -0.3625
value_90up0 = value[:,4] * -0.3625
value_90down0 = value[:,0] * -0.3625

gap_66up = median0-value_66up0
gap_66down = median0-value_66down0
gap_90up = median0-value_90up0
gap_90down = median0-value_90down0

median = (median0 + 24.7963724799037)
value_66up = (median-gap_66down)
value_66down = (median-gap_66up)
value_90up = (median-gap_90down)
value_90down = (median-gap_90up)

th95_Ho = np.percentile(median[1:59],5)
th95_Ho_66d = np.percentile(value_66down[1:59],5)
th95_Ho_66u = np.percentile(value_66up[1:59],5)
th95_Ho_90d = np.percentile(value_90down[1:59],5)
th95_Ho_90u = np.percentile(value_90up[1:59],5)

th95_Mid = np.percentile(median[25:35],5)
th95_Mid_66d = np.percentile(value_66down[25:35],5)
th95_Mid_66u = np.percentile(value_66up[25:35],5)
th95_Mid_90d = np.percentile(value_90down[25:35],5)
th95_Mid_90u = np.percentile(value_90up[25:35],5)

Ho_median = np.median(median[1:59])
mid_median = np.median(median[25:35])
print("Ho_median:{}".format(Ho_median))
print("Mid_median:{}".format(mid_median))
print("Ho:{}".format(th95_Ho))
print("Mid:{}".format(th95_Mid))


fig, ax1 = plt.subplots(figsize=(15, 3))
ax1.plot(time, median, label='Creel et al. (2023)', color='#0b0b0b', marker='o', markersize=0)

ax1.fill_between(time, value_66down, value_66up, color='#a4b9ca', alpha=0.4,label='66% credible interval')
ax1.fill_between(time, value_90down, value_90up, color='#d5dadd', alpha=0.4,label='90% credible interval')

ax1.plot(year,ais_mean,label='Frederikse et al. (2020)',color='#2626ff')
ax1.fill_between(year, ais_low, ais_high, color='#d3d3ff', alpha=0.4,label='90% credible interval')

th_time = np.array(year[51:] + time)
ax1.plot(th_time,np.full_like(th_time,th95_Ho),label='Holocene Baseline',color='#ff0001')


font_style={'family':'Times New Roman',
            'size':12}


x_pos = -3.4
y_min,y_max = ais_mean1[-1]-6*0.3625,ais_mean1[-1]-0.05*0.3625
center = (y_min + y_max) / 2
lower_err = center - y_min
upper_err = y_max - center
ax1.errorbar(x=[x_pos], y=[center],
             yerr=[[lower_err], [upper_err]],
             fmt='none',
             ecolor='#d57eeb',
             label='Committed change with credible interval',
             capsize=5,
             elinewidth=2)

ax1.set_ylabel('Ice volume (10^6 Gt)',font=font_style)  # y_label

plt.legend(frameon=False)
plt.gca().invert_xaxis()


current_year = 1950
bias = 0
ax1.xaxis.set_major_locator(
    ticker.FixedLocator([bias+10,bias+8,bias+6,bias+4,bias+2,bias, bias - 1.5,  bias - 2.5,bias-3.5]))
ax1.set_xticklabels([
str(int(bias+10)),
str(int(bias+8)),
str(int(bias+6)),
str(int(bias+4)),
str(int(bias+2)),
    str(int(current_year - bias)),

    str(int(current_year - (bias - 1.5)*20)),

    str(int(current_year - (bias - 2.5)*20)),

    str(int(current_year - (bias - 3.5)*20))
])

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')

plt.axvline(0, c='gray', linestyle='--', alpha=0.5, linewidth=1)

ax1.set_xlim(left=11.7)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
plt.show()
# out_path = r"./Extended Data Fig 1b.svg"
# plt.savefig(out_path, dpi=500, format='svg')

