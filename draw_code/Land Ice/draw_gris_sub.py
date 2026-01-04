# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

plt.rcParams['svg.fonttype'] = 'none'

excel = r"../../draw_data/Land Ice/nature_draw_GT_sub.xlsx"
ex_data = pd.read_excel(excel)
year = ex_data['Year']
ais_low = ex_data['Greenland Ice Sheet [lower]']
ais_high = ex_data['Greenland Ice Sheet [upper]']
ais_mean = ex_data['Greenland Ice Sheet [mean]']


fig, ax1 = plt.subplots(figsize=(6, 3))

ax1.plot(year,ais_mean,label='Frederikse et al. 2020',color='#2626ff')
ax1.fill_between(year, ais_low, ais_high, color='#d3d3ff', alpha=0.4,label='90% credible interval')

th95_Ho = np.full_like(year,2.6692507203972253,dtype=float)

ax1.plot(year,th95_Ho,label='Holocene Baseline 95%',color='#ff0001')


font_style={'family':'Times New Roman',
            'size':14}

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')

plt.axvline(1950, c='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.yaxis.set_major_locator(MaxNLocator(4))
ax1.xaxis.set_major_locator(MaxNLocator(5))

plt.show()
# out_path = r"./Extended Data Fig 1c_sub.svg"
# plt.savefig(out_path, dpi=500, format='svg')

