# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(6,3))
excel = r"../../draw_data/Land Ice/nature_draw_GT_sub.xlsx"
ex_data = pd.read_excel(excel)
year = ex_data['Year']
ais_low = ex_data['Glaciers [lower]']
ais_high = ex_data['Glaciers [upper]']
ais_mean = ex_data['Glaciers [mean]']
ax.plot(year,ais_mean,label='Frederikse et al. 2020',color ='#2626ff')
ax.fill_between(year, ais_low, ais_high, color='#d3d3ff', alpha=0.4,label='90% credible interval')

th95_Ho = np.full_like(year,0.08334212576693405,dtype=float)
ax.plot(year,th95_Ho,label='Holocene Baseline 95%',color='#ff0001')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

font_style={'family':'Times New Roman',
            'size':14}
plt.legend(frameon=False)
plt.axvline(1950, c='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_locator(MaxNLocator(5))


plt.legend(frameon=False)
plt.show()
# out_path = r"./Extended Data Fig 1d_sub.svg"
# plt.savefig(out_path, dpi=500, format='svg')
