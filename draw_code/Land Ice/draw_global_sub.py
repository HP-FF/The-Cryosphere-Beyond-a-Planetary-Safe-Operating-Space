# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

plt.rcParams['svg.fonttype'] = 'none'


if __name__=='__main__':

    excel = "../../draw_data/Land Ice/nature_draw_GT_sub.xlsx"
    ex_data = pd.read_excel(excel)
    year = ex_data['Year']
    g_low = ex_data['Global [lower]']
    g_high = ex_data['Global [upper]']
    g_mean = ex_data['Global [mean]']

    fig, ax1 = plt.subplots(figsize=(6, 3))

    ax1.plot(year, g_mean, label='Frederikse et al. 2020', color='#2626ff')
    ax1.fill_between(year, g_low, g_high, color='#d3d3ff', alpha=0.5, label='90% credible interval')

    th95_Ho = np.full_like(year, 27.485659258519114, dtype=float)

    ax1.plot(year, th95_Ho, label='Holocene Baseline 95%', color='#ff0001')

    font_style = {'family': 'Times New Roman',
                  'size': 14}

    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    plt.axvline(1950, c='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax1.xaxis.set_major_locator(MaxNLocator(5))

    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.show()
    # out_path = "./Extended Data Fig 1a_sub.svg"
    # plt.savefig(out_path, dpi=500, format='svg')


