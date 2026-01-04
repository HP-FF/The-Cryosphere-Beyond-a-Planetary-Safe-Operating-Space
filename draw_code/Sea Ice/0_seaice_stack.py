# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm

import os
import numpy as np
import pandas as pd
from matplotlib import ticker
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['svg.fonttype'] = 'none'


def one_2_tow(time_list, mena_list):
    if 1850 in time_list:
        index_01 = time_list.index(1850)
        histroy_time = time_list[0:index_01]
        now_time = time_list[index_01:]

        histroy_mean = mena_list[0:index_01]
        now_mean = mena_list[index_01:]
    else:
        histroy_time = None
        histroy_mean = None
        now_time = time_list
        now_mean = mena_list

    return np.array(histroy_time), np.array(histroy_mean), np.array(now_time), np.array(now_mean)

def one_2_tow_list(time_list, mena_list):
    if 1850 in time_list:
        index_01 = time_list.index(1850)
        histroy_time = time_list[0:index_01]
        now_time = time_list[index_01:]

        histroy_mean = mena_list[0:index_01]
        now_mean = mena_list[index_01:]
    else:
        histroy_time = None
        histroy_mean = None
        now_time = time_list
        now_mean = mena_list

    return histroy_time, histroy_mean, now_time, now_mean


def transform_years(time_list, zoom, bias):
    new_list = []
    for x in time_list:
        if x < 1850:
            new_list.append(x / zoom)
        else:
            new_list.append(bias + (x - 1850) / 2)
    return new_list


def inverse_transform_years(x, pos, zoom, bias):
    if x < bias:
        return f"{int(x * zoom)}"  # recover 1500-1850
    else:
        return f"{int(1850 + (x - bias) * 2)}"  # recover 1850-2100


if __name__ == "__main__":
    excel_folder = r"../../draw_data/Sea Ice/G"

    cesm2_N = pd.read_excel(r"../../draw_data/Sea Ice/N/Arctic SeaIce_CESM2.xlsx")
    cesm2_N_time = cesm2_N['time'].tolist()  # time
    cesm2_N_data = cesm2_N['data'].tolist()  # mean
    cesm2N_x_h, cesm2N_y_h, cesm2N_x_n, cesm2N_y_n = one_2_tow(cesm2_N_time, cesm2_N_data)

    cesm2_S = pd.read_excel(r"../../draw_data/Sea Ice/S/Antarctic SeaIce_CESM2.xlsx")
    cesm2_S_time = cesm2_S['time'].tolist()  # time
    cesm2_S_data = cesm2_S['data'].tolist()  # mean
    cesm2S_x_h,  cesm2S_y_h,  cesm2S_x_n,  cesm2S_y_n = one_2_tow(cesm2_S_time, cesm2_S_data)

    AME_N = pd.read_excel(r"../../draw_data/Sea Ice/N/Arctic SeaIce_AME.xlsx")
    AME_N_time = AME_N['time'].tolist()  # time
    AME_N_data = AME_N['data'].tolist()  # mean
    AMEN_x_h, AMEN_y_h, AMEN_x_n, AMEN_y_n = one_2_tow(AME_N_time, AME_N_data)

    AME_S = pd.read_excel(r"../../draw_data/Sea Ice/S/Antarctic SeaIce_AME.xlsx")
    AME_S_time = AME_S['time'].tolist()  # time
    AME_S_data = AME_S['data'].tolist()  # mean
    AMES_x_h, AMES_y_h, AMES_x_n, AMES_y_n = one_2_tow(AME_S_time, AME_S_data)

    zoom_factor = 5
    gap = 4
    bias = 1850 / zoom_factor + gap
    ex_folder = []
    for ex in os.listdir(excel_folder):
        if ex.endswith('.xlsx'):
            ec_path = os.path.join(excel_folder, ex)
            ex_folder.append(ec_path)

    color_index = 0
    plt.figure(figsize=(15, 3))

    name_figure = None

    ind_model_index = 0
    for expath in ex_folder:
        name_figure = os.path.basename(expath).split('_')[0]
        name = os.path.basename(expath).split('_')[0] + ' ($10^6\ KM^2$)'  # Y

        ex_data = pd.read_excel(expath)
        x_axis_data = ex_data['time'].tolist()  # time
        y_axis_data = ex_data['data'].tolist()  # mean

        x_h, y_h, x_n, y_n = one_2_tow_list(x_axis_data, y_axis_data)

        if x_h is None:
            x_n = transform_years(x_n, zoom_factor, bias)
        else:
            x_n = transform_years(x_n, zoom_factor, bias)
            x_h = transform_years(x_h, zoom_factor, bias)

        if os.path.basename(expath).split('_')[1].split('.')[0] == 'CESM2':

            yh_05 = np.percentile(y_h, 5)
            yn_05 = np.percentile(y_n[0:50],5)
            print("current file：{}，5%:{}".format(os.path.basename(expath), yh_05))
            print("current file：{}，5_now%:{}".format(os.path.basename(expath), yn_05))

            x0 = x_h[0]
            if color_index == 0:
                plt.plot(x_h, y_h, c='#2626ff', linestyle='-', alpha=0.8, linewidth=2,label='CESM2 mid-Holocene and historical series')
                plt.plot(x_n, y_n, c='#2626ff', linestyle='-', alpha=0.8, linewidth=2)
                ax = plt.gca()
                formatter = partial(inverse_transform_years, zoom=zoom_factor, bias=bias)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))

                ax.xaxis.set_major_locator(
                    ticker.FixedLocator([bias, bias + 25, bias + 50, bias + 67.5, bias + 85]))

                ax.yaxis.set_major_locator(MaxNLocator(5))

                plt.axvline((bias), c='gray', linestyle='--', alpha=0.5, linewidth=1)  # 1850
                plt.axvline((bias + 25), c='gray', linestyle='--', alpha=0.5, linewidth=1)  # 1900

                plt.axvline(bias-gap, c='gray', linestyle='--', alpha=0.8, linewidth=1)

                ax.spines['left'].set_position(('data', x0))

                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')

                ax.set_xlim(left=x0,right=bias+90)

                x = np.linspace(x0, bias + 90, 5000)

                plt.plot(x[x <= bias-gap], [yh_05] * len(x[x <= bias-gap]), c='#2626ff', linestyle='--', alpha=0.8, linewidth=2,label='Mid-Holocene baseline based on CESM2')
                plt.plot(x[x >= bias], [yh_05] * len(x[x >= bias]), c='#2626ff', linestyle='--', alpha=0.8,
                         linewidth=2)

                plt.fill_between(x_h,0,cesm2S_y_h,label='Antarctic',color='#B8E2F4')
                plt.fill_between(x_h,cesm2S_y_h,cesm2S_y_h+cesm2N_y_h,label='Arctic',color='#cee6c1')

        else:
            if os.path.basename(expath).split('_')[1].split('.')[0] == 'AME':
                yn_05 = np.percentile(y_n[0:50],5)
                print("current file：{}，5_H%:{}".format(os.path.basename(expath), yn_05))

                x = np.linspace(bias, bias + 90, 500)

                plt.plot(x, [yn_05] * len(x[x >= bias]), c='red', linestyle='--', alpha=0.8, linewidth=2,label='Pre-industrial baseline based on ISIMIP multimodel mean')
                plt.plot(x_n, y_n, c='#0b0b0b', linestyle='-', alpha=0.8, linewidth=2, label='ISIMIP multimodel mean, 1850-2014')

                plt.fill_between(x_n,0,AMES_y_n,color='#B8E2F4')
                plt.fill_between(x_n,AMES_y_n,AMES_y_n+AMEN_y_n,color='#cee6c1')

            elif os.path.basename(expath).split('_')[1].split('.')[0] == 'Satellite':
                plt.plot(x_n, y_n, c='#fea443', linestyle='-', alpha=0.8, linewidth=2,label='NSIDC satellite observation data')

        plt.legend(loc='lower left')
        plt.ylabel('Area (10^6 km2)')  # y_label

    plt.show()
    # out_path = "./Fig 2b.svg"
    # plt.savefig(out_path,dpi=500,format='svg')
