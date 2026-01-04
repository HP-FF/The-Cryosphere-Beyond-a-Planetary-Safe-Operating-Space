# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

plt.rcParams['svg.fonttype'] = 'none'
def get_ais():
    nc = "../../draw_data/Land Ice/AIS/rslpismflat_ais_quants.nc"

    data = xr.open_dataset(nc)
    value = data['esl']
    time = data['age']
    median = value[:, 2]
    value_66up = value[:, 3]
    value_66down = value[:, 1]
    value_90up = value[:, 4]
    value_90down = value[:, 0]
    return time,median,value_66up,value_66down,value_90up,value_90down

def get_gris():
    nc = "../../draw_data/Land Ice/GRIS/rslpismflat_gris_quants.nc"
    data = xr.open_dataset(nc)
    value = data['esl']
    time = data['time']
    median = value[:, 2]
    value_66up = value[:, 3]
    value_66down = value[:, 1]
    value_90up = value[:, 4]
    value_90down = value[:, 0]
    return time, median, value_66up, value_66down, value_90up, value_90down


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.
    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    # print(data)
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((quantile > 1.) or (quantile < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    return np.interp(quantile, Pn, sorted_data)

def get_mountain():
    quantiles = [0.025, 0.05, 0.16, 0.5, 0.83, 0.95, 0.975]

    mg_wts = xr.open_dataset(
        '../../draw_data/Land Ice/Mountain/rslpismflat_mountain_wts.nc').wts
    mg_esls = xr.open_dataset(
        '../../draw_data/Land Ice/Mountain/mountain_esls.nc').esl
    try:
        mg_esls = mg_esls.sel(icemod=mg_wts.icemod)
    except:
        pass
    try:
        mg_wts = mg_wts.sel(icemod=mg_esls.icemod)
    except:
        pass
    q025, q05, q16, q50, q83, q95, q975 = [mg_esls.quantile(q, 'icemod') for q in quantiles]
    ages = mg_esls.age.values

    ds_q = xr.DataArray(quantiles, coords={'quantile': quantiles}, dims=['quantile'])

    ds_qs = xr.apply_ufunc(quantile_1D,
                           mg_esls.chunk(dict(icemod=-1)).astype(float),
                           mg_wts.chunk(dict(icemod=-1)).astype(float),
                           ds_q,
                           vectorize=True,
                           dask="parallelized",
                           input_core_dims=[["icemod"], ["icemod"], []],
                           output_dtypes=[ds_q.dtype],
                           ).compute()
    wq025, wq05, wq16, wq50, wq83, wq95, wq975 = [ds_qs.sel(quantile=q) for q in quantiles]

    return ages,q50,wq83,wq16,wq95,wq05



if __name__=='__main__':
    excel = r"../../draw_data/Land Ice/nature_draw_GT.xlsx"
    ex_data = pd.read_excel(excel)
    year = ex_data['Year'].tolist()
    g_low = ex_data['Global [lower]']
    g_high = ex_data['Global [upper]']
    g_mean = ex_data['Global [mean]']
    g_mean1 = ex_data['Global [mean]'].tolist()

    time_ais, median_ais, value_66up_ais, value_66down_ais, value_90up_ais, value_90down_ais = get_ais()
    time_gris, median_gris, value_66up_gris, value_66down_gris, value_90up_gris, value_90down_gris = get_gris()
    time_mountain, median_mountain, value_66up_mountain, value_66down_mountain, value_90up_mountain, value_90down_mountain = get_mountain()

    median_mountain_new = np.pad(median_mountain,(0,len(median_ais)- len(median_mountain)),'constant')
    value_66up_mountain_new = np.pad(value_66up_mountain,(0,len(value_66up_ais)- len(value_66up_mountain)),'constant')
    value_66down_mountain_new = np.pad(value_66down_mountain, (0, len(value_66down_ais) - len(value_66down_mountain)),
                                     'constant')
    value_90up_mountain_new = np.pad(value_90up_mountain, (0, len(value_90up_ais) - len(value_90up_mountain)),
                                     'constant')
    value_90down_mountain_new = np.pad(value_90down_mountain, (0, len(value_90down_ais) - len(value_90down_mountain)),
                                       'constant')
    value_median0 = -(np.array(median_gris) + np.array(median_ais) + median_mountain_new)*0.3625
    value_66up0 = -(np.array(value_66up_gris) + np.array(value_66up_ais) + value_66up_mountain_new)*0.3625
    value_66down0 = -(np.array(value_66down_gris) + np.array(value_66down_ais) + value_66down_mountain_new)*0.3625
    value_90up0 = -(np.array(value_90up_gris) + np.array(value_90up_ais) + value_90up_mountain_new)*0.3625
    value_90down0 = -(np.array(value_90down_gris) + np.array(value_90down_ais) + value_90down_mountain_new)*0.3625

    gap_66up = value_median0 - value_66up0
    gap_66down = value_median0 - value_66down0
    gap_90up = value_median0 - value_90up0
    gap_90down = value_median0 - value_90down0

    median = (value_median0 + 27.6597900769764)
    value_66up = (median - gap_66down)
    value_66down = (median - gap_66up)
    value_90up = (median - gap_90down)
    value_90down = (median - gap_90up)

    th95_Ho = np.percentile(median[1:59], 5)

    th95_Mid = np.percentile(median[25:35], 5)

    Ho_median = np.median(median[1:59])
    mid_median = np.median(median[25:35])
    print("Ho_median:{}".format(Ho_median))
    print("Mid_median:{}".format(mid_median))

    print("Ho:{}".format(th95_Ho))
    print("Mid:{}".format(th95_Mid))


    fig, ax1 = plt.subplots(figsize=(15, 3))
    ax1.plot(time_ais, median, label='Creel et al. (2023)', color='#0b0b0b', marker='o', markersize=0)

    ax1.fill_between(time_ais, value_66down, value_66up, color='#a4b9ca', alpha=0.4, label='66% credible interval')
    ax1.fill_between(time_ais, value_90down, value_90up, color='#d5dadd', alpha=0.4, label='90% credible interval')

    ax1.plot(year, g_mean, label='Frederikse et al. (2020)', color='#2626ff')
    ax1.fill_between(year, g_low, g_high, color='#d3d3ff', alpha=0.5, label='90% credible interval')

    time = np.array(time_ais).tolist()
    th_time = np.array(year[51:] + time)
    ax1.plot(th_time, np.full_like(th_time, th95_Ho), label='Holocene Baseline', color='#ff0001')

    x_pos = -3.4
    y_min, y_max = g_mean1[-1]-6.546*0.3625, g_mean1[-1]-0.299*0.3625
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

    font_style = {'family': 'Times New Roman',
                  'size': 10}

    ax1.set_ylabel('Ice volume (10^6 Gt)', font=font_style)  # y_label
    plt.legend(frameon=False)
    plt.gca().invert_xaxis()

    current_year = 1950
    bias = 0
    ax1.xaxis.set_major_locator(
        ticker.FixedLocator(
            [bias + 10, bias + 8, bias + 6, bias + 4, bias + 2, bias, bias - 1.5, bias - 2.5, bias - 3.5]))

    ax1.set_xticklabels([
        str(int(bias + 10)),
        str(int(bias + 8)),
        str(int(bias + 6)),
        str(int(bias + 4)),
        str(int(bias + 2)),
        str(int(current_year - bias)),

        str(int(current_year - (bias - 1.5) * 20)),

        str(int(current_year - (bias - 2.5) * 20)),

        str(int(current_year - (bias - 3.5) * 20))
    ])

    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    plt.axvline(0, c='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlim(left=11.7)

    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.show()
    # out_path = "./Extended Data Fig 1a.svg"
    # plt.savefig(out_path, dpi=500, format='svg')


