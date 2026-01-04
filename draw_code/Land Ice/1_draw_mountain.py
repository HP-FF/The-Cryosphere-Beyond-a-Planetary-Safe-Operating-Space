# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker


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
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]

    return np.interp(quantile, Pn, sorted_data)


quantiles = [0.025, 0.05, 0.16, 0.5, 0.83, 0.95, 0.975]
plt.rcParams['svg.fonttype'] = 'none'

mg_wts = xr.open_dataset(
    r'../../draw_data/Land Ice/Mountain/rslpismflat_mountain_wts.nc').wts
mg_esls = xr.open_dataset(
    r'../../draw_data/Land Ice/Mountain/mountain_esls.nc').esl
try:
    mg_esls = mg_esls.sel(icemod=mg_wts.icemod)
except:
    pass
try:
    mg_wts = mg_wts.sel(icemod=mg_esls.icemod)
except:
    pass

q025, q05, q16, q50, q83, q95, q975 = [mg_esls.quantile(q, 'icemod') for q in quantiles]
ages = np.array(mg_esls.age.values).tolist()

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

wq05 = wq05 * -0.3625
wq16 = wq16 * -0.3625
q50 = q50 * -0.3625
wq83 = wq83 * -0.3625
wq95 = wq95 * -0.3625

gap_66up = q50 - wq83
gap_66down = q50 - wq16
gap_90up = q50 - wq95
gap_90down = q50 - wq05

median = q50 + 0.100071730955597  # convert absolute ice content
value_66up = (median - gap_66down)
value_66down = (median - gap_66up)
value_90up = (median - gap_90down)
value_90down = (median - gap_90up)
fig, ax = plt.subplots(figsize=(15, 3))

f2 = 10

font_style = {'family': 'Times New Roman',
              'size': 10}

ax.fill_between(ages, value_90down, value_90up, alpha=0.4, hatch='///', color='None', edgecolor='black',
                label="90% credible interval")
ax.fill_between(ages, 0, median, color='#8d5591', alpha=0.4)
ax.plot(ages, median, color='#0b0b0b', label='Creel et al. (2023)')
ax.set_xlim(-7.6, 80)
plt.gca().invert_xaxis()

excel = r"../../draw_data/Land Ice/nature_draw_GT.xlsx"
ex_data = pd.read_excel(excel)
year = ex_data['Year'].tolist()
ais_low = ex_data['Glaciers [lower]']
ais_high = ex_data['Glaciers [upper]']
ais_mean = ex_data['Glaciers [mean]']
ais_mean1 = (ex_data['Glaciers [mean]']).tolist()

# start 1950
ax.plot(year[50:], ais_mean[50:], label='Frederikse et al. (2020)', color='#2626ff')
ax.fill_between(year[50:], ais_low[50:], ais_high[50:], alpha=0.4, hatch='///', color='None', edgecolor='black')
ax.fill_between(year[50:], 0, ais_mean[50:], alpha=0.4, color='#8d5591')

th95_Ho = np.percentile(median[1:59], 5)
th95_Mid = np.percentile(median[25:35], 5)
Ho_median = np.median(median[1:59])
mid_median = np.median(median[25:35])
print("Ho_median:{}".format(Ho_median))
print("Mid_median:{}".format(mid_median))
print("Ho:{}".format(th95_Ho))
print("Mid:{}".format(th95_Mid))

x_pos = -3.4
y_min, y_max = ais_mean1[-1] - 0.204 * 0.3625, ais_mean1[-1] - 0.043 * 0.3625
center = (y_min + y_max) / 2
lower_err = center - y_min
upper_err = y_max - center
ax.errorbar(x=[x_pos], y=[center],
            yerr=[[lower_err], [upper_err]],
            fmt='none',
            ecolor='#d57eeb',
            label='Committed change with credible interval',
            capsize=5,
            elinewidth=2)


current_year = 1950
bias = 0
ax.xaxis.set_major_locator(
    ticker.FixedLocator(
        [bias + 10, bias + 8, bias + 6, bias + 4, bias + 2, bias, bias - 1.5, bias - 2.5, bias - 3.5]))

# Convert the scale values to year labels
ax.set_xticklabels([
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


ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.legend(frameon=False)
plt.axvline(0, c='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlim(left=11.7, right=-4)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
plt.show()
# out_path = r"./fig 2a_sub.svg"
# plt.savefig(out_path, dpi=500, format='svg')
