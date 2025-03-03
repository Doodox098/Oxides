import pandas as pd
import numpy as np
import os
import torch

def gen_gauss(size, sigma):
    mean = size // 2
    x = np.arange(0, size)
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) / (sigma * (2 * np.pi) ** 0.5)

def reference_read(path, config):
    temp_step = config['temp_step']
    sigma = config['reference_sigma']
    data = pd.read_csv(path)
    values = data['dO3']
    total_oxygen = float(os.path.basename(path).split()[2].replace(',', '.'))
    temp = np.interp(np.arange(len(data['dO3'])), np.arange(np.array(~data['dT1'].isna()).sum() * 10, step=10),
                           data['dT1'][~data['dT1'].isna()].sort_values())
    temp = (np.sort(temp) + 273)[::temp_step]
    gauss = gen_gauss(len(values), sigma)
    smoothed_mean = np.convolve(values, gauss, mode='same')[::temp_step]
    v_max = smoothed_mean.max()

    return torch.tensor(smoothed_mean / v_max), torch.tensor(temp), v_max, total_oxygen