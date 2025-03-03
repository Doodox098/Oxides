import argparse
import json

import numpy as np
from train import temperature_shift
import scipy
import torch
from reference_read import reference_read
from OxideModel import OxideModel
from train import train


def get_ppm(oxide_models, global_shift_delta, reference, config):
    oxide_groups = config['oxide_groups']
    temp_step = config['temp_step']
    grid = np.linspace(0, 448.9, num=1000)
    group_oxygen = dict([(key, 0) for key in oxide_groups.keys()])
    oxides_oxygen = {}
    for name, oxide in oxide_models.items():
        global_shift = temperature_shift(global_shift_delta)

        def time_to_temp(time):
            t = reference[1]
            return oxide(torch.tensor(t[int(time * 10 / temp_step):int(time * 10 / temp_step) + 1]), global_shift)

        y = []
        for time in grid:
            y.append(time_to_temp(time).item())
        integral = scipy.integrate.simpson(np.array(y), grid) * reference[2] / 1.6e6
        oxides_oxygen[name] = integral
    total_oxygen = reference[3]
    total_oxygen_integrate = scipy.integrate.simpson(reference[0], np.linspace(0, 448.9, num=len(reference[0]))) * reference[2] / 1.6e6

    sum = 0
    for name, value in oxides_oxygen.items():
        oxides_oxygen[name] = value * total_oxygen / total_oxygen_integrate
        sum += oxides_oxygen[name]

        for key, value in oxide_groups.items():
            if name in value:
                group_oxygen[key] += oxides_oxygen[name]

    return group_oxygen, oxides_oxygen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--reference_path', type=str)
    parser.add_argument('--oxide_params', type=str, default='params.json')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    with open(args.oxide_params, 'r') as f:
        oxide_params = json.load(f)
    reference = reference_read(args.reference_path, config)
    oxygen, time = reference[:2]
    oxide_models = {}
    global_shift_delta = torch.nn.parameter.Parameter(torch.tensor([0], dtype=torch.float64), requires_grad=True)
    for oxide_name, oxide in oxide_params.items():

        e_var = config['e_var']
        e_scale = config['e_scale']
        v_var = max(0, oxygen[np.argmin(np.abs(time - oxide['Tm']))])
        v_scale = config['v_scale']
        t_max_delta = config['t_max_delta']
        t_max_delta_scale = config['t_max_delta_scale']
        t_beg_delta = config['t_beg_delta']
        t_beg_delta_scale = config['t_beg_delta_scale']
        if v_var < 0.01:
            continue
        oxide_models[oxide_name] = OxideModel(
            {'init': e_var,
             'scale': e_scale, },
            {'init': v_var,
             'scale': v_scale, },
            {'init': oxide["Tm"],
             'delta': t_max_delta,
             'delta_scale': t_max_delta_scale},
            {'init': oxide["Tb"],
             'delta': t_beg_delta,
             'delta_scale': t_beg_delta_scale},
            model=config['model']
        )

    oxide_models = torch.nn.ModuleDict(oxide_models)
    import torch.optim

    optimizer = getattr(torch.optim, config['optim'])([x for x in oxide_models.parameters()] + [global_shift_delta], **config['optim_params'])

    train(oxide_models, global_shift_delta, reference[:2], optimizer, config)
    group_oxygen, oxides_oxygen = get_ppm(oxide_models, global_shift_delta, reference, config)
    print(group_oxygen)
