import argparse
import json
from itertools import chain

import numpy as np
from PIL import Image as PILImage

from oxid import get_tb, load_chem_data, load_oxides, get_tb_tm
from train import temperature_shift
import scipy
import torch
from reference_read import reference_read
from OxideModel import OxideModel
from train import train, draw_simple


class OxideServer: # Из этого надо сделать ViewModel
    def __init__(self, ):
        pass


def get_oxide_ppm(oxide_models, global_shift_delta, reference, config):
    # oxide_groups = config['oxide_groups']
    temp_step = config['temp_step']
    def time_to_temp(time):
        t = reference[1]
        return oxide(t[(time * 10 / temp_step).astype(int)], global_shift)
    grid = np.linspace(0, 448.9, num=1000)
    # group_oxygen = dict([(key, 0) for key in oxide_groups.keys()])
    oxides_oxygen = {}
    for name, oxide in oxide_models.items():
        global_shift = temperature_shift(global_shift_delta)

        y = time_to_temp(grid).detach().numpy()
        integral = scipy.integrate.simpson(y, grid)
        oxides_oxygen[name] = integral
    total_oxygen = reference[3]
    total_oxygen_integrate = scipy.integrate.simpson(reference[0], np.linspace(0, 448.9, num=len(reference[0])))

    sum = 0
    for name, value in oxides_oxygen.items():
        oxides_oxygen[name] = value * total_oxygen / total_oxygen_integrate
        sum += oxides_oxygen[name]

        # for key, value in oxide_groups.items():
        #     if name in value:
        #         group_oxygen[key] += oxides_oxygen[name]

    return oxides_oxygen


def get_oxides_vf(oxides_oxygen, elements_table, oxide_params, use_density=False):
    """
    Calculate volume fractions of oxides based on oxygen content and oxide parameters.

    Args:
        oxides_oxygen: Dictionary of oxygen content for each oxide (in ppm)
        elements_table: Dictionary of atomic weights for elements
        oxide_params: Dictionary containing oxide composition and density information

    Returns:
        Dictionary of volume fractions for each oxide
    """
    oxides_vf = {}
    total_volume = 0.0
    volume_dict = {}

    for oxide_name, oxygen_content in oxides_oxygen.items():
        if oxide_name not in oxide_params:
            continue

        # Get oxide composition and density from params
        composition = oxide_params[oxide_name]['structure']
        density = None
        if use_density:
            density = oxide_params[oxide_name]['density']  # g/cm³

        # Calculate molecular weight of the oxide
        mol_weight = 0.0
        oxygen_count = 0

        for element, count in composition.items():
            if element == 'O':
                oxygen_count += count
            else:
                mol_weight += elements_table[element] * count
        mol_weight += elements_table['O'] * oxygen_count

        oxide_mass = (oxygen_content * mol_weight) / (elements_table['O'] * oxygen_count)

        # Calculate volume of the oxide
        oxide_volume = oxide_mass / density if density else oxide_mass
        volume_dict[oxide_name] = oxide_volume
        total_volume += oxide_volume

    # Calculate volume fractions
    if total_volume > 0:
        oxides_vf = {oxide: volume / total_volume for oxide, volume in volume_dict.items()}

    return oxides_vf


def init_model(config, oxide_params, reference):
    oxygen, time = reference[:2]
    oxide_models = {}
    global_shift_delta = torch.nn.parameter.Parameter(torch.tensor([0], dtype=torch.float64), requires_grad=True)
    for oxide_name, oxide in oxide_params.items():
        # oxide['Tm'] = oxide["Tb"] + 173
        e_var = config['e_var']
        e_scale = np.exp(config['e_scale'])
        v_var = max(0, oxygen[np.argmin(np.abs(time - oxide['Tm']))])
        v_scale = np.exp(config['v_scale'])
        t_max_delta = config['t_max_delta']
        t_max_delta_scale = np.exp(config['t_max_delta_scale'])
        t_beg_delta = config['t_beg_delta']
        t_beg_delta_scale = np.exp(config['t_beg_delta_scale'])
        if v_var < 0.01:
            continue
        oxide_models[oxide_name] = OxideModel(
            {'init': e_var,
             'scale': e_scale, },
            {'init': v_var,
             'scale': v_scale, },
            {'init': oxide['Tm'],
             'delta': t_max_delta,
             'delta_scale': t_max_delta_scale},
            {'init': oxide["Tb"],
             'delta': t_beg_delta,
             'delta_scale': t_beg_delta_scale},
            model=config['model']
        )

    return torch.nn.ModuleDict(oxide_models), global_shift_delta


def get_oxides_structure(oxide_params, elements_table):
    for oxide_name in oxide_params.keys():
        key = oxide_name
        structure = {}
        for l in range(2, 0, -1):
            for element in filter(lambda x: len(x) == l, elements_table.keys()):
                idx = oxide_name.find(element) + l
                if idx != -1 + l:
                    if idx == len(oxide_name) or not oxide_name[idx].isdigit():
                        structure[element] = 1
                    else:
                        structure[element] = ''
                        while idx < len(oxide_name) and oxide_name[idx].isdigit():
                            structure[element] += oxide_name[idx]
                            oxide_name = oxide_name[:idx] + oxide_name[idx + 1:]
                        structure[element] = int(structure[element])
                    oxide_name = oxide_name.replace(element, '')

        oxide_params[key]['structure'] = structure


def main_process(reference_path, oxide_params, config, chemistry):
    # Set matplotlib to use Agg backend (non-interactive)
    import matplotlib
    matplotlib.use('Agg')  # This must be done before importing pyplot
    from matplotlib import pyplot as plt

    chem_data = load_chem_data('chem.txt')
    oxides = load_oxides('oxid.txt')
    with open('elements_table.json', 'r') as f:
        elements_table = json.load(f)
    oxides_tb = get_tb_tm(chem_data, oxides, chemistry, dRamp=2.0)
    print(sorted(list(oxides_tb.items()), key=lambda x: x[1][0]))
    config['guaranteed_oxides'] = oxide_params['guaranteed_oxides']
    print(config['guaranteed_oxides'])
    oxide_params = {key: {} for key in chain.from_iterable(oxide_params.values())}
    for key in oxide_params.keys():
        oxide_params[key]['Tb'] = oxides_tb[key][0]
        oxide_params[key]['Tm'] = oxides_tb[key][1]
    get_oxides_structure(oxide_params, elements_table)

    reference = reference_read(reference_path, config)

    oxide_models, global_shift_delta = init_model(config, oxide_params, reference)
    import torch.optim

    optimizer = getattr(torch.optim, config['optim'])([x for x in oxide_models.parameters()] + [global_shift_delta],
                                                      **config['optim_params'])

    fig = draw_simple(None, temperature_shift(global_shift_delta), reference[:2], config, 'Нормированая входная зависимость')
    fig.savefig('input.png', format='png', dpi=100)
    fig = draw_simple(oxide_models, temperature_shift(global_shift_delta), reference[:2], config, 'Первое приближение')
    fig.savefig('first_approximation.png', format='png', dpi=100)

    train(oxide_models, global_shift_delta, reference[:2], optimizer, config)
    oxides_oxygen = get_oxide_ppm(oxide_models, global_shift_delta, reference, config)
    oxides_vf = get_oxides_vf(oxides_oxygen, elements_table, oxide_params)
    fig = draw_simple(oxide_models, temperature_shift(global_shift_delta), reference[:2], config, 'Разложение на составляющие')
    fig.savefig('result.png', format='png', dpi=100)

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=70)
    buf.seek(0)
    image = PILImage.open(buf)
    plt.close(fig)
    global_shift = temperature_shift(global_shift_delta)
    oxides_result = {}
    for oxide_name, oxide in oxide_models.items():
        oxides_result[oxide_name] = {}
        oxides_result[oxide_name]['ppm'] = oxides_oxygen[oxide_name]
        oxides_result[oxide_name]['vf'] = oxides_vf[oxide_name]
        oxides_result[oxide_name]['Tb'] = oxide.get_t_beg().item() + global_shift.item()
        oxides_result[oxide_name]['Tm'] = oxide.get_t_max().item() + global_shift.item()
        oxides_result[oxide_name]['Vm'] = oxide.get_v_max().item()
        oxides_result[oxide_name]['E'] = oxide.get_E().item()
    return oxides_result, image


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--reference_path', type=str, default=None)
    parser.add_argument('--oxide_params', type=str, default='params.json')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    with open(args.oxide_params, 'r') as f:
        oxide_params = json.load(f)
    with open('elements_table.json', 'r') as f:
        elements_table = json.load(f)
    if args.reference_path is None:
        args.reference_path = input('Enter path to FGA file: ')
    reference = reference_read(args.reference_path, config)

    oxide_models, global_shift_delta = init_model(config, oxide_params, reference)
    import torch.optim

    optimizer = getattr(torch.optim, config['optim'])([x for x in oxide_models.parameters()] + [global_shift_delta],
                                                      **config['optim_params'])

    train(oxide_models, global_shift_delta, reference[:2], optimizer, config)
    oxides_oxygen = get_oxide_ppm(oxide_models, global_shift_delta, reference, config)
    print(oxides_oxygen)
    oxides_vf = get_oxides_vf(oxides_oxygen, elements_table, oxide_params)
    print(oxides_vf)
    fig = draw_simple(oxide_models, temperature_shift(global_shift_delta), reference[:2], config)
    fig.savefig('result.png', format='png', dpi=70)
