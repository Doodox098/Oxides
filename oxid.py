import json
import math
import xml.etree.ElementTree as ET
from collections import namedtuple
from functools import partial

from hyperopt import hp, fmin, tpe, space_eval
import hyperopt
import numpy as np

# Constants
R = 8.31441  # J/(mol·K)

# Data structures
Oxide = namedtuple('Oxide', ['name', 'base1', 'base2', 'x0', 'x1', 'a', 'b', 'c'])
HentParam = namedtuple('HentParam', ['dH', 'dS'])
EpsParam = namedtuple('EpsParam', ['a', 'b'])
McParam = namedtuple('McParam', ['value'])


class Component:
    def __init__(self, name, m_part=0.0):
        self.name = name
        self.m_part = m_part

    def __eq__(self, other):
        return self.name == other.name

    def isEmpty(self):
        return self.name == ""


class Composition(dict):
    pass


class CChemData:
    def __init__(self):
        self.eps_params = {}  # {(base, comp, ox): EpsParam}
        self.mc_params = {}  # {(base, comp): McParam}
        self.hent_params = {}  # {(base, comp): HentParam}
        self.base_params = {}  # {base: EpsParam}

    def GetEpsParam(self, base, comp, ox):
        return self.eps_params.get((base, comp, ox), EpsParam(0, 0))

    def GetMcParam(self, base, comp):
        return self.mc_params.get((base, comp), McParam(0))

    def GetHentParam(self, base, comp):
        return self.hent_params.get((base, comp), HentParam(0, 0))

    def GetBaseParam(self, base):
        return self.base_params.get(base, EpsParam(0, 0))


def load_chem_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    chem_data = CChemData()

    for base in root.findall('base'):
        base_name = base.get('name')

        # Load base_eps parameters
        base_eps = base.find('base_eps')
        if base_eps is not None:
            a = float(base_eps.get('a', 0))
            b = float(base_eps.get('b', 0))
            chem_data.base_params[base_name] = EpsParam(a, b)

        # Load epsilon parameters
        for eps in base.findall('epsilon'):
            comp = eps.get('name')[:2]  # First two characters are the component
            ox = eps.get('name')[2:]  # Rest is the oxide part
            a = float(eps.get('a', 0))
            b = float(eps.get('b', 0))
            chem_data.eps_params[(base_name, comp, ox)] = EpsParam(a, b)

        # Load mc parameters
        for mc in base.findall('mc'):
            comp = mc.get('name')
            value = float(mc.text)
            chem_data.mc_params[(base_name, comp)] = McParam(value)

        # Load hent parameters
        for hent in base.findall('oxid/hent'):
            comp = hent.get('name')
            dH = float(hent.get('h', 0))
            dS = float(hent.get('s', 0))
            chem_data.hent_params[(base_name, comp)] = HentParam(dH, dS)

    return chem_data


def load_oxides(oxide_file):
    oxides = []
    with open(oxide_file, 'r') as f:
        for line in f:
            if line.startswith('Oxide') or not line.strip():
                continue

            parts = line.strip().split(';')
            name = parts[0]
            base1 = parts[1]
            x0 = float(parts[2])
            base2 = parts[3]
            x1 = 0.0 if parts[4] == '*' else float(parts[4])
            a = float(parts[5])
            b = float(parts[6])
            c = float(parts[7])

            oxides.append(Oxide(name, base1, base2, x0, x1, a, b, c))

    return oxides


def count_oxide_func(Temp, ox, ch_data, base, comps, Pco):
    kKTemp = Temp / 1000.0  # Convert to kK

    # Get epsilon parameters for base components
    p_ab = ch_data.GetEpsParam(base.name, ox.base1, "C")
    epsRC = p_ab.a / Temp + p_ab.b if p_ab else 0

    epsPC = 0.0
    if ox.base2 != '*':
        p_ab = ch_data.GetEpsParam(base.name, ox.base2, "C")
        epsPC = p_ab.a / Temp + p_ab.b if p_ab else 0

    # Get base parameters
    p_ab = ch_data.GetBaseParam(base.name)
    xC = math.exp((p_ab.a / Temp + p_ab.b) * math.log(10.0)) if p_ab else 0

    sum1 = 0.0
    sum2 = 0.0
    xR = 1.0
    xP = 1.0

    # Calculate sums and mole fractions
    for comp_name, comp in comps.items():
        if comp.name == ox.base1:
            xR = comp.m_part
        if ox.base2 != '*' and comp.name == ox.base2:
            xP = comp.m_part
        if comp.name == base.name or comp.name == "C":
            continue

        # For C
        p_mCi = ch_data.GetMcParam(base.name, comp.name)
        xC += p_mCi.value * comp.m_part if p_mCi else 0

        p_ab = ch_data.GetEpsParam(base.name, comp.name, ox.base1)
        epsRi = p_ab.a / Temp + p_ab.b if p_ab else 0
        sum1 += epsRi * comp.m_part

        if ox.base2 == '*':
            continue

        p_ab = ch_data.GetEpsParam(base.name, comp.name, ox.base2)
        epsRi = p_ab.a / Temp + p_ab.b if p_ab else 0
        sum2 += epsRi * comp.m_part

    # Calculate gammaR and gammaP
    p_hp = ch_data.GetHentParam(base.name, ox.base1)
    gammaR = p_hp.dH / (R * Temp) - p_hp.dS / R if p_hp else 0
    gammaR += epsRC * xC
    gammaR += sum1

    gammaP = 0.0
    if ox.base2 != '*':
        p_hp = ch_data.GetHentParam(base.name, ox.base2)
        gammaP = p_hp.dH / (R * Temp) - p_hp.dS / R if p_hp else 0
        gammaP += epsPC * xC
        gammaP += sum2

    # Calculate logK
    logK = ox.a + ox.b / kKTemp + ox.c * math.log(kKTemp)

    # Calculate result based on oxide type
    if ox.base1 == "Mg" or (ox.base2 != '*' and ox.base2 == "Mg"):
        res = logK - ox.x0 * math.log(ox.x0) - ox.x1 * (math.log(xP) + gammaP) - \
              (1 - ox.x1) * math.log(Pco) - (1 - ox.x0 - ox.x1) * math.log(1 - ox.x0 - ox.x1) - \
              (1 - ox.x1) * math.log(1 - ox.x1)
    elif ox.base1 == "Ca" or (ox.base2 != '*' and ox.base2 == "Ca"):
        res = logK - ox.x1 * (math.log(xP) + gammaP) - (1 - ox.x0 - ox.x1) * math.log(Pco)
    else:
        res = logK - ox.x0 * (math.log(xR) + gammaR) - ox.x1 * (math.log(xP) + gammaP) - \
              (1 - ox.x0 - ox.x1) * math.log(Pco)

    return res


def calc_oxide_tb(T1, T2, ox, ch_data, base, comps, Pco, eps=1.0):
    t1 = T1
    t2 = T2

    while t2 - t1 > eps:
        G1 = count_oxide_func(t1, ox, ch_data, base, comps, Pco)
        G2 = count_oxide_func(t2, ox, ch_data, base, comps, Pco)
        G = count_oxide_func((t2 + t1) / 2.0, ox, ch_data, base, comps, Pco)

        if G1 * G < 0:
            t2 = (t2 + t1) / 2.0
        else:
            t1 = (t2 + t1) / 2.0

    return (t2 + t1) / 2.0

def get_tb(chem_data, oxides, composition_percent):
    comps = Composition()
    for elem, percent in composition_percent.items():
        comps[elem] = Component(elem, percent / 100)

    # Base component (typically Fe)
    base = Component('Fe', 1 - sum(composition_percent.values()) / 100)

    # CO partial pressure (atm)
    Pco = 1.85

    # Temperature range for search (K)
    T1 = 300  # Lower bound
    T2 = 2500  # Upper bound

    # Calculate reduction temperatures for all oxides
    results = {}
    for ox in oxides:
        try:
            tb = calc_oxide_tb(T1, T2, ox, chem_data, base, comps, Pco)
            results[ox.name] = tb
        except Exception as e:
            results[ox.name] = None
    return results


def get_tb_tm(chem_data, oxides, composition_percent, dRamp=1.0):
    comps = Composition()
    for elem, percent in composition_percent.items():
        comps[elem] = Component(elem, percent / 100)

    # Base component (typically Fe)
    base = Component('Fe', 1 - sum(composition_percent.values()) / 100)

    # CO partial pressure (atm)
    Pco = 1.85

    # Temperature range for search (K)
    T1 = 300  # Lower bound
    T2 = 2500  # Upper bound

    # Calculate reduction temperatures for all oxides
    results = {}
    for ox in oxides:
        try:
            tb = calc_oxide_tb(T1, T2, ox, chem_data, base, comps, Pco)
            results[ox.name] = tb, tb + 215.869 * dRamp / (1.315 + dRamp)
        except Exception as e:
            results[ox.name] = None
    return results


def CalcOxideTm(ox, dRamp):
    deltaT = 215.869 * dRamp / (1.315 + dRamp)
    ox.m_Tm = ox.m_Tb + deltaT


def objective(params, chem_data, oxides, goal):
    res = get_tb(chem_data, oxides, params)
    return np.mean([abs(res[oxide] - goal[oxide]['Tb']) for oxide in goal.keys() if oxide in res.keys()])

def main():
    # Load data
    chem_data = load_chem_data('chem.txt')
    oxides = load_oxides('oxid.txt')
    oxide_params = {
      "MnSiO3": {
        "Tb": 1497,
        "Tm": 1627
      },
      "Mn2SiO4": {
        "Tb": 1503,
        "Tm": 1634
      },
      "SiO2": {
        "Tb": 1593,
        "Tm": 1743
      },
      "Al2TiO5": {
        "Tb": 1722,
        "Tm": 1852
      },
      "Al2SiO5": {
        "Tb": 1768,
        "Tm": 1898
      },
      "Al6Si2O13": {
        "Tb": 1801,
        "Tm": 1932
      },
      "MnAl2O4": {
        "Tb": 1807,
        "Tm": 1937
      },
      "MgSiO3": {
        "Tb": 1808,
        "Tm": 1938
      },
      "Al2O3": {
        "Tb": 1864,
        "Tm": 1994
      },
      "MgAl2O4": {
        "Tb": 1895,
        "Tm": 2025
      },
      "Mg2SiO4": {
        "Tb": 1925,
        "Tm": 2056
      },
      "CaAl4O7": {
        "Tb": 1974,
        "Tm": 2104
      },
      "CaAl2O4": {
        "Tb": 2009,
        "Tm": 2140
      },
      "Ca2SiO4": {
        "Tb": 2097,
        "Tm": 2227
      },
      "MgO": {
        "Tb": 2107,
        "Tm": 2237
      },
      "CaO": {
        "Tb": 2165,
        "Tm": 2295
      }
    }

    # This should be provided as percentages of each element in the metal
    # composition_percent = {
    #     'Al': 0.001, 'Si': 0.27, 'Mn': 0.3, 'Cr': 1.48, 'Ti': 0.001, 'Mg': 0.001, 'Ca': 0.001,
    # }
    composition_percent = {
      "Al": 0.075,
      "Ca": 0.006,
      "Cr": 0.0013,
      "Mg": 0.38,
      "Mn": 0.14,
      "Si": 0.59,
      "Ti": 0.0022
    }
    space = {}
    space['Al'] = hp.loguniform(f'Al', np.log(1e-5), np.log(5.))
    space['Si'] = hp.loguniform(f'Si', np.log(1e-5), np.log(5.))
    space['Mn'] = hp.loguniform(f'Mn', np.log(1e-5), np.log(5.))
    space['Ti'] = hp.loguniform(f'Ti', np.log(1e-5), np.log(5.))
    space['Cr'] = hp.loguniform(f'Cr', np.log(1e-5), np.log(5.))
    space['Ca'] = hp.loguniform(f'Ca', np.log(1e-5), np.log(5.))
    space['Mg'] = hp.loguniform(f'Mg', np.log(1e-5), np.log(5.))

    # trials_obj = hyperopt.Trials()
    # best = fmin(
    #     fn=partial(objective, chem_data=chem_data, oxides=oxides, goal=oxide_params),
    #     space=space,
    #     algo=tpe.suggest,
    #     max_evals=4000,
    #     trials=trials_obj
    # )
    # print(best)
    # for elem, percent in best.items():
    #     print(elem, percent)
    # for oxide, tb in sorted(get_tb_tm(chem_data, oxides, best).items(), key=lambda x: x[1] if x[1] is not None else float('inf')):
    #     if oxide in oxide_params:
    #         print(oxide, tb, oxide_params[oxide]['Tb'])
    res = get_tb_tm(chem_data, oxides, composition_percent, dRamp=2.0)
    for oxide, (tb, tm) in sorted(res.items(), key=lambda x: x[1] if x[1] is not None else float('inf')):
        if oxide in oxide_params:
            print(oxide, round(tm), oxide_params[oxide]['Tm'], round(tb), oxide_params[oxide]['Tb'])
        # else:
        #     print(oxide, tm)
    print(np.mean([abs(round(res[oxide][1]) - oxide_params[oxide]['Tm']) for oxide in oxide_params.keys() if oxide in res.keys()]))
if __name__ == "__main__":
    main()
