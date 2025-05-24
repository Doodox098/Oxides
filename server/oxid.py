import json
import math
import xml.etree.ElementTree as ET
from collections import namedtuple
from functools import partial

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


def calculate_gibbs_energy(Temp, ox, ch_data, base, comps, Pco):
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
        G1 = calculate_gibbs_energy(t1, ox, ch_data, base, comps, Pco)
        G2 = calculate_gibbs_energy(t2, ox, ch_data, base, comps, Pco)
        G = calculate_gibbs_energy((t2 + t1) / 2.0, ox, ch_data, base, comps, Pco)

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
