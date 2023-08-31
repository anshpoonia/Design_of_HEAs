# utils.py
#
# The main function of utils.py is to calculate thermodynamic properties of an alloy, and to extract elements and
# elemental concentration in terms of mole fraction for each alloy.
#
# The purpose of each function has been mentioned as per their occurrence.

import numpy as np
import pandas as pd
import re
import math
from sympy.utilities.iterables import combinations
from elemental_database import element_data, enthalpy_of_mixing_arr, elements_arr, columns_names


# calculate_params take an array of elements of each alloy with respective elemental concentration as input
# alloy_elements is 2d array, each row contains an array of constituent elements of an alloy
# elemental_concentration is a 2d array, each row contains an array of concentration of each element of an alloy
# It returns a DataFrame containing thermodynamic properties for each alloy given as input
def calculate_params(alloy_elements, elemental_concentration):
    td_params_df = pd.DataFrame(columns=get_col_names())

    for single_alloy_elements, single_concentration in zip(alloy_elements, elemental_concentration):
        single_concentration = np.array([single_concentration])
        specifications_arr = []
        element_index = []

        for i in single_alloy_elements:
            j, = np.where(elements_arr == i)
            element_index.append(j[0])
            specifications_arr.append(element_data.loc[i])

        element_index = np.array(element_index)
        specifications_arr = np.array(specifications_arr).transpose()
        binary_combination = np.array(list(combinations(element_index, 2)))

        binary_enthalpy_of_mixing = []
        for i in range(binary_combination.shape[0]):
            binary_enthalpy_of_mixing.append(
                enthalpy_of_mixing_arr[
                    min(binary_combination[i, 0], binary_combination[i, 1]),
                    max(binary_combination[i, 0], binary_combination[i, 1])
                ]
            )

        binary_enthalpy_of_mixing = np.array(binary_enthalpy_of_mixing)

        thermodynamic_parameters = params(single_concentration, specifications_arr, binary_enthalpy_of_mixing)
        td_params_df.loc[len(td_params_df.index)] = thermodynamic_parameters

    return td_params_df


# Returns a list of alloy the thermodynamic parameters that are calculated
def get_col_names():
    average_col_name = []
    delta_col_name = []
    for i in columns_names:
        average_col_name.append("Average " + i)
        delta_col_name.append("Delta " + i)

    remaining_col_names = [
        "Atomic Volume",
        "Mixing Entropy",
        "Mixing Enthalpy",
        "Gibbs Phase Energy",
        "Delta",
        "Omega",
        "Lambda"
    ]

    col_names = ["Concentration Combination"]

    for i in average_col_name:
        col_names.append(i)

    for i in delta_col_name:
        col_names.append(i)

    for i in remaining_col_names:
        col_names.append(i)

    return col_names


# params function is internally used to calculate thermodynamic properties by calculate_params
#
# concentration_arr is a list of different concentration for which we have to calculate the parameters
# specifications_arr is the list of all the properties of selected elements

# concentration_arr shape will be  m x n   ---> where m is the number of different concentrations
#                                          ---> and n is the number of selected elements
# specifications_arr shape will be p x n   ---> where p is the number of properties of each element

# properties currently included in the specifications_arr
#  0 --> Atomic Radius
#  1 --> Melting Point
#  2 --> Boiling Point
#  3 --> Pauling Electronegativity
#  4 --> Allen Electronegativity
#  5 --> Valance Electron Concentration
#  6 --> Itinerant Electron Per Atom
#  7 --> Atomic Weight
#  8 --> Density
#  9 --> Molar Heat Capacity
# 10 --> Thermal Conductivity

# binary_enthalpy_of_mixing is a list of all the enthalpy of the binary combinations of the chosen elements
# binary_enthalpy_of_mixing shape will be  n * (n-1) / 2    ---> where n is number of selected elements
def params(concentration_arr, specifications_arr, binary_enthalpy_of_mixing):
    # gas constant
    R = 0.00831

    # epsilon - small value
    epsilon = 1 * math.pow(10, -20)
    specifications_arr[6] = specifications_arr[6] + epsilon

    # average_properties will contain the average of all the properties for all the provided concentrations
    # its shape will be   m x p
    average_properties = np.matmul(concentration_arr, np.transpose(specifications_arr))

    # delta_properties will contain the delta/difference of all the properties for all concentration
    # its shape will be   m x p
    # it will be computed individually for each combination of concentration ( in the for loop )
    delta_properties = []

    # atomic_volume will contain the atomic volume for all the given concentrations
    # its shape will be   m x 1
    atomic_volume = np.matmul(concentration_arr, ((4 / 3) * math.pi * np.power(specifications_arr[0], 3)))

    # mixing entropy will contain the mixing entropy for all the given concentrations
    # its shape will be   m x 1
    mixing_entropy = -R * np.sum(concentration_arr * np.log(concentration_arr), axis=1)

    # for computing mixing enthalpy, we have to sum over all the possible binary enthalpy of mixing after multiplication
    # with respective concentration of both the elements
    # so, first we will compute all the possible combinations for the binary mixing them multiply them with the
    # respective binary enthalpy of mixing
    # binary_combinations will contain all the binary combinations for the given elements
    # its shape will be    m x ( n!/ (2 * (n-2)!))
    binary_combinations = []

    # lambda_param represents the number to quantify lattice distortion in the alloy for each concentration level
    # min_r represents the smallest atomic radius in the chosen elements, it stays same irrespective of concentration
    # max_r represents the largest atomic radius
    # its shape is   m x 1
    min_r = np.min(specifications_arr[0])
    max_r = np.max(specifications_arr[0])
    min_r_sq = np.power((min_r + average_properties[:, 0]), 2)
    max_r_sq = np.power((max_r + average_properties[:, 0]), 2)
    lambda_num = np.power(((min_r_sq - np.power(average_properties[:, 0], 2)) / min_r_sq), 1 / 2)
    lambda_den = np.power(((max_r_sq - np.power(average_properties[:, 0], 2)) / max_r_sq), 1 / 2)
    lambda_param = (1 - lambda_num) / (1 - lambda_den)

    for i in range(concentration_arr.shape[0]):
        per_concentration_average_prop = average_properties[i]

        per_concentration_delta_prop = np.power(
            (np.matmul(concentration_arr[i],
                       np.power((1 - specifications_arr.transpose() / per_concentration_average_prop), 2))), 1 / 2)

        per_concentration_binary_comb = np.array(list(combinations(concentration_arr[i], 2)))
        per_concentration_binary_comb = np.prod(per_concentration_binary_comb, axis=1)

        delta_properties.append(per_concentration_delta_prop)
        binary_combinations.append(per_concentration_binary_comb)

    delta_properties = np.array(delta_properties)
    binary_combinations = np.array(binary_combinations)

    # delta is computed by dividing mixing entropy by the square of the delta/difference of the atomic radius
    # its shape will be   m x 1
    delta = mixing_entropy / np.power(delta_properties[:, 0], 2)

    # mixing_enthalpy will contain the mixing enthalpy for each concentration computed by summing up the binary
    # enthalpy of mixing
    # its shape will be   m x 1
    mixing_enthalpy = 4 * np.matmul(binary_combinations, binary_enthalpy_of_mixing) + epsilon

    # omega will contain the combination effect computed by dividing the product of average melting point and mixing
    # entropy by mixing enthalpy
    # its shape will be   m x 1
    omega = average_properties[:, 1] * mixing_entropy / np.abs(mixing_enthalpy)

    # gibbs_phase will have the gibbs phase energy computed by multiplying average melting temperature with mixing
    # entropy and subtracting it from mixing enthalpy
    # its shape will be   m x 1
    gibbs_phase = mixing_enthalpy - average_properties[:, 1] * mixing_entropy

    thermodynamic_parameters = [str(concentration_arr[0])] + list(average_properties[0]) + list(
        delta_properties[0]) + list(atomic_volume) + list(mixing_entropy) + list(mixing_enthalpy) + list(
        gibbs_phase) + list(delta) + list(omega) + list(lambda_param)

    return thermodynamic_parameters


# weights function extracts the elemental concentration from string form of alloy
def weights(string):
    weight = re.findall(r"[-+]?(?:\d*\.*\d+)", string)
    if len(weight) == 0:
        return float(1)
    return float(weight[0])


weights = np.vectorize(weights)


# names function extracts element names from string form of alloy
def names(string):
    return re.split(r"[-+]?(?:\d*\.*\d+)", string)[0]


names = np.vectorize(names)


# to_mode_fraction function converts elemental concentration to the form of mole fraction
def to_mole_fraction(array):
    total = np.sum(array)
    return array / total


# extract_alloy function take input a list of string form of alloy
# It extracts the element names and elemental concentration from each alloy
def extract_alloy(alloys):
    elements = []
    composition = []
    for alloy in alloys:
        el_string = alloy
        if '(' not in el_string and '[' not in el_string and '{' not in el_string:
            el_arr = np.array(re.findall('[A-Z][^A-Z]*', el_string))
            el_name = names(el_arr)
            el_con = weights(el_arr)
            if all(item in elements_arr for item in el_name):
                if 0.0 not in to_mole_fraction(el_con):
                    elements.append(el_name)
                    composition.append(to_mole_fraction(el_con))
                else:
                    return f"total not zero: {el_string}", "error"
            else:
                return f"not in the list: {el_string}", "error"
        else:
            return f"have bracket: {el_string}", "error"
    return elements, composition
