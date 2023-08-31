# hea_identification.py
#
# This file contains code relevant for the identifying whether an alloy is HEA or not

from joblib import load
from utils import calculate_params, extract_alloy

# The model with the best Recall and Accuracy Score during cross-validation testing is loaded for phase prediction
# Standard Scalar is load for data processing before prediction
hea_identification_model = load("HEA_Identification_Files/HEA_Identification_Model.bin")
standard_scalar = load("HEA_Identification_Files/StandardScalar.bin")


# identify_alloy function takes input a list of alloys for identification
def identify_alloy(alloys):
    alloy_elements, elemental_concentration = extract_alloy(alloys)
    return identify_alloy_direct(alloy_elements, elemental_concentration)


# identify_alloy_direct function takes input an array of elements of each alloy with respective elemental concentration
# This is an internal function used by identify_alloy
def identify_alloy_direct(alloy_elements, elemental_concentration):
    identification_input = calculate_params(alloy_elements, elemental_concentration)

    identification_input[identification_input.columns[1:]] = standard_scalar.transform(
        identification_input[identification_input.columns[1:]])
    prediction = hea_identification_model.predict(identification_input[identification_input.columns[1:]], verbose=0)
    return prediction

