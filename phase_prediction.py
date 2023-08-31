# phase_prediction.py
#
# This file contains code relevant for the phase prediction of HEAs

from joblib import load
from tensorflow.keras.models import load_model
from utils import calculate_params, extract_alloy
import pandas as pd

# The model with the best Hamming Accuracy Score during cross-validation testing is loaded for phase prediction
# Standard Scalar and One-Hot Encoder is load for data processing before prediction
# Threshold value is calculated after learning phase of phase prediction model by trail and error method
# phases array specify the 10 single phases that are being predicted individually or in a combination
phase_prediction_model = load_model("Phase_Classification_Files/phase-classification-nn1-model")
phase_standard_scalar = load("Phase_Classification_Files/Standard_Scalar.bin")
phase_one_hot_encoder = load("Phase_Classification_Files/One_Hot_Encoder.bin")
threshold = 0.438
phases = ['B2', 'BCC', 'FCC', 'Im', 'SS', 'sigma', 'BCC1', 'BCC2', 'FCC1', 'FCC2']


# predict_phase function takes input a list of alloys and the processing route followed during the synthesis of the
# alloys
def predict_phase(alloys, processing_route):
    alloy_elements, elemental_concentration = extract_alloy(alloys)
    return predict_phase_direct(alloy_elements, elemental_concentration, processing_route)


# predict_phase_direct function takes input an array of elements of each alloy with respective elemental concentration
# and the processing route followed during the synthesis of the alloys
# This is an internal function used by predict_phase
# It outputs a DataFrame in which each entry represent the predicted phase for each alloy given as input in the form
# of 10-dimensional vector where 1 represents the presence of the phase corresponding to phase array
def predict_phase_direct(alloy_elements, elemental_concentration, processing_route):
    phase_prediction_input = calculate_params(alloy_elements, elemental_concentration)

    processing_route_onehot = phase_one_hot_encoder.transform([[processing_route]])[0]

    phase_prediction_input[phase_prediction_input.columns[1:]] = phase_standard_scalar.transform(
        phase_prediction_input[phase_prediction_input.columns[1:]])
    phase_prediction_input[phase_one_hot_encoder.categories_[0]] = processing_route_onehot
    predicted_phase = phase_prediction_model.predict(phase_prediction_input[phase_prediction_input.columns[1:]],
                                                     verbose=0)
    predicted_phase = predicted_phase > threshold
    predicted_phase = predicted_phase.astype("int")
    return pd.DataFrame(predicted_phase, columns=phases)
