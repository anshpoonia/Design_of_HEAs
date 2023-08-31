from joblib import load
from tensorflow.keras.models import load_model
from utils import calculate_params, extract_alloy
import pandas as pd

phase_prediction_model = load_model("Phase_Classification_Files/phase-classification-nn1-model")
phase_standard_scalar = load("Phase_Classification_Files/Standard_Scalar.bin")
phase_one_hot_encoder = load("Phase_Classification_Files/One_Hot_Encoder.bin")
threshold = 0.438
phases = ['B2', 'BCC', 'FCC', 'Im', 'SS', 'sigma', 'BCC1', 'BCC2', 'FCC1', 'FCC2']


def predict_phase(alloys, processing_route):
    alloy_elements, elemental_concentration = extract_alloy(alloys)
    return predict_phase_direct(alloy_elements, elemental_concentration, processing_route)


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
