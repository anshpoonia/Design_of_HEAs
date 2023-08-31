# hardness_prediction.py
#
# This file contains code relevant for the hardness prediction of HEAs

from catboost import CatBoostRegressor
from joblib import load
from phase_prediction import predict_phase_direct
from utils import calculate_params, extract_alloy

# The model with best R2 Score during cross-validation testing is loaded for hardness prediction
# Standard Scalar is load for data processing before prediction
standard_scalar = load("Hardness_Prediction_Files/Standard_Scalar.bin")
prediction_model = CatBoostRegressor().load_model("Hardness_Prediction_Files/hardness_prediction_model")


# predict_hardness function takes input a list of alloys and the processing route followed during the synthesis of the
# alloys (processing route is necessary for prediction of phase)
def predict_hardness(alloys, processing_route):
    alloy_elements, elemental_concentration = extract_alloy(alloys)
    return predict_hardness_direct(alloy_elements, elemental_concentration, processing_route)


# predict_hardness_direct function takes input an array of elements of each alloy with respective elemental
# concentration and the processing route followed during the synthesis of the alloys
# This is an internal function used by predict_hardness
def predict_hardness_direct(alloy_elements, elemental_concentration, processing_route):
    thermodynamic_params = calculate_params(alloy_elements, elemental_concentration)

    predicted_phase = predict_phase_direct(alloy_elements, elemental_concentration, processing_route)

    thermodynamic_params[thermodynamic_params.columns[1:]] = standard_scalar.transform(
        thermodynamic_params[thermodynamic_params.columns[1:]])
    thermodynamic_params[predicted_phase.columns] = predicted_phase
    predicted_hardness = prediction_model.predict(thermodynamic_params[thermodynamic_params.columns[1:]])
    return predicted_hardness

