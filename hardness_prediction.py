from catboost import CatBoostRegressor
from joblib import load
from phase_prediction import predict_phase_direct
from utils import calculate_params, extract_alloy

standard_scalar = load("Hardness_Prediction_Files/Standard_Scalar.bin")
prediction_model = CatBoostRegressor().load_model("Hardness_Prediction_Files/hardness_prediction_model")


def predict_hardness(alloys, processing_route):
    alloy_elements, elemental_concentration = extract_alloy(alloys)
    return predict_hardness_direct(alloy_elements, elemental_concentration, processing_route)


def predict_hardness_direct(alloy_elements, elemental_concentration, processing_route):
    thermodynamic_params = calculate_params(alloy_elements, elemental_concentration)

    predicted_phase = predict_phase_direct(alloy_elements, elemental_concentration, processing_route)

    thermodynamic_params[thermodynamic_params.columns[1:]] = standard_scalar.transform(
        thermodynamic_params[thermodynamic_params.columns[1:]])
    thermodynamic_params[predicted_phase.columns] = predicted_phase
    predicted_hardness = prediction_model.predict(thermodynamic_params[thermodynamic_params.columns[1:]])
    return predicted_hardness


if __name__ == "__main__":
    # print(predict_hardness(["Co18Cr7Fe35Ni5V35",
    #                         "Al20Cr5Cu15Fe15Ni5Ti10V30",
    #                         "Al21Cr27Fe29Ni5Mo18",
    #                         "Co33W7Al33Nb24Cr3",
    #                         "Ti18Ni24Ta12Cr22Co24",
    #                         "Co6W9Al36Mo38Ni11",
    #                         "Ni47Co2Ta12Ti9Nb30",
    #                         "Ti44Ni2Nb21Cr21C012",
    #                         "Ti32Nb9Ta1Cr19Co39",
    #                         "Ti39W4Nb31Ta4Co22",
    #                         "Al41.68Cr24.72Fe8.02Ni5.8C019.78",
    #                         "Al44.38Cr31.79Fe11.67Ni12.17",
    #                         "Al1.2Cr17.42Fe25.42Ni28.32Ti27.62",
    #                         "Al44.18Cr18.58Fe12.08Ni11.38V13.78",
    #                         "Co10Cr20Fe30Ni40",
    #                         "Co10Cr20Fe40Ni30",
    #                         "Al41Co20Cr19Fe15Ni5",
    #                         "Al46Co16Cr15Fe15Ni8",
    #                         "Al32Co13Cr33Fe22",
    #                         "Al5.4Cr17.83Fe38.29Mn21.97Ni16.77",
    #                         "Al7.78Cr19.99Fe21.47Ni50.77"], "AM"))
    print(predict_hardness(["Al5.66Co18.87Cr18.87Fe18.87Mn18.87Ni18.87"], "AM"))
