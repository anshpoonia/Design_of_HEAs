import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from hardness_prediction import predict_hardness_direct
from phase_prediction import predict_phase_direct
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def cost_function(concentration, alloy):
    concentration = concentration.transpose()

    if np.shape(concentration) != (len(alloy),):
        alloy = np.tile(alloy, (concentration.shape[0], 1))
    else:
        concentration = np.array([concentration])
        alloy = np.array([alloy])

    predicted_hardness = predict_hardness_direct(alloy, concentration, "AM")
    con_deviation = np.abs(1.0 - np.sum(concentration, axis=1))
    cost = - (predicted_hardness - 10000 * con_deviation)
    cost[np.min(concentration, axis=1) < 0.05] = 1e10
    return cost


def optimize(alloyArr):
    alloyArr = np.array(alloyArr)
    con_limits = [(0.05, 0.35)] * len(alloyArr)
    solver = differential_evolution(cost_function, con_limits, args=(alloyArr,), strategy='best2bin', maxiter=100,
                                    popsize=15, mutation=(0.0, 1.0), recombination=0.7, init="sobol", tol=0.01,
                                    polish=True, vectorized=True, seed=42, disp=True)
    return predict_hardness_direct([alloyArr], [solver.x], "AM")[0], predict_phase_direct([alloyArr], [solver.x], "AM")


if __name__ == "__main__":
    hardness, phase = optimize(["Al", "Cr", "Co", "Cu"])
    print(hardness)
    print(phase)
