import numpy as np
import pandas as pd

columns_names = ["Atomic Radius", "Melting Point", "Boiling Point",
                 "Pauling Electronegativity", "Allen Electronegativity", "Valance Electron Concentration",
                 "Itinerant Electron Per Atom", "Atomic Weight", "Density", "Molar Heat Capacity",
                 "Thermal Conductivity"]

elements_arr = np.array(
    ["Li", "Be", "B", "C", "N", "Na", "Mg", "Al", "Si", "P", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
     "Cu", "Zn", "Ga", "Ge", "As", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
     "Sb", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"])

atomic_number_arr = np.array(
    [3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83])

atomic_radius_arr = np.array(
    [151.94, 112.80, 82.00, 77.30, 75.00, 185.70, 160.13, 143.17, 115.30, 106.00, 231.00, 197.60, 164.10, 146.15,
     131.60, 124.91, 135.00, 124.12, 125.10, 124.59, 127.80, 139.45, 139.20, 124.00, 115.00, 244.00, 215.20, 180.15,
     160.25, 142.90, 136.26, 133.84, 134.50, 137.54, 144.47, 156.83, 165.90, 162.00, 145.00, 145.20, 265.00, 217.60,
     187.90, 157.75, 143.00, 136.70, 137.50, 135.23, 135.73, 138.70, 144.20, 150.00, 171.60, 174.97, 160.00])

melting_point_arr = np.array(
    [454, 1560, 2348, 4742, 63, 371, 923, 933, 1687, 317, 337, 1115, 1814, 1941, 2183, 2180, 1519, 1811, 1768, 1728,
     1358, 693, 303, 1211, 1090, 312, 1050, 1795, 2128, 2750, 2896, 2607, 2237, 1828, 1235, 594, 430, 505, 904, 723,
     302, 1000, 1193, 2506, 3290, 3695, 3458, 3306, 2719, 2041, 1337, 234, 577, 600, 544])

boiling_point_arr = np.array(
    [1615, 2742, 4200, 4300, 77, 1156, 1363, 2792, 3538, 550, 1032, 1757, 3103, 3560, 3680, 2944, 2334, 3134, 3200,
     3186, 2835, 1180, 2477, 3106, 887, 961, 1655, 3609, 4682, 5017, 4912, 4538, 4423, 3968, 3236, 2435, 1040, 2345,
     2875, 1860, 944, 2143, 3737, 4876, 5731, 5930, 5869, 5285, 4701, 4098, 3129, 630, 1746, 2022, 1837])

pauling_electronegativity_arr = np.array(
    [0.98, 1.57, 2.04, 2.55, 3.04, 0.93, 1.31, 1.61, 1.90, 2.19, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88,
     1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96,
     2.05, 2.10, 0.79, 0.89, 1.10, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02])

allen_electronegativity_arr = np.array(
    [0.91, 1.58, 2.05, 2.54, 3.07, 0.87, 1.29, 1.61, 1.92, 2.25, 0.73, 1.03, 1.19, 1.38, 1.53, 1.65, 1.75, 1.80, 1.84,
     1.88, 1.85, 1.59, 1.76, 1.99, 2.21, 0.71, 0.96, 1.12, 1.32, 1.41, 1.47, 1.51, 1.54, 1.56, 1.58, 1.87, 1.52, 1.66,
     1.82, 1.98, 0.66, 0.88, 0.00, 1.16, 1.34, 1.47, 1.60, 1.65, 1.68, 1.72, 1.92, 1.77, 1.79, 1.85, 2.01])

valance_electron_concentration_arr = np.array(
    [1.00, 2.00, 3.00, 4.00, 5.00, 1.00, 2.00, 3.00, 4.00, 5.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00,
     10.00, 11.00, 12.00, 3.00, 4.00, 5.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00, 9.00, 10.00, 11.00, 12.00, 3.00,
     4.00, 5.00, 6.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 5.00])

itinerant_electrons_arr = np.array(
    [0.00, 0.00, 3.00, 0.00, 0.00, 0.00, 0.00, 3.00, 4.00, 0.00, 0.00, 0.00, 0.00, 2.00, 2.00, 1.00, 2.00, 2.00, 2.00,
     2.00, 1.00, 2.00, 0.00, 3.00, 0.00, 0.00, 0.00, 2.00, 2.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     4.00, 0.00, 0.00, 0.00, 0.00, 2.00, 2.00, 2.00, 2.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00])

atomic_weight_arr = np.array(
    [6.94, 9.01, 10.81, 12.01, 14.01, 22.99, 24.31, 26.98, 28.09, 30.97, 39.10, 40.08, 44.96, 47.87, 50.94, 52.00,
     54.94, 55.85, 58.93, 58.69, 63.55, 65.38, 69.72, 72.63, 74.92, 85.47, 87.62, 88.91, 91.22, 92.91, 95.95, 96.91,
     101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 132.91, 137.33, 138.91, 178.49, 180.95, 183.84,
     186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.20, 208.98])

density_arr = np.array(
    [0.53, 1.86, 2.46, 2.27, 1.03, 0.97, 1.74, 2.70, 2.33, 1.82, 0.85, 1.53, 3.00, 4.50, 6.12, 7.19, 7.47, 7.88, 8.84,
     8.91, 8.94, 7.14, 5.91, 5.33, 5.79, 1.53, 2.58, 4.47, 6.51, 8.58, 10.23, 12.37, 12.43, 12.43, 10.50, 8.65, 7.29,
     7.29, 6.69, 6.24, 1.87, 3.60, 6.20, 13.28, 16.68, 19.41, 21.02, 22.59, 22.56, 21.46, 19.29, 14.24, 11.87, 11.35,
     9.81])

molar_heat_capacity_arr = np.array(
    [24.86, 16.44, 11.09, 8.52, 29.12, 28.20, 24.87, 24.20, 19.79, 21.20, 29.60, 25.93, 25.52, 25.06, 24.89, 23.35,
     26.32, 25.10, 24.81, 26.07, 24.44, 25.39, 25.86, 23.22, 24.64, 21.06, 26.40, 26.53, 25.36, 24.60, 24.06, 24.27,
     24.06, 24.98, 25.98, 25.35, 26.02, 26.74, 27.11, 25.23, 32.21, 28.07, 27.11, 25.73, 25.36, 24.27, 25.48, 24.70,
     25.10, 25.86, 25.42, 27.98, 26.32, 26.65, 25.52])

thermal_conductivity_arr = np.array(
    [84.70, 200.00, 27.00, 129.00, 0.03, 140.00, 156.00, 237.00, 148.00, 0.24, 100.00, 200.00, 16.00, 21.90, 30.70,
     93.70, 7.82, 80.20, 100.00, 90.70, 401.00, 116.00, 40.60, 59.90, 50.00, 58.00, 35.30, 17.20, 22.70, 53.70, 53.70,
     51.00, 120.00, 150.00, 72.00, 429.00, 97.00, 82.00, 66.60, 24.00, 36.00, 18.00, 13.50, 23.00, 57.50, 174.00, 48.00,
     88.00, 150.00, 72.00, 317.00, 8.30, 46.00, 35.00, 8.00])

enthalpy_of_mixing_arr = np.array(
    [[0, -5, -6, -61, -145, -4, 0, -4, -30, -45.5, 11, -1, 12, 34, 37, 35, 19, 26, 8, 1, -5, -7, -9, -34.5, -29, 13, 0,
      8, 27, 51, 49, 8, 5, -14, -40, -16, -13, -12, -18, -28, 16, 0, 6, 30, 48, 50, 29, 11, -9, -33, -37, -19, -15, -21,
      -23],
     [0, 0, 0, -15, -39, 18, -3, 0, -15, -3.5, 27, -14, -36, -30, -16, -7, -10, -4, -4, -4, 0, 4, 5, -3.5, 7, 28, -10,
      -32, -43, -25, -7, -3, -3, -6, -8, 6, 11, 16, 15, 18, 29, -10, -29, -37, -24, -3, 0, -2, -5, -10, 0, 15, 23, 25,
      26],
     [0, 0, 0, -10, -28, 18, -4, 0, -14, 0.5, 27, -22, -55, -58, -42, -31, -32, -26, -24, -24, 0, 4, 6, -0.5, 10, 28,
      -18, -50, -71, -54, -34, -25, -24, -25, -24, 5, 13, 18, 18, 23, 29, -19, -47, -66, -54, -31, -25, -24, -26, -28,
      -2, 19, 27, 30, 31],
     [0, 0, 0, 0, -2, -45, -55, -36, -39, -4.5, -43, -89, -118, -109, -82, -61, -66, -50, -42, -39, -33, -32, -33,
      -29.5,
      -14, -44, -87, -117, -131, -102, -67, -39, -35, -35, -32, -32, -27, -27, -23, -13, -43, -90, -116, -123, -101,
      -60,
      -42, -35, -32, -30, -20, -20, -19, -13, -12],
     [0, 0, 0, 0, 0, -141, -134, -92, -81, -24.5, -152, -201, -224, -190, -143, -107, -119, -87, -75, -69, -84, -88,
      -95,
      -78.5, -59, -154, -206, -232, -233, -174, -115, -68, -61, -63, -62, -94, -91, -98, -90, -74, -155, -212, -235,
      -218, -173, -103, -72, -60, -54, -52, -58, -81, -91, -82, -80],
     [0, 0, 0, 0, 0, 0, 10, 13, -11, -26.5, 1, 1, 34, 68, 73, 71, 49, 62, 41, 32, 16, 6, 5, -21.5, -14, 2, -2, 28, 59,
      93, 93, 47, 44, 19, -15, 0, -3, -5, -8, -20, 3, -3, 24, 63, 89, 97, 73, 52, 28, -1, -14, -11, -11, -18, -20],
     [0, 0, 0, 0, 0, 0, 0, -2, -26, -39.5, 20, -6, -3, 16, 23, 24, 10, 18, 3, -4, -3, -4, -4, -26.5, -21, 23, -4, -6, 6,
      32, 36, 3, 0, -17, -40, -10, -6, -4, -9, -16, 25, -4, -7, 10, 30, 38, 21, 5, -13, -35, -32, -10, -3, -8, -10],
     [0, 0, 0, 0, 0, 0, 0, 0, -19, -20.5, 23, -20, -38, -30, -16, -10, -19, -11, -19, -22, -1, 1, 1, -14.5, -6, 25, -18,
      -38, -44, -18, -5, -20, -21, -32, -46, -4, 3, 7, 4, 2, 26, -20, -38, -39, -19, -2, -9, -18, -30, -44, -22, 4, 11,
      10, 10],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, -25.5, -4, -51, -74, -66, -48, -37, -45, -35, -38, -40, -19, -18, -17, -14.5, -17, -4,
      -49, -73, -84, -56, -35, -38, -38, -46, -55, -20, -13, -10, -11, -8, -3, -52, -73, -77, -56, -31, -31, -36, -43,
      -53, -30, -10, -4, -2, -2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -24.5, -81.5, -112.5, -100.5, -70.5, -49.5, -57.5, -39.5, -35.5, -34.5, -17.5,
      -17.5,
      -18.5, -17, -2.5, -24.5, -81.5, -113.5, -127.5, -89.5, -53.5, -33.5, -30.5, -34.5, -36.5, -18.5, -11.5, -10.5,
      -7.5, 2.5, -24.5, -85.5, -112.5, -117.5, -89.5, -46.5, -32.5, -29.5, -30.5, -34.5, -13.5, -4.5, -1.5, 4.5, 5.5],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 58, 94, 96, 91, 66, 81, 55, 45, 25, 13, 12, -19.5, -11, 0, 7, 50, 88, 123,
      120, 65, 60, 31, -9, 7, 1, -4, -7, -22, 0, 6, 46, 92, 119, 124, 95, 70, 42, 9, -9, -10, -13, -21, -24],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 43, 44, 38, 19, 25, 2, -7, -13, -22, -28, -59.5, -61, 15, 1, 11, 37, 63,
      56, 1, -4, -28, -63, -28, -32, -35, -45, -62, 19, 1, 8, 39, 60, 57, 28, 4, -23, -55, -60, -43, -40, -52, -56],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 1, -8, -11, -30, -39, -24, -29, -38, -69.5, -77, 64, 25, 1, 4, 18,
      11,
      -39, -44, -61, -86, -28, -30, -30, -45, -61, 70, 26, 2, 5, 16, 9, -17, -39, -62, -89, -74, -37, -26, -40, -46],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -8, -17, -28, -35, -9, -15, -23, -51.5, -60, 100, 53, 15, 0, 2,
      -4, -39, -43, -52, -65, -2, -8, -5, -21, -33, 104, 57, 20, 0, 1, -6, -25, -41, -57, -74, -47, -10, 2, -8, -14],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -14, -18, 5, -2, -8, -31.5, -35, 100, 54, 17, -4, -1, 0,
      -21, -25, -29, -35, 17, 9, 12, -1, -8, 103, 57, 22, 2, -1, -1, -13, -23, -34, -45, -19, 10, 22, 15, 10],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1, -4, -7, 12, 5, -1, -18.5, -19, 94, 47, 11, -12, -7, 0, -9,
      -12, -13, -15, 27, 17, 20, 10, 7, 97, 50, 17, -9, -7, 1, -4, -11, -18, -24, 0, 21, 31, 28, 24],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -8, 4, -6, -13, -31.5, -31, 69, 27, -1, -15, -4, 5, -8,
      -11, -16, -23, 13, 2, 3, -7, -11, 71, 29, 3, -12, -4, 6, -1, -9, -18, -28, -11, 4, 11, 7, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 13, 4, -2, -15.5, -14, 83, 34, -1, -25, -16, -2, -3,
      -5, -5, -4, 28, 17, 19, 11, 10, 85, 37, 5, -21, -15, 0, 0, -4, -9, -13, 8, 22, 31, 29, 26],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -5, -11, -21.5, -18, 57, 10, -22, -41, -25, -5, 0,
      -1, -2, -1, 19, 6, 7, 0, 2, 58, 11, -17, -35, -24, -1, 2, 0, -3, -7, 7, 12, 18, 17, 14],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -9, -15, -23.5, -19, 47, -1, -31, -49, -30, -7, 1,
      0, -1, 0, 15, 2, 2, 4, -1, 48, 0, -27, -42, -29, -3, 2, 1, -2, -5, 7, 8, 13, 13, 10],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -11.5, -3, 27, -9, -22, -23, 3, 19, 8, 7, -2,
      -14, 2, 6, 10, 7, 7, 28, -9, -21, -17, 2, 22, 18, 10, 0, -12, -9, 8, 15, 15, 15],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15.5, -6, 14, -21, -31, -29, -1, 12, -4, -5,
      -17, -33, -4, 1, 3, 1, -1, 15, -23, -31, -24, -3, 15, 8, -1, -13, -29, -16, 1, 6, 5, 4],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15.5, -6, 13, -27, -40, -40, -8, 7, -10,
      -11,
      -25, -42, -5, 1, 3, 1, -1, 14, -30, -41, -34, -10, 11, 3, 7, -21, -38, -19, 1, 6, 5, 4],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -12.5, -19.5, -59.5, -72.5, -72.5, -36.5,
      -13.5, -18.5, -18.5, -29.5, -43.5, -17.5, -14.5, -13.5, -12.5, -10.5, -19.5, -63.5, -73.5, -65.5, -37.5, -7.5,
      -7.5, -14.5, -24.5, -37.5, -21.5, -11.5, -9.5, -7.5, -7.5],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -12, -61, -80, -84, -44, -16, -15, -14,
      -24, -36, -8, -4, -3, -1, 3, -12, -66, -81, -75, -45, -9, 6, -11, -19, -31, -11, 0, 3, 6, 7],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 56, 95, 130, 125, 67, 62, 33,
      -9,
      7, 1, -4, -7, -24, 0, 9, 52, 98, 125, 129, 96, 72, 44, 9, -10, -11, -14, -23, -26],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 48, 76, 69, 10, 5, -22, -61,
      -27, -33, -37, -46, -66, 14, 0, 14, 50, 73, 70, 39, 13, -16, -50, -59, -45, -44, -56, -61],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 30, 24, -29, -34, -54, -84,
      -29, -35, -36, -51, -68, 62, 20, 0, 11, 27, 24, -4, -28, -53, -83, -74, -43, -35, -48, -54],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -6, -53, -59, -72, -91,
      -20, -26, -25, -43, -60, 101, 52, 13, 0, 3, -9, -35, -55, -76, -100, -74, -31, -19, -33, -40],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, -36, -41, -46, -53,
      16, 11, 15, -1, -11, 135, 81, 36, 4, 0, -8, -26, -39, -53, -67, -32, 11, 26, 17, 12],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -11, -14, -15, -15,
      37, 28, 33, 20, 17, 128, 73, 31, -4, -5, 0, 7, -14, -21, -28, 3, 32, 45, 42, 38],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 24, 10,
      11, 5, 8, 69, 11, -23, -47, -35, -7, 0, 0, -2, -3, 14, 18, 25, 26, 23],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 23, 9,
      10,
      4, 9, 64, 6, -28, -52, -39, -10, -1, 0, -1, -1, 15, 18, 21, 25, 23],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 10, -6,
      -8, -13, -8, 34, -21, -50, -63, -45, -9, 1, 2, 1, -2, 7, 2, 5, 6, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7, -26,
      -31, -34, -28, -9, -62, -82, -90, -52, -6, 6, 8, 6, 2, 0, -18, -21, -18, -21],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2,
      -2,
      -3, -4, 8, -28, -30, -13, 15, 43, 38, 28, 16, -1, -6, -1, 3, 3, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, -2, 1, -36, -36, -19, 9, 33, 25, 14, 0, -18, -11, 0, 2, 2, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, -4, -4, -42, -39, -18, 13, 38, 29, 16, 0, -21, -11, -1, 0, -1, -1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, -1, -7, -51, -53, -35, -3, 27, 20, 9, -5, -25, -10, 0, 2, 2, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, -25, -72, -71, -50, -13, 25, 23, 14, 1, -17, -4, -1, -1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 12, 57, 104, 130, 132, 101, 74, 45, 10, -9, -11, -15, -25, -27],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 16, 54, 77, 74, 42, 15, -14, -50, -60, -49, -49, -62, -68],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 15, 33, 32, 3, -21, -48, -80, -73, -45, -38, -51, -58],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 3, -6, -30, -48, -68, -90, -63, -23, -11, -23, -30],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, -7, -24, 38, -52, -66, -32, 9, 24, 15, 9],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, -4, -10, -16, -20, 12, 38, 52, 49, 45],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -3, -4, 20, 33, 44, 44, 40],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 18, 23, 30, 32, 29],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 9, 14, 16, 14],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9, -8, -5, -8],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

element_data = pd.DataFrame(
        np.array([atomic_radius_arr, melting_point_arr, boiling_point_arr,
                  pauling_electronegativity_arr, allen_electronegativity_arr, valance_electron_concentration_arr,
                  itinerant_electrons_arr, atomic_weight_arr, density_arr, molar_heat_capacity_arr,
                  thermal_conductivity_arr]).transpose(), columns=columns_names, index=elements_arr)

if __name__ == '__main__':
    element_data = pd.DataFrame(
        np.array([elements_arr, atomic_number_arr, atomic_radius_arr, melting_point_arr, boiling_point_arr,
                  pauling_electronegativity_arr, allen_electronegativity_arr, valance_electron_concentration_arr,
                  itinerant_electrons_arr, atomic_weight_arr, density_arr, molar_heat_capacity_arr,
                  thermal_conductivity_arr]).transpose(), columns=columns_names)


