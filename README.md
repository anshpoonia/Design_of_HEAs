## Design of High Entropy Alloys for high hardness using a metaheuristic algorithm with phase prediction

*Ansh Poonia, Modalavalasa Kishor, A.K.Prasada Rao*


This repository contains the codebase associated with our research paper focusing on the Design of High-Entropy Alloys (HEAs) through the utilization of a metaheuristic algorithm. <br><br>
The repository contains Python code and relevant files required for:
* HEA Identification
* Multi-Labeled Phase Classification
* Hardness Prediction
* Composition Optimization

#### Repository Tree

```bash
│   .gitignore
│   elemental_database.py
│   hardness_prediction.py
│   hea_identification.py
│   optimization.py
│   phase_prediction.py
│   README.md
│   utils.py
│
├───Hardness_Prediction_Files
│       hardness_prediction_model
│       Standard_Scalar.bin
│
├───HEA_Identification_Files
│       HEA_Identification_Model.bin
│       StandardScalar.bin
│
└────Phase_Classification_Files
   │   One_Hot_Encoder.bin
   │   Standard_Scalar.bin
   │
   └───phase-classification-nn1-model
       │   keras_metadata.pb
       │   saved_model.pb
       │
       ├───assets
       └───variables
               variables.data-00000-of-00001
               variables.index



```

#### Dependencies
``Python 3.9.6``
* ``Numpy 1.22.4``
* ``Pandas 1.5.3``
* ``TensorFlow 2.10.0``
* ``Scikit-Learn 1.2.2``
* ``SciPy 1.9.1``
* ``SymPy 1.11.1``
* ``joblib 1.20``
* ``catboost 1.1.1``