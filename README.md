# Photovoltaic Performance Prediction with CatBoost

This repository contains a dataset, a trained CatBoost model, and a Jupyter Notebook for analyzing photovoltaic performance using SCAPS-1D Monte-Carlo simulation results. The project aims to predict key performance parameters such as open-circuit voltage (Voc), short-circuit current density (Jsc), fill factor (FF), and power conversion efficiency (PCE) for various configurations of perovskite solar cells.

## Dataset Overview

The dataset includes three main Excel files:

### 1. **cell_simulation_results.xlsx**
- Contains SCAPS-1D Monte-Carlo simulation results with **7182 rows and 14 columns**.
- Includes current-voltage characteristics (I-V curves) and key performance metrics:
  - Open-circuit voltage (Voc)
  - Short-circuit current density (Jsc)
  - Fill factor (FF)
  - Power conversion efficiency (PCE)
- Results are provided for various combinations of:
  - Thicknesses
  - Charge transport layer materials (ETL and HTL)
  - Ionic compositions

### 2. **Material_database.xlsx**
- Contains material properties for different electron transport layers (ETL) and hole transport layers (HTL).
- Parameters include:
  - Type
  - Reference thickness (in μm)
  - Band gap (Eg, in eV)
  - Electron affinity (χ, in eV)
  - Dielectric permittivity
  - Conduction band effective density of states
  - Valence band effective density of states
  - Electron mobility
  - Hole mobility
  - Shallow donor density (ND)
  - Shallow acceptor density (NA)
  - Defect density
  - Electron thermal velocity
  - Hole thermal velocity

### 3. **Material_database_IV_PCE.xlsx**
- Contains the same parameters as in `Material_database.xlsx`, but only for various ion compositions of perovskites with calculated current-voltage characteristics.

## Trained Model

- The repository includes a trained **CatBoost model** (`final_model_catboost.json`) that predicts photovoltaic performance based on the provided dataset.
- The model is ready to use and can be loaded for predictions or further analysis.

## Workflow and Analysis

The Jupyter Notebook file `ml_shap_analysis.ipynb` provides a complete workflow for:
- **Data preprocessing**: Cleaning and preparing the dataset for machine learning.
- **Machine learning modeling**: Training and evaluating the CatBoost model.
- **SHAP analysis**: Explaining model predictions using SHAP values.
- **Visualization**: Generating plots to interpret results and identify optimal parameters for high PCE.

This notebook serves as a **ready-to-use environment** for researchers, enabling them to:
- Reproduce the results.
- Test new configurations.
- Find and visualize optimal parameters for high PCE.
- Further optimize the predictive model.

## Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install catboost pandas numpy shap matplotlib seaborn scikit-learn openpyxl
```
### Using the Model

1. Clone the repository:
```bash
git clone https://github.com/ivan-e-novoselov/perovskites_SCAPS_data.git
cd perovskites_SCAPS_data
```

2. Load the model:
```python
from catboost import CatBoostRegressor

# Load the trained model
model = CatBoostRegressor()
model.load_model('final_model_catboost.json')
```
3. Prepare your input data:
```python
import pandas as pd

# Example input data (replace with your own)
data = {
    'Pero th, nm': [400],
    'ETL': [1],
    'Cs': [0.1],
    'MA': [0.15],
    'FA': [0.75],
    'I': [3],
    'Br': [0], 
    'HTL': [1],
    'FTO': [0],
    'nip': [1],
    'Voc': [1.075],
    'Jsc': [25.82],
    'FF': [82.01]
    }
df = pd.DataFrame(data)

# Predict PCE
predictions = model.predict(df)
print("Predicted PCE:", predictions)
```

## Running the Notebook
Open [ml_shap_analysis.ipynb](https://github.com/ivan-e-novoselov/perovskites_SCAPS_data/blob/main/ml_shap_analysis.ipynb) in Jupyter Notebook or JupyterLab.
Follow the steps in the notebook to preprocess data, train the model, and analyze results using SHAP.

## License
This project is licensed under the MIT License . See the [LICENSE](https://github.com/ivan-e-novoselov/perovskites_SCAPS_data/blob/main/LICENSE) file for more details.
