# Aircraft Engine Remaining Useful Life Prediction

A predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the NASA **C-MAPSS FD001–FD004** datasets.

The project applies machine learning and deep learning models to predict the remaining operating cycles of aircraft engines and support maintenance planning through an interactive web dashboard.

---

## Project Overview

Aircraft engines generate large amounts of operational sensor data. Predicting when an engine may require maintenance can help reduce unexpected failures, improve safety, and optimize maintenance schedules.

This project processes NASA C-MAPSS turbofan engine degradation data and compares multiple prediction models:

- Bidirectional Long Short-Term Memory (**BiLSTM**)
- Gated Recurrent Unit with Attention (**GRU + Attention**)
- Extreme Gradient Boosting (**XGBoost**)
- Ensemble prediction

The system also includes a web dashboard that displays engine health, predicted RUL, maintenance priorities, recommended actions, and maintenance schedules.

---

## Main Features

- Supports NASA C-MAPSS subsets **FD001, FD002, FD003, and FD004**
- Calculates Remaining Useful Life labels
- Applies feature scaling and time-series sequence generation
- Uses engine-based data splitting
- Uses Group K-Fold cross-validation
- Trains BiLSTM models
- Trains GRU models with an attention mechanism
- Trains XGBoost models
- Generates ensemble predictions
- Evaluates models using MAE and RMSE
- Predicts RUL for individual engines
- Classifies engines by maintenance priority
- Generates maintenance schedules
- Provides an interactive web dashboard
- Includes an AI assistant powered by OpenRouter

---

## Models

### BiLSTM

The Bidirectional Long Short-Term Memory model processes sequential engine sensor data in both forward and backward directions.

### GRU with Attention

The GRU model uses an attention mechanism to identify the most relevant time steps when predicting engine RUL.

### XGBoost

XGBoost is used as a machine learning model for predicting RUL from engineered sequence features.

### Ensemble

The ensemble prediction combines the predictions from multiple models to improve prediction stability.

---

## Dataset

This project uses the NASA C-MAPSS turbofan engine degradation datasets:

- FD001
- FD002
- FD003
- FD004

The raw dataset files are not included in this repository because of file-size limitations.

Place the following files inside the `datasets/` folder before running the full training pipeline:

```text
datasets/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
````

---

## Project Structure

```text
aircraft-engine-rul-prediction/
├── datasets/
│   └── README.md
├── outputs/
├── rul_dashboard_web/
│   ├── public/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   ├── logo.png
│   ├── server.js
│   ├── package.json
│   ├── package-lock.json
│   └── start.cmd
├── RUL_Prediction_Framework_FD001_FD004.ipynb
├── aircraft_engine_rul_project_report.pdf
├── rul_prediction.py
├── generate_dashboard_data.py
├── combine_final_results.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Python Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

The project uses the following main libraries:

* NumPy
* Pandas
* Scikit-learn
* PyTorch
* XGBoost
* Matplotlib

---

## Run the Prediction Pipeline

To run the complete RUL prediction pipeline:

```bash
python rul_prediction.py
```

The pipeline performs the following steps:

1. Loads the C-MAPSS datasets
2. Calculates RUL labels
3. Preprocesses and scales sensor features
4. Creates time-series sequences
5. Trains the prediction models
6. Evaluates model performance
7. Saves model results to the `outputs/` folder

The full pipeline may take a long time to complete, especially when running deep learning models on a CPU.

---

## Generate Dashboard Data

After generating the model results, run:

```bash
python generate_dashboard_data.py
```

This script generates per-engine prediction files and maintenance planning outputs used by the web dashboard.

Generated files may include:

```text
outputs/
├── engine_predictions_FD001.csv
├── engine_predictions_FD002.csv
├── engine_predictions_FD003.csv
├── engine_predictions_FD004.csv
├── maintenance_priority_FD001.csv
├── maintenance_priority_FD002.csv
├── maintenance_priority_FD003.csv
├── maintenance_priority_FD004.csv
├── maintenance_schedule_FD001.csv
├── maintenance_schedule_FD002.csv
├── maintenance_schedule_FD003.csv
└── maintenance_schedule_FD004.csv
```

---

## Combine Final Results

To combine generated model result files:

```bash
python combine_final_results.py
```

This script combines available evaluation results into final summary files used for comparison and dashboard visualization.

---

## Run the Web Dashboard

Open a terminal inside the dashboard folder:

```bash
cd rul_dashboard_web
```

Install the Node.js dependencies:

```bash
npm install
```

Start the dashboard server:

```bash
npm start
```

The dashboard will run at:

```text
http://localhost:3000
```

On Windows, the dashboard may also be started using:

```text
start.cmd
```

---

## Environment Variables

The AI assistant uses an OpenRouter API key.

Create a file named `.env` inside the `rul_dashboard_web/` folder:

```env
OPENROUTER=your_openrouter_api_key_here
PORT=3000
```

Do not upload the `.env` file to GitHub.

An optional safe example file can be included as:

```text
.env.example
```

With the following content:

```env
OPENROUTER=your_openrouter_api_key_here
PORT=3000
```

---

## Dashboard Features

The dashboard includes:

* Engine health overview
* Predicted cycles remaining
* Actual cycles remaining
* Most critical engines
* Maintenance priority classification
* Recommended maintenance actions
* Maintenance schedule generation
* Fleet health visualizations
* Dataset filtering
* AI maintenance assistant

---

## Maintenance Priority Levels

The dashboard groups engines into the following maintenance categories:

| Priority |       Predicted RUL | Dashboard Label  |
| -------- | ------------------: | ---------------- |
| Critical |   20 cycles or less | Immediate Check  |
| High     |        21–50 cycles | Upcoming Service |
| Medium   |        51–90 cycles | Monitor          |
| Low      | More than 90 cycles | Healthy          |

These classifications are designed as predictive maintenance decision-support indicators.

They are not certified aircraft safety instructions.

---

## Evaluation Metrics

The models are evaluated using:

### Mean Absolute Error

MAE measures the average absolute difference between the predicted RUL and the real RUL.

Lower MAE values indicate better prediction performance.

### Root Mean Squared Error

RMSE gives greater weight to large prediction errors.

Lower RMSE values indicate better model performance.

---

## Validation Strategy

The project uses engine-based data splitting to prevent time-series samples from the same engine from appearing in both training and testing data.

It also uses Group K-Fold cross-validation, where the engine identifier is treated as the grouping variable.

This approach provides a more reliable evaluation of model generalization across different engines.

---

## Outputs

The project generates:

* Model evaluation results
* MAE and RMSE comparisons
* Group K-Fold summaries
* Best-model summaries
* Per-engine RUL predictions
* Maintenance priorities
* Recommended maintenance actions
* Maintenance schedules
* Dashboard-ready CSV files

---

## Jupyter Notebook

The complete notebook version of the prediction framework is available in:

```text
RUL_Prediction_Framework_FD001_FD004.ipynb
```

The notebook contains the project workflow, model training process, evaluation, and result comparison.

---

## Project Report

The full graduation project report is available in:

```text
aircraft_engine_rul_project_report.pdf
```

---

## Technologies Used

### Machine Learning and Data Science

* Python
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* XGBoost
* Matplotlib

### Web Dashboard

* Node.js
* Express.js
* HTML
* CSS
* JavaScript
* Chart.js

### AI Assistant

* OpenRouter API

---

## Security Notes

* Do not upload the `.env` file
* Do not publish API keys
* Do not upload `node_modules/`
* Do not upload server log files
* Do not upload trained model files unless required
* Revoke and replace any API key that was accidentally committed

---

## Disclaimer

This project was developed for academic and research purposes.

The maintenance recommendations produced by the system are predictive decision-support outputs and should not be treated as certified aviation maintenance instructions.

---

## Project Team

- Abdulaziz Alsaidalani
- Omar Daffa
- Ahmed Babader
- Firas Matari
- Bandar Farhan

## Supervisor

- Dr. Tariq Alsahfi
