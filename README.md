# University Ranking Prediction

This project involves predicting university rankings based on various features using machine learning. The dataset includes multiple metrics related to academic reputation, employer reputation, and more. The primary objective is to build a predictive model that accurately forecasts university rankings.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Model Development](#model-development)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)

## Project Overview

This project aims to predict university rankings using a Random Forest Regressor. The dataset provides information about various universities, including their academic reputation, employer reputation, and other relevant metrics. The model has achieved an impressive accuracy of 87%.

## Data Description

The dataset consists of the following columns:
- `2025 Rank`: Ranking for the year 2025 (object)
- `2024 Rank`: Ranking for the year 2024 (object)
- `Institution Name`: Name of the institution (object)
- `Location`: Location of the institution (object)
- `Location Full`: Detailed location (object)
- `Size`: Size of the institution (object)
- `Academic Reputation`: Academic reputation score (float64)
- `Employer Reputation`: Employer reputation score (float64)
- `Faculty Student Ratio`: Ratio of faculty to students (float64)
- `Citations per Faculty`: Citations per faculty member (float64)
- `International Faculty`: Percentage of international faculty (float64)
- `International Students`: Percentage of international students (float64)
- `International Research Network`: Research network score (float64)
- `Employment Outcomes`: Employment outcomes score (float64)
- `Sustainability`: Sustainability score (float64)
- `QS Overall Score`: Overall QS score (object)

## Model Development

### Data Preprocessing

- Handling missing values
- Converting categorical columns to numerical values
- Scaling and normalizing data

### Model Training

- Model: Random Forest Regressor
- Performance Metric: R² Score, Mean Squared Error

### Hyperparameter Optimization

Performed using Grid Search to find the best parameters:
- `bootstrap`: True
- `max_depth`: 43
- `min_samples_leaf`: 1
- `min_samples_split`: 4
- `n_estimators`: 360

The best model achieved an R² Score of 87% and a Mean Squared Error of 16.83.

## Results

- **Mean Squared Error (MSE)**: 16.83
- **R² Score**: 87%

The model performs well with an accuracy of 87%, indicating its effectiveness in predicting university rankings.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/university-ranking-prediction.git
    cd university-ranking-prediction
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the project:**

    ```bash
    python main.py
    ```

## Usage

1. **Data Loading**: Load your dataset into the script by updating the `data_path` variable in `main.py`.
2. **Model Training**: Run `main.py` to train the model and evaluate its performance.
3. **Prediction**: Use the trained model to make predictions on new data.

