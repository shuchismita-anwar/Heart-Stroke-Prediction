# Stroke Prediction

This repository contains a stroke prediction model implemented in Python using various machine learning algorithms. The model is trained on a dataset containing health-related features to predict the likelihood of a stroke. The steps include data preprocessing, handling missing values, oversampling the minority class, model training, and evaluation.

## Dataset
The dataset used for this project includes information such as age, gender, hypertension, heart disease, average glucose level, BMI, smoking status, and other factors. The goal is to predict whether an individual is likely to have a stroke or not. Link: https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset

## Data Preprocessing
1. **Loading Data:** The dataset is loaded into a Pandas DataFrame for analysis.
2. **Handling Missing Values:** Null values in the 'bmi' and 'smoking_status' columns are handled through imputation and dropping rows.
3. **Encoding:** Categorical variables are encoded using one-hot encoding.
4. **Scaling:** Continuous features are standardized using `StandardScaler`.

## Oversampling
Due to the imbalanced nature of the dataset regarding the target variable ('stroke'), oversampling is performed using the SMOTENC algorithm to balance the classes.

## Model Training
Several machine learning algorithms are employed for training the stroke prediction model:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine (Linear Kernel)
- Support Vector Machine (RBF Kernel)
- Neural Network (MLPClassifier)
- Random Forest

## Evaluation
Model performance is evaluated using accuracy, precision, recall, and F1-score. Confusion matrices are also visualized to understand the model's predictions.

## Results
The accuracy and error rates of each model are visualized using bar charts, providing a comparative analysis of their performance.

## How to Use
1. Ensure you have the required Python libraries installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `imbalanced-learn`, `scikit-learn`).
2. Run the provided Jupyter Notebook script (`PROJECT.py`) in your preferred environment.
3. The script will load the dataset, preprocess the data, oversample, train the models, and display the evaluation metrics and visualizations.
