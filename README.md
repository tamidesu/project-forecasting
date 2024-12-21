# Happiness Score Prediction for Kazakhstan (2015-2023)

![Happiness](https://png.pngtree.com/png-vector/20221119/ourmid/pngtree-happy-woman-feel-shocked-and-stunned-png-image_6471556.png)

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
  - [Data Sources](#data-sources)
  - [Data Cleaning](#data-cleaning)
  - [Challenges and Mistakes](#challenges-and-mistakes)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Visualization](#visualization)
  - [Correlation Analysis](#correlation-analysis)
  - [Challenges and Mistakes](#eda-challenges-and-mistakes)
- [Modeling and Analysis](#modeling-and-analysis)
  - [Traditional Time Series Models](#traditional-time-series-models)
    - [ARIMA](#arima)
    - [SARIMA](#sarima)
    - [Prophet](#prophet)
    - [Challenges and Mistakes](#time-series-model-challenges-and-mistakes)
  - [Machine Learning Models](#machine-learning-models)
    - [Random Forest](#random-forest)
    - [XGBoost](#xgboost)
    - [LightGBM](#lightgbm)
    - [Support Vector Regression (SVR)](#support-vector-regression-svr)
    - [Linear Regression](#linear-regression)
    - [Gradient Boosting](#gradient-boosting)
    - [Challenges and Mistakes](#ml-model-challenges-and-mistakes)
  - [Deep Learning Models](#deep-learning-models)
    - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
    - [Challenges and Mistakes](#deep-learning-model-challenges-and-mistakes)
- [Results and Evaluation](#results-and-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Feature Importance](#feature-importance)
  - [Visualization of Predictions](#visualization-of-predictions)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **Happiness Score Prediction for Kazakhstan (2015-2023)** project repository! This project aims to analyze and predict the Happiness Scores of Kazakhstan using various statistical and machine learning models. By leveraging historical data from 2015 to 2023, we explore different modeling approaches, identify key factors influencing happiness, and forecast future trends.

## Repository Structure

```plaintext
├── README.md
├── basisModelsAnalysis.ipynb
├── dataPreparation-2015-2019.ipynb
├── dataPreparation-2015-2023.ipynb
├── hackingLSTM.ipynb
├── 2015.csv
├── 2016.csv
├── 2017.csv
├── 2018.csv
├── 2019.csv
├── 2020.csv
├── 2021.csv
├── 2022.csv
└── 2023.csv
```
- **dataPreparation-2015-2019.ipynb**: Notebook for preparing and cleaning data from 2015 to 2019.
- **dataPreparation-2015-2023.ipynb**: Extended data preparation covering up to 2023.
- **basisModelsAnalysis.ipynb**: Initial analysis using traditional statistical models.
- **hackingLSTM.ipynb**: Advanced modeling using LSTM neural networks.
- **data/**: Directory containing raw CSV files for each year.
- **requirements.txt**: Python dependencies required to run the notebooks.

## **Data Preparation**

### **Data Sources**
The data used in this project is sourced from the [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness) from the Kaggle. Each year's report is available as a separate CSV file from 2015 to 2023. These files contain various indicators that contribute to the overall Happiness Score of countries worldwide.

## **Data Cleaning**

Data preparation involves several steps to ensure the dataset is suitable for analysis and modeling:

1. **Consistent Column Naming**: Renaming columns to maintain consistency across different years, as the naming conventions may vary.

2. **Handling Missing Values**:
   - Filling missing values using appropriate strategies (e.g., forward fill, median imputation).
   - Replacing zero values with median values for certain features to address potential data entry errors.

3. **Feature Selection**: Selecting relevant features that have a significant correlation with the Happiness Score.

4. **Lag Features**: Creating lag features to incorporate historical data into the models.

### Challenges and Mistakes

- **Data Integration Issues:**  
  Merging data across years introduced discrepancies in feature dimensions due to variable changes. Resolving this required balancing dimensionality reduction and feature retention strategies.

- **Overfitting During Feature Engineering:**  
  Initial feature engineering efforts included overly complex lagged features, leading to overfitting. Simplified feature sets and PCA helped address this.

- **Spurious Correlations in EDA:**  
  Misleading relationships between features (e.g., Trust and Generosity) complicated early modeling. Partial correlation analysis was later employed to identify genuine links.

- **Non-Stationarity in Time Series Data:**  
  Many features exhibited non-stationary behavior, violating model assumptions. Differencing and transformations were applied to ensure stationarity before training.

- **Data Leakage:**  
  Improper handling of lagged features in LSTM models led to data leakage. This was resolved by re-structuring data pipelines and feature extraction processes.

- **Scalability Limitations:**  
  Deep learning models like LSTM faced computational bottlenecks. Techniques like reduced batch sizes, pruning, and learning rate adjustments improved scalability.

- **Encoding Challenges:**  
  One-hot encoding of categorical data inflated dimensionality. Switching to target encoding preserved predictive power while reducing computational costs.

- **Outlier Predictions:**  
  Misaligned scaling inversions caused extreme deviations in predictions. Reworking the scaling pipeline resolved these anomalies.

- **Inconsistent Metrics:**  
  Different models used varying evaluation metrics, complicating comparisons. Standardizing RMSE, MAPE, and R² across all models ensured uniform assessments.

---

## **Exploratory Data Analysis (EDA)**

### **Visualization**
EDA was performed to understand the distribution and trends of various features:

- **Time Series Plots**: Visualizing the trend of Happiness Scores and other indicators over the years.
- **Histograms**: Examining the distribution of individual features to identify skewness and outliers.
- **Box Plots**: Detecting outliers and understanding the variability in the data.

## **Correlation Analysis**
A correlation matrix was computed to identify the relationships between different features and the Happiness Score:

- **High Correlation Features**: GDP, Health, Family, and Freedom showed strong positive correlations with Happiness Score.
- **Low or Negative Correlations**: Features like Generosity and Trust had weaker or negative correlations.

---

## **EDA Challenges and Mistakes**

- **Overplotting**: Initial plots were cluttered, making it difficult to interpret trends. This was mitigated by adjusting plot styles and layouts.
- **Misalignment of Data**: Ensuring that all features were aligned correctly across years to avoid misleading correlations.
- **Interpretation Errors**: Misinterpreting correlation coefficients initially led to incorrect feature selection, which was corrected in subsequent iterations.

## **Modeling and Analysis**

This section covers the various models applied to predict the Happiness Score, ranging from traditional time series models to advanced machine learning and deep learning techniques.

---

### **Traditional Time Series Models**

#### **ARIMA**
- **Description**: Autoregressive Integrated Moving Average (ARIMA) model used for univariate time series forecasting.
- **Implementation**: Used the `statsmodels` library to fit the ARIMA model with determined order parameters.
- **Challenges**:
  - Selecting the appropriate order `(p,d,q)` was time-consuming.
  - Residual analysis revealed non-stationarity, requiring differencing or transformation.

#### **SARIMA**
- **Description**: Seasonal ARIMA model to account for seasonality in the data.
- **Implementation**: Extended ARIMA with seasonal components.
- **Challenges**:
  - Computationally intensive due to additional seasonal parameters.
  - Overfitting risk with high seasonal orders.

#### Prophet

- **Description**: Facebook's Prophet model for time series forecasting, known for handling seasonality and holidays.
- **Implementation**: Used `fbprophet` to model and forecast the Happiness Score.
- **Challenges**:
  - Adjusting hyperparameters for optimal performance.
  - Incorporating external regressors (e.g., GDP, Health) required careful feature engineering.

### Time Series Model Challenges and Mistakes

- **Model Selection**: Initially selected inappropriate orders leading to poor forecasts.
- **Overfitting**: Complex models like SARIMA overfitted the training data.
- **Data Leakage**: Mishandling of lag features resulted in data leakage, compromising model integrity.

### Machine Learning Models

#### Random Forest
- **Description**: Ensemble learning method using multiple decision trees for regression.
- **Implementation**: Utilized `scikit-learn`'s `RandomForestRegressor` with hyperparameter tuning.
- **Challenges**:
  - High variance with too many trees or deep trees.
  - Computationally expensive with large datasets.

#### XGBoost
- **Description**: Extreme Gradient Boosting for high-performance gradient boosting.
- **Implementation**: Employed `xgboost` for regression tasks with parameter tuning.
- **Challenges**:
  - Parameter tuning was critical to avoid overfitting.
  - Handling missing values required specific strategies.

#### LightGBM
- **Description**: Gradient Boosting framework that uses tree-based learning algorithms.
- **Implementation**: Applied `lightgbm` with randomized search for hyperparameter optimization.
- **Challenges**:
  - Managing categorical features and ensuring proper encoding.
  - Balancing between model complexity and performance.

#### Support Vector Regression (SVR)
- **Description**: Support Vector Machine for regression tasks.
- **Implementation**: Used `scikit-learn`'s `SVR` with kernel tuning.
- **Challenges**:
  - Sensitive to feature scaling, requiring careful preprocessing.
  - Computationally intensive with large feature sets.

#### Linear Regression
- **Description**: Simple linear model for regression.
- **Implementation**: Applied `scikit-learn`'s `LinearRegression`.
- **Challenges**:
  - Limited to linear relationships, underfitting complex data.
  - Assumes homoscedasticity and normality of residuals.

#### Gradient Boosting
- **Description**: Ensemble method that builds models sequentially to correct errors.
- **Implementation**: Implemented `GradientBoostingRegressor` with hyperparameter adjustments.
- **Challenges**:
  - Prone to overfitting if not properly regularized.
  - Requires careful tuning of learning rates and tree depths.

#### Machine Learning Model Challenges and Mistakes
- **Feature Engineering**: Incorrect creation of lag features led to poor model performance.
- **Scaling Issues**: Models like SVR required meticulous feature scaling, which was initially overlooked.
- **Hyperparameter Tuning**: Ineffective tuning strategies caused suboptimal model performance.

### Deep Learning Models

#### Long Short-Term Memory (LSTM)
- **Description**: Recurrent Neural Network architecture capable of learning long-term dependencies.
- **Implementation**: Utilized TensorFlow and Keras to build and train LSTM models for time series forecasting.
- **Challenges**:
  - Ensuring data was correctly shaped for LSTM input.
  - Managing overfitting through dropout and regularization techniques.
  - Computational resource requirements for training deep models.

#### Deep Learning Model Challenges and Mistakes
- **Data Reshaping**: Initial attempts had mismatched dimensions, leading to runtime errors.
- **Activation Functions**: Choosing appropriate activation functions was critical for model performance.
- **Prediction Scaling**: Incorrect inverse scaling of predictions resulted in unrealistic forecasts.

## Results and Evaluation

### Performance Metrics
Various metrics were used to evaluate model performance:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R² Score**

### Feature Importance
Analyzed feature importance for models like LightGBM, Random Forest, and XGBoost to identify key drivers of Happiness Score:
- **GDP**
- **Health**
- **Family**
- **Freedom**

### Visualization of Predictions
Visual comparisons between actual and predicted Happiness Scores were plotted to assess model accuracy and forecast reliability.

---

## Future Work
- **Data Augmentation**: Incorporate additional years or related features to enhance model robustness.
- **Advanced Deep Learning Models**: Explore architectures like GRU or Transformer-based models for better performance.
- **Hyperparameter Optimization**: Implement more exhaustive tuning strategies, such as Bayesian Optimization.
- **Model Ensemble**: Combine predictions from multiple models to improve accuracy.
- **Deployment**: Develop a web application or API for real-time Happiness Score predictions.

---

## Acknowledgements
- **World Happiness Report**: For providing the foundational data used in this project.
- **Kaggle Community**: For inspiration and shared knowledge on data science projects.

## Contact

For any questions or feedback, please reach out to:


- **Email**: mogleg2@gmail.com
- **GitHub**: [tamidesu](https://github.com/tamidesu)

---

> “Happiness is not something ready-made. It comes from your own actions.” – Dalai Lama
