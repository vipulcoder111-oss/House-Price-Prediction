# 🏠 House Price Prediction

A simple Machine Learning project that predicts house prices based on important property features such as living area, number of bedrooms, number of bathrooms, and year built.

This project uses Python, Pandas, and Scikit-learn to build and evaluate a Linear Regression model.

---

## 📌 Project Overview

The goal of this project is to develop a basic machine learning model that can estimate the selling price of a house from selected property features.

The project uses the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

The model is trained using historical house data and then used to predict the price of a new house.

---

## 🚀 Features

- Load and process house price data using Pandas
- Select relevant features for prediction
- Handle missing numerical values
- Split the dataset into training and testing data
- Train a Linear Regression model
- Evaluate model performance using:
  - Mean Absolute Error (MAE)
  - R² Score
- Predict the price of a new house

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Linear Regression**
- **CSV Dataset**

---

## 📊 Dataset

The project uses the Kaggle **House Prices - Advanced Regression Techniques** dataset.

### Selected Features

| Feature | Description |
|---|---|
| `GrLivArea` | Above-ground living area in square feet |
| `BedroomAbvGr` | Number of bedrooms above ground |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Year the house was originally built |

### Target Variable

```text
SalePrice
