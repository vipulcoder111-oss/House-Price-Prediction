import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(data))
print()


# 2. Select features
X = data[
    [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "YearBuilt"
    ]
]

# Target variable
y = data["SalePrice"]


# 3. Handle missing values
X = X.fillna(X.mean())


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create Linear Regression model
model = LinearRegression()


# 6. Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")
print()


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("========================================")
print("       HOUSE PRICE PREDICTION")
print("========================================")

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# 9. Predict price for a new house
new_house = [
    [
        2000,   # Living Area
        3,      # Bedrooms
        2,      # Bathrooms
        2015    # Year Built
    ]
]

price = model.predict(new_house)


# 10. Display prediction
print()
print("New House Details")
print("----------------------------------------")
print("Living Area : 2000 sq ft")
print("Bedrooms    : 3")
print("Bathrooms   : 2")
print("Year Built  : 2015")
print("----------------------------------------")

print("Predicted House Price:", round(price[0], 2))

print("========================================")