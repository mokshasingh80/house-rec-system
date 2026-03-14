import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("housingdata.csv")

# Features and target
X = data[["area", "bedrooms", "age"]]
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("House Price Prediction System")

# User input
area = int(input("Enter house area: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter house age: "))

# Prediction
prediction = model.predict([[area, bedrooms, age]])

print("Estimated house price:", int(prediction[0]))

# Visualization
plt.scatter(data["area"], data["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price vs Area")
plt.show()
