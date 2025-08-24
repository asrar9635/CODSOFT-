# Sales Prediction using Machine Learning
# Author: Asrar Ahmed (BCA, Data Science)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("advertising.csv")   # <-- make sure your CSV file is in the same folder
print("First 5 rows of dataset:\n", df.head(), "\n")

# Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}\n")

# Show some predictions vs actual
results = pd.DataFrame({"Actual Sales": y_test.values, "Predicted Sales": y_pred})
print("🔎 Sample Predictions:\n", results.head())

# Plot Actual vs Predicted Sales
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", s=60)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Plot feature impact on sales
plt.figure(figsize=(10,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap (Advertising vs Sales)")
plt.show()
