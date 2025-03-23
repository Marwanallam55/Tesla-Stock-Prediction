import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import datetime

# File paths
MODEL_PATH = "trading_model.pkl"  
STARTING_CAPITAL = 10000  # Starting capital in USD
TRANSACTION_FEE = 0.01  # 1% transaction fee

# Load the existing Tesla stock data
FILE_PATH = "TSLA.csv"  
data = pd.read_csv(FILE_PATH)

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing 'Close' values and handle them
if 'Close' not in data.columns:
    raise ValueError("The 'Close' column is missing from the dataset.")

# Calculate moving averages and daily returns
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['Daily_Return'] = data['Close'].pct_change()

# Drop rows with missing values
data.dropna(inplace=True)

# Shift the target variable (Close) to the next day's closing price
data['Next_Close'] = data['Close'].shift(-1)

# Drop rows where 'Next_Close' is NaN (because the last row has no next day)
data.dropna(inplace=True)

# Define features and target for prediction
features = ['SMA_5', 'SMA_10', 'Daily_Return']
target = 'Next_Close'  # Predict the next day's closing price

X = data[features]
y = data[target]

# Train-test split (Using the entire dataset for training)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model performance (on the training/validation dataset)
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print("Validation Mean Absolute Error (MAE):", mae)
print("Validation Mean Squared Error (MSE):", mse)

# Save the trained model for later use
joblib.dump(model, MODEL_PATH)
print(f"Model saved successfully at {MODEL_PATH}")

# --- Simulate trading during the simulation period --- 
# Initial capital
capital = STARTING_CAPITAL
shares_held = 0  # Number of Tesla shares held
account_balance = capital  # Account balance

# Simulate daily trading decisions (Buy, Sell, Hold)
simulation_dates = pd.to_datetime(['2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28'])

for current_date in simulation_dates:
    # Use the most recent data available to make predictions
    if current_date > data['Date'].max():
        print(f"Simulation date {current_date} exceeds available data range. Skipping this day.")
        continue

    # Get the most recent day's data from the dataset
    current_day = data[data['Date'] == current_date].iloc[0]

    current_price = current_day['Close']
    prediction = model.predict([current_day[features]])[0]

    # Trading decision based on predicted next day's price
    if prediction > current_price:
        # Buy signal: Predicting higher price, decide to buy
        if account_balance >= current_price:
            shares_to_buy = int(account_balance // current_price)
            cost = shares_to_buy * current_price * (1 + TRANSACTION_FEE)  # Include transaction fee
            if cost <= account_balance:
                account_balance -= cost
                shares_held += shares_to_buy
                print(f"Bought {shares_to_buy} shares at {current_price} on {current_date.strftime('%Y-%m-%d')}")
    elif prediction < current_price and shares_held > 0:
        # Sell signal: Predicting lower price, decide to sell
        proceeds = shares_held * current_price * (1 - TRANSACTION_FEE)  # Include transaction fee
        account_balance += proceeds
        print(f"Sold {shares_held} shares at {current_price} on {current_date.strftime('%Y-%m-%d')}")
        shares_held = 0

    # Hold: No action taken, continue to next day
    print(f"Holding at {current_price} on {current_date.strftime('%Y-%m-%d')}")

# Final account balance calculation
final_balance = account_balance + (shares_held * data.iloc[-1]['Close'])  # Add value of held shares
performance = (final_balance - STARTING_CAPITAL) / STARTING_CAPITAL * 100
print(f"Final Account Balance: ${final_balance}")
print(f"Performance over the period: {performance:.2f}%")

# --- Save the results to a CSV file --- 
results = {
    "Initial Capital": STARTING_CAPITAL,
    "Final Account Balance": final_balance,
    "Performance (%)": performance
}

results_df = pd.DataFrame([results])
results_df.to_csv("trading_results.csv", index=False)
print("Results saved to trading_results.csv")
