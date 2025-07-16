import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Load the Excel sheet
df = pd.read_excel('Lab Session Data.xlsx', sheet_name="IRCTC Stock Price", engine='openpyxl')

# Convert 'Date' to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])

# Extract required columns
price_col = df.columns[3]   
chg_col = df.columns[8]     

prices = df[price_col].dropna().tolist()

# Mean and Variance of Price Data
mean_price = statistics.mean(prices)
var_price = statistics.variance(prices)
print(f"Mean Price: {mean_price:.2f}")
print(f"Variance of Price: {var_price:.2f}")

# Sample Mean for Wednesdays
df['Weekday'] = df['Date'].dt.day_name()
wednesday_prices = df[df['Weekday'] == 'Wednesday'][price_col].dropna().tolist()
mean_wed = statistics.mean(wednesday_prices)
print(f"\nMean Price on Wednesdays: {mean_wed:.2f}")
print(f"Difference from population mean: {abs(mean_price - mean_wed):.2f}")

# Sample Mean for April
df['Month'] = df['Date'].dt.month_name()
april_prices = df[df['Month'] == 'April'][price_col].dropna().tolist()
mean_april = statistics.mean(april_prices)
print(f"\nMean Price in April: {mean_april:.2f}")
print(f"Difference from population mean: {abs(mean_price - mean_april):.2f}")

# Probability of Making a Loss (Chg% < 0)
chg_values = df[chg_col].dropna()
loss_days = list(filter(lambda x: x < 0, chg_values))
prob_loss = len(loss_days) / len(chg_values)
print(f"\nProbability of making a loss: {prob_loss:.2f}")

# Probability of Profit on Wednesday (Chg% > 0 on Wednesday)
wednesday_df = df[df['Weekday'] == 'Wednesday']
wednesday_profits = wednesday_df[wednesday_df[chg_col] > 0]
prob_profit_wed = len(wednesday_profits) / len(wednesday_df)
print(f"Probability of making profit on Wednesday: {prob_profit_wed:.2f}")

# Conditional Probability: P(Profit | Wednesday)
print(f"Conditional probability of profit given it's Wednesday: {prob_profit_wed:.2f}")

# Scatter plot of Chg% vs Day of Week
plt.figure(figsize=(10, 5))
plt.scatter(df['Weekday'], df[chg_col], alpha=0.6, color='green')
plt.title("Chg% vs Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Change %")
plt.grid(True)
plt.show()
