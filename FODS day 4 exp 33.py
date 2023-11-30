import pandas as pd
# Sample stock price data
data = {
'Date': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
'ClosingPrice': [100, 105, 102, 98, 110]
}
stock_data = pd.DataFrame(data)
stock_data['PriceChange'] = stock_data['ClosingPrice'].diff()
mean_change = stock_data['PriceChange'].mean()
std = stock_data['PriceChange'].std()
print("Stock Price Variability Analysis")
print(f"Mean Daily Price Change: {mean_change:.2f}")
print(f"Standard Deviation of Daily Price Changes: {std:.2f}")
positive_changes = stock_data[stock_data['PriceChange'] > 0]['PriceChange'].count()
negative_changes = stock_data[stock_data['PriceChange'] < 0]['PriceChange'].count()
print("\nStock Movement Direction")
print(f"Days with Positive Price Change: {positive_changes}")
print(f"Days with Negative Price Change: {negative_changes}")

