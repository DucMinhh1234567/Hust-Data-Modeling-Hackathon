import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
print('=== SHAPE ===')
print(df.shape)

print('\n=== DTYPES ===')
print(df.dtypes)

print('\n=== NULLS ===')
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])
print('\n=== DATE RANGE ===')
print('Min:', df['Date'].min())
print('Max:', df['Date'].max())
print('Unique dates:', df['Date'].nunique())

print('\n=== QUANTITY STATS ===')
print(df['Quantity'].describe())
print('Negative (returns):', (df['Quantity'] < 0).sum())
print('Zero quantity rows:', (df['Quantity'] == 0).sum())

print('\n=== SKU COUNT ===')
print('Unique SKUs in train:', df['ItemCode'].nunique())

# Daily sales per SKU
daily = df.groupby(['Date', 'ItemCode'])['Quantity'].sum().reset_index()
print('\n=== DAILY SALES STATS PER SKU ===')
sku_stats = df.groupby('ItemCode')['Quantity'].agg(['sum', 'count', 'mean'])
print('SKUs with positive total sales:', (sku_stats['sum'] > 0).sum())
print('SKUs with zero total sales:', (sku_stats['sum'] == 0).sum())
print('SKUs with negative total sales:', (sku_stats['sum'] < 0).sum())

# Sparsity
sku_days = df.groupby('ItemCode')['Date'].nunique()
total_days = df['Date'].nunique()
print('\n=== SPARSITY ===')
print('Total training days:', total_days)
print('Median active days per SKU:', sku_days.median())
print('SKUs with < 10 active days:', (sku_days < 10).sum())
print('SKUs with < 50 active days:', (sku_days < 50).sum())

# Profit weights
df['Cost Amount'] = df['Cost Amount'].astype(str).str.replace(',', '').astype(float)
profit = (df['SalesAmount'] - df['Cost Amount']).groupby(df['ItemCode']).sum()
print('\n=== PROFIT DISTRIBUTION ===')
print('SKUs with positive profit:', (profit > 0).sum())
print('SKUs with zero profit:', (profit == 0).sum())
print('SKUs with negative profit (weight=0):', (profit < 0).sum())
top_profit = profit[profit > 0].sort_values(ascending=False)
print('Top 10 SKUs by profit:')
print(top_profit.head(10))
cum_pct = top_profit.cumsum() / top_profit.sum()
n50 = (cum_pct < 0.50).sum()
n80 = (cum_pct < 0.80).sum()
print(f'\nTop {n50} SKUs account for 50% of profit')
print(f'Top {n80} SKUs account for 80% of profit')
