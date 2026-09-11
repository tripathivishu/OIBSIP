
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set visual style for clean graphs
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ==========================================
# 1. DATA LOADING & INSPECTION
# ==========================================
print('--- 1. LOADING DATASET ---')
file_path = '/Users/vishutiwari/Documents/Unemployment in India.csv'

try:
  df = pd.read_csv(file_path)
except FileNotFoundError:
  print(
      f'Error: Could not find {file_path}. Please download it from Kaggle and'
      ' place it in the folder.'
  )
  exit()

print(f'Dataset Shape (Rows, Columns): {df.shape}\n')

print('--- DATA INFO ---')
print(df.info())

print('\n--- NULL VALUE CHECK ---')
print(df.isnull().sum())

# Clean column names (strip leading/trailing whitespaces)
df.columns = df.columns.str.strip()

# ==========================================
# 2. DATA CLEANING & TYPE CONVERSION
# ==========================================
print('\n--- 2. DATA CLEANING & TYPE CONVERSION ---')

# Convert Date column to datetime format
# Note: Handle date formatting based on your specific CSV file structure
df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')

# Drop rows where Date conversion failed
df = df.dropna(subset=['Date'])

# Rename columns for standard mapping (adjust if your CSV uses different names)
# Common columns: 'Region', 'Date', 'Estimated Unemployment Rate (%)',
# 'Estimated Employed', 'Estimated Labour Participation Rate (%)'
expected_cols = {
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
}

# Rename only the columns that exist in the dataframe
df = df.rename(columns=expected_cols)

# Ensure numeric types
for col in [
    'Unemployment_Rate',
    'Employed',
    'Labour_Participation_Rate',
]:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print('Data preprocessing complete.\n')

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print('--- 3. EXPLORATORY DATA ANALYSIS ---')

# A. Region-wise average unemployment rates
region_avg = (
    df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False)
)
print('\nTop 5 States/Regions with Highest Average Unemployment Rates:')
print(region_avg.head(5))

# ==========================================
# 4. VISUALISATIONS
# ==========================================
print('\n--- 4. GENERATING VISUALISATIONS ---')
print(
    'Generating graphs (close each graph window to proceed to the next step)...'
)

# Visualisation 1: Bar Chart - Top 10 States with Highest Average Unemployment Rate
plt.figure(figsize=(12, 6))
top_10 = region_avg.head(10)
sns.barplot(
    x=top_10.values, y=top_10.index, palette='viridis', hue=top_10.index, legend=False
)
plt.title(
    'Top 10 States/Regions with Highest Average Unemployment Rate',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Average Unemployment Rate (%)', fontsize=12)
plt.ylabel('State / Region', fontsize=12)
plt.tight_layout()
plt.show()

# Visualisation 2: Time-Series Line Chart for 3 Major States
# Let's dynamically pick top 3 states with maximum records or select 3 prominent ones
major_states = region_avg.head(3).index.tolist()
plt.figure(figsize=(14, 7))
for state in major_states:
  state_data = df[df['State'] == state].sort_values('Date')
  plt.plot(
      state_data['Date'],
      state_data['Unemployment_Rate'],
      marker='o',
      linewidth=2,
      label=state,
  )

plt.title(
    'Unemployment Rate Over Time for Top 3 Impacted States',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualisation 3: Heatmap - Correlation Matrix
plt.figure(figsize=(8, 6))
numeric_cols = [
    'Unemployment_Rate',
    'Employed',
    'Labour_Participation_Rate',
]
# Keep only columns that exist
available_numeric = [c for c in numeric_cols if c in df.columns]
corr_matrix = df[available_numeric].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title(
    'Correlation Heatmap: Unemployment, Employment & Labour Participation',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()
plt.show()

# ==========================================
# 5. PRE-COVID VS. POST-COVID COMPARISON
# ==========================================
print('\n--- 5. PRE-COVID VS. POST-COVID ANALYSIS ---')

# Defining India's strict lockdown / COVID outbreak benchmark (e.g., March 2020)
covid_outbreak_date = pd.to_datetime('2020-03-01')

pre_covid = df[df['Date'] < covid_outbreak_date]
post_covid = df[df['Date'] >= covid_outbreak_date]

pre_covid_mean = pre_covid['Unemployment_Rate'].mean()
post_covid_mean = post_covid['Unemployment_Rate'].mean()

print(f'Average Unemployment Rate (Pre-COVID): {pre_covid_mean:.2f}%')
print(f'Average Unemployment Rate (Post-COVID): {post_covid_mean:.2f}%')

print('\nAnalysis completed successfully!')