
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set visual style
sns.set_theme(style='whitegrid')

# =========================================
# 1. DATA LOADING & INSPECTION
# ==========================================
print('--- 1. LOADING DATASET ---')
file_path = (
    '/Users/vishutiwari/Documents/car data.csv'  
)

try:
  df = pd.read_csv(file_path)
except FileNotFoundError:
  print(
      f'Error: Could not find {file_path}. Please download it from Kaggle and'
      ' place it in the folder.'
  )
  exit()

print(f'Original Dataset Shape: {df.shape}\n')
print(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================
print('\n--- 2. DATA CLEANING ---')

# Remove duplicate rows
initial_count = len(df)
df = df.drop_duplicates()
print(f'Removed {initial_count - len(df)} duplicate rows.')

# Check for null values
print('Null values per column:\n', df.isnull().sum())

# Clean categorical text columns (fix capitalization mismatches like "Petrol" vs "petrol")
categorical_cols = [
    col
    for col in ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
    if col in df.columns
]
for col in categorical_cols:
  if df[col].dtype == 'object':
    df[col] = df[col].str.strip().str.capitalize()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print('\n--- 3. FEATURE ENGINEERING ---')

# Calculate car age based on the manufacturing year (Current year: 2026)
if 'Year' in df.columns:
  current_year = 2026
  df['Car_Age'] = current_year - df['Year']

# Extract brand from car name column (assuming column is 'Car_Name' or 'Name')
name_col = 'Car_Name' if 'Car_Name' in df.columns else 'Name'
if name_col in df.columns:
  df['Brand'] = df[name_col].apply(lambda x: str(x).split()[0].capitalize())
  # Drop the raw name column as it has too many unique identifier values
  df = df.drop(columns=[name_col])

print('Feature engineering completed. New columns added: Car_Age, Brand')

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (EDA) & VISUALISATION
# ==========================================
print('\n--- 4. VISUALISATIONS ---')
print('Generating graphs (close each window to proceed)...')

# A. Distribution of Selling Prices
plt.figure(figsize=(10, 5))
sns.histplot(df['Selling_Price'], kde=True, color='purple')
plt.title('Distribution of Car Selling Prices', fontsize=14, fontweight='bold')
plt.xlabel('Selling Price (Lakhs)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# B. Price vs Fuel Type Box Plot
if 'Fuel_Type' in df.columns and 'Selling_Price' in df.columns:
  plt.figure(figsize=(10, 5))
  sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, palette='Set2')
  plt.title('Selling Price vs. Fuel Type', fontsize=14, fontweight='bold')
  plt.show()

# C. Price vs Car Age Scatter Plot
if 'Car_Age' in df.columns:
  plt.figure(figsize=(10, 5))
  sns.scatterplot(x='Car_Age', y='Selling_Price', data=df, alpha=0.7, color='b')
  plt.title('Selling Price vs. Car Age', fontsize=14, fontweight='bold')
  plt.xlabel('Car Age (Years)', fontsize=12)
  plt.ylabel('Selling Price (Lakhs)', fontsize=12)
  plt.show()

# ==========================================
# 5. ENCODING & CORRELATION HEATMAP
# ==========================================
print('\n--- 5. ENCODING & CORRELATION ---')

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Correlation Heatmap for numerical features
plt.figure(figsize=(10, 8))
corr_matrix = df_encoded.corr(numeric_only=True)
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
)
plt.title(
    'Feature Correlation Heatmap', fontsize=14, fontweight='bold'
  )
plt.show()

# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================
print('\n--- 6. TRAIN / TEST SPLIT ---')

# Target variable is 'Selling_Price'
X = df_encoded.drop(columns=['Selling_Price'])
y = df_encoded['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Training features shape: {X_train.shape}')
print(f'Testing features shape: {X_test.shape}\n')

# ==========================================
# 7. MODEL TRAINING & EVALUATION
# ==========================================
print('--- 7. MODEL TRAINING & EVALUATION ---')

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
}

trained_models = {}

for name, model in models.items():
  print(f'=== Training {name} ===')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  # Evaluation metrics
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  print(f'Mean Absolute Error (MAE): {mae:.4f}')
  print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
  print(f'R² Score: {r2:.4f}\n')

  trained_models[name] = model

# ==========================================
# 8. FEATURE IMPORTANCE (RANDOM FOREST)
# ==========================================
print('--- 8. FEATURE IMPORTANCE CHART ---')

rf_model = trained_models['Random Forest Regressor']
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title(
    'Top 10 Feature Importances (Random Forest)',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

print('Task completed successfully!')