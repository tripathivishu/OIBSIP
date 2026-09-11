
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print('--- 1. LOADING DATASET ---')
# Load the built-in Iris dataset from scikit-learn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(
    {0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]}
)
print(f'Dataset successfully loaded! Shape: {df.shape}\n')


print('--- 2. EXPLORATORY DATA ANALYSIS (EDA) ---')
print('Data Types & Info:')
print(df.info())
print('\nNull Value Check:')
print(df.isnull().sum())
print('\nDescriptive Statistics:')
print(df.describe())
print('\n')


print('--- 3. DATA VISUALISATION ---')
print('Opening visual graphs (close each graph window to continue script)...')

# Set visual style
sns.set_theme(style='whitegrid')

# Pairplot
sns.pairplot(
    df.drop('target', axis=1), hue='species', markers=['o', 's', 'D'], palette='Set2'
)
plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
plt.show()

# Box plots
plt.figure(figsize=(12, 6))
for i, col in enumerate(iris.feature_names):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(x='species', y=col, data=df, palette='Set2')
  plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


print('--- 4. FEATURE SELECTION DISCUSSION ---')
print(
    'Observation: Petal Length and Petal Width show the clearest separation'
)
print('between species (especially Setosa). All 4 features are retained.')
print('\n')


print('--- 5. TRAIN / TEST SPLIT ---')
X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Training set samples: {X_train.shape[0]}')
print(f'Testing set samples: {X_test.shape[0]}\n')


print('--- 6. MODEL TRAINING & EVALUATION ---')
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

for name, model in models.items():
  print(f'=== Training {name} ===')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)
  report = classification_report(y_test, y_pred, target_names=iris.target_names)

  print(f'Accuracy: {acc * 100:.2f}%')
  print('Confusion Matrix:')
  print(cm)
  print('Classification Report:')
  print(report)
  print('=' * 40 + '\n')


print('--- 7. FINAL CONCLUSION ---')
print(
    'Both models perform exceptionally well, achieving high accuracy due to'
    ' the clean separation of features in the Iris dataset.'
)
print(
    'Random Forest and Logistic Regression are both robust choices for this'
    ' task.'
)