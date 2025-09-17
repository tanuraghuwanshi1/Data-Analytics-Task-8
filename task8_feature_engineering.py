
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
 df = pd.read_csv('House-Prices-Advanced-Regression-Dataset.csv')

# Synthetic features for demonstration
np.random.seed(0)
df['OverallQual'] = np.random.randint(1, 10, size=df.shape[0])
df['TotalSF'] = np.random.randint(500, 4000, size=df.shape[0])
df['Neighborhood'] = np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], size=df.shape[0])
df['YearBuilt'] = np.random.randint(1950, 2010, size=df.shape[0])
df['GarageCars'] = np.random.choice([0, 1, 2, 3], size=df.shape[0])

# One-hot encoding
 df = pd.get_dummies(df, columns=['Neighborhood'], prefix='Nbr')

# Create Age feature
current_year = 2025
df['Age'] = current_year - df['YearBuilt']

# Log transform TotalSF if skewed
from scipy.stats import skew
if skew(df['TotalSF']) > 1:
    df['TotalSF_log'] = np.log1p(df['TotalSF'])
else:
    df['TotalSF_log'] = df['TotalSF']

# Prepare features and target
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save cleaned engineered data
engineered_df = df.drop(['Id'], axis=1)
engineered_df.to_csv('cleaned_engineered_data.csv', index=False)

# Save model
joblib.dump(model, 'random_forest_model.pkl')

