import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Sample dataset
sample_data = pd.DataFrame({
    'Size (sq ft)': [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5],
    'Age (years)': [20, 15, 10, 8, 5, 3, 1],
    'Price ($)': [300000, 320000, 340000, 360000, 380000, 400000, 420000]
})

X = sample_data[['Size (sq ft)', 'Bedrooms', 'Age (years)']]
y = sample_data['Price ($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
dump(model, 'house_price_model.joblib')
