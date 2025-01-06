import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from joblib import dump

def train_model():
    # Dummy dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'target': [3, 5, 7, 9, 11]
    })

    # Train-test split
    X = data[['feature1', 'feature2']]
    y = data['target']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    dump(model, 'linear_model.joblib')
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
