import os
import sys
sys.path.append('src')  # Add the src directory to the Python path

from train import train_model

def test_model_training():
    train_model()
    assert os.path.exists("../models/linear_model.joblib")
