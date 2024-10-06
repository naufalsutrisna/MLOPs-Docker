import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Directory to save the model
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(n_estimators=100, max_depth=None):
    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    print(f"Model with n_estimators={n_estimators}, max_depth={max_depth}, Accuracy={accuracy}")

# Training and saving the model
train_model(n_estimators=100, max_depth=None)
