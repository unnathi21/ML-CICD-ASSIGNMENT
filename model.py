from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def load_data():
    iris = load_iris()
    return iris.data, iris.target
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy
if __name__ == "__main__":
    model, accuracy = train_model()
    print("ML Model: Logistic Regression")
    print("Dataset: Iris")
    print("Model Accuracy:", accuracy)
    sample_prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    print("Sample Prediction:", sample_prediction)
