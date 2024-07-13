import joblib
from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.01)

    def train(self, X):
        self.model.fit(X)
        joblib.dump(self.model, 'isolation_forest_model.pkl')

    def load(self, file_path='isolation_forest_model.pkl'):
        self.model = joblib.load(file_path)

    def predict(self, X):
        y_pred = self.model.predict(X)
        anomalies = (y_pred == -1)
        return anomalies