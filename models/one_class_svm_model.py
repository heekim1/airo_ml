import joblib
from sklearn.svm import OneClassSVM

class OneClassSVMModel:
    def __init__(self):
        self.model = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)

    def train(self, X):
        self.model.fit(X)
        joblib.dump(self.model, 'one_class_svm_model.pkl')

    def load(self, file_path='one_class_svm_model.pkl'):
        self.model = joblib.load(file_path)

    def predict(self, X):
        y_pred = self.model.predict(X)
        anomalies = (y_pred == -1)
        return anomalies