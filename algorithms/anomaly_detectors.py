import numpy as np
import pandas as pd

class CustomAnomalyDetector:
    def __init__(self, neighbor_threshold=1, group_size_threshold=5):
        self.neighbor_threshold = neighbor_threshold
        self.group_size_threshold = group_size_threshold
    
    def group_anomalies(self, anomalies_indices):
        print(f">>>>>>>>>> anomalies_indices : {anomalies_indices}")
        anomalies_indices.sort()
        groups = []
        current_group = [anomalies_indices[0]]
        
        for index in anomalies_indices[1:]:
            if index <= current_group[-1] + self.neighbor_threshold:
                current_group.append(index)
            else:
                groups.append(current_group)
                current_group = [index]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def classify_anomalies(self, anomalies_indices):
        groups = self.group_anomalies(anomalies_indices)
        true_anomalies = []
        outliers = []
        
        for group in groups:
            if len(group) > self.group_size_threshold:
                true_anomalies.extend(group)
            else:
                outliers.extend(group)
        
        return true_anomalies, outliers
    
    def detect_anomalous_group(self, anomalies_indices):
        true_anomalies, outliers = self.classify_anomalies(anomalies_indices)
        
        return true_anomalies, outliers
    
    def detect_anomalous_data(self, anomalies_indices, data):
        true_anomalies, outliers = self.classify_anomalies(anomalies_indices)
        
        true_anomalies_data = data.loc[data.index.isin(true_anomalies)]
        outliers_data = data.loc[data.index.isin(outliers)]
        
        return true_anomalies_data, outliers_data

if __name__ == "__main__":
    anomalies_detected_indices = np.array([6, 32, 38, 42, 48, 49, 51, 64, 98])
    
    anomalous_data = pd.DataFrame({
        'sound_db': [83.327572, 42.610363, 78.822806, 31.131574, 30.288538],
        'noise_db': [22.936974, 71.776207, 27.259026, 86.485601, 34.230134],
        'breath_rate': [6.060764, 9.629911, 7.641558, 7.641455, 6.035145],
        'heart_rate': [53.916429, 49.021330, 44.180646, 51.810048, 52.409303],
        'temperature': [36.824841, 36.134683, 36.218936, 36.801204, 37.128940],
        'humidity': [59.082944, 30.137390, 45.236545, 63.339790, 33.209618],
        'dt_cr': [60, 0, 60, 180, 120]
    }, index=[1, 48, 49, 51, 98])
    
    detector = CustomAnomalyDetector(neighbor_threshold=2, group_size_threshold=2)
    true_anomalies, outliers = detector.detect_anomalous_group(anomalies_detected_indices)
    
    print("True Anomaly Data:")
    print(true_anomalies)
    print("\nOutlier Data:")
    print(outliers)
