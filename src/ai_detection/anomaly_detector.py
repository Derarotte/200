#!/usr/bin/env python3
"""
AI-based network anomaly detection system
Uses machine learning to detect network anomalies and security threats
"""

import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class NetworkAnomalyDetector:
    """AI-powered network anomaly detection system"""
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.isolation_forest = None
        self.random_forest = None
        self.neural_network = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
    def load_traffic_data(self, data_file="data/traffic_log.json"):
        """Load and preprocess traffic data"""
        try:
            with open(data_file, 'r') as f:
                traffic_data = json.load(f)
            
            df = pd.DataFrame(traffic_data)
            return self.preprocess_data(df)
        except Exception as e:
            print(f"Error loading traffic data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess traffic data for ML models"""
        # Convert timestamp to datetime features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Extract host numbers from src and dst
        df['src_num'] = df['src'].str.extract('(\d+)').astype(int)
        df['dst_num'] = df['dst'].str.extract('(\d+)').astype(int)
        
        # Extract IP octets
        df['src_ip_last'] = df['src_ip'].str.split('.').str[-1].astype(int)
        df['dst_ip_last'] = df['dst_ip'].str.split('.').str[-1].astype(int)
        
        # Encode categorical variables
        df['protocol_encoded'] = self.label_encoder.fit_transform(df['protocol'])
        
        # Binary encoding for traffic type (0 = normal, 1 = anomaly)
        df['is_anomaly'] = (df['type'] == 'anomaly').astype(int)
        
        # Calculate flow-based features
        df = self._calculate_flow_features(df)
        
        # Select features for ML
        feature_cols = [
            'hour', 'minute', 'second', 'day_of_week',
            'src_num', 'dst_num', 'src_ip_last', 'dst_ip_last',
            'protocol_encoded', 'packet_size', 'bandwidth',
            'flow_duration', 'packet_rate', 'byte_rate'
        ]
        
        self.feature_columns = feature_cols
        return df
    
    def _calculate_flow_features(self, df):
        """Calculate additional flow-based features"""
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate flow duration (time since first packet in flow)
        df['flow_key'] = df['src'] + '->' + df['dst'] + ':' + df['protocol']
        df['flow_start'] = df.groupby('flow_key')['timestamp'].transform('min')
        df['flow_duration'] = (df['timestamp'] - df['flow_start']).dt.total_seconds()
        
        # Calculate packet and byte rates
        df['packet_rate'] = df['packet_size'] / (df['flow_duration'] + 1)
        df['byte_rate'] = df['bandwidth'] / (df['flow_duration'] + 1)
        
        return df
    
    def train_models(self, df):
        """Train multiple ML models for anomaly detection"""
        X = df[self.feature_columns].fillna(0)
        y = df['is_anomaly']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        self.isolation_forest.fit(X_train_scaled)
        
        print("Training Random Forest...")
        self.random_forest = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.random_forest.fit(X_train_scaled, y_train)
        
        print("Training Neural Network...")
        self.neural_network = self._build_neural_network(X_train_scaled.shape[1])
        self.neural_network.fit(
            X_train_scaled, y_train,
            epochs=50, batch_size=32, validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def _build_neural_network(self, input_dim):
        """Build neural network for anomaly detection"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate trained models"""
        print("\n=== Model Evaluation ===")
        
        # Isolation Forest
        if_pred = self.isolation_forest.predict(X_test)
        if_pred = (if_pred == -1).astype(int)  # Convert to binary
        print("\nIsolation Forest:")
        print(classification_report(y_test, if_pred))
        
        # Random Forest
        rf_pred = self.random_forest.predict(X_test)
        print("\nRandom Forest:")
        print(classification_report(y_test, rf_pred))
        
        # Neural Network
        nn_pred = (self.neural_network.predict(X_test) > 0.5).astype(int)
        print("\nNeural Network:")
        print(classification_report(y_test, nn_pred.flatten()))
    
    def detect_anomalies(self, new_data):
        """Detect anomalies in new data using ensemble of models"""
        if not self.is_trained:
            print("Models not trained yet!")
            return None
        
        # Preprocess new data
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            new_data = pd.DataFrame(new_data)
        
        processed_data = self.preprocess_data(new_data)
        X = processed_data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        if_pred = (self.isolation_forest.predict(X_scaled) == -1).astype(int)
        rf_pred = self.random_forest.predict(X_scaled)
        nn_pred = (self.neural_network.predict(X_scaled) > 0.5).astype(int).flatten()
        
        # Ensemble prediction (majority vote)
        ensemble_pred = (if_pred + rf_pred + nn_pred >= 2).astype(int)
        
        # Get prediction probabilities
        rf_prob = self.random_forest.predict_proba(X_scaled)[:, 1]
        nn_prob = self.neural_network.predict(X_scaled).flatten()
        
        results = []
        for i in range(len(X)):
            results.append({
                'isolation_forest': int(if_pred[i]),
                'random_forest': int(rf_pred[i]),
                'neural_network': int(nn_pred[i]),
                'ensemble': int(ensemble_pred[i]),
                'rf_probability': float(rf_prob[i]),
                'nn_probability': float(nn_prob[i]),
                'anomaly_score': float((rf_prob[i] + nn_prob[i]) / 2)
            })
        
        return results
    
    def save_models(self, model_dir="models"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.is_trained:
            joblib.dump(self.isolation_forest, f"{model_dir}/isolation_forest.pkl")
            joblib.dump(self.random_forest, f"{model_dir}/random_forest.pkl")
            joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
            joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.pkl")
            joblib.dump(self.feature_columns, f"{model_dir}/feature_columns.pkl")
            self.neural_network.save(f"{model_dir}/neural_network.h5")
            print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir="models"):
        """Load pre-trained models"""
        try:
            self.isolation_forest = joblib.load(f"{model_dir}/isolation_forest.pkl")
            self.random_forest = joblib.load(f"{model_dir}/random_forest.pkl")
            self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
            self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
            self.feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
            self.neural_network = keras.models.load_model(f"{model_dir}/neural_network.h5")
            self.is_trained = True
            print(f"Models loaded from {model_dir}/")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def visualize_results(self, df, save_plots=True):
        """Create visualizations of anomaly detection results"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Traffic type distribution
        axes[0, 0].pie(df['type'].value_counts(), labels=df['type'].value_counts().index, autopct='%1.1f%%')
        axes[0, 0].set_title('Traffic Type Distribution')
        
        # Protocol distribution
        df['protocol'].value_counts().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Protocol Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Packet size distribution by traffic type
        sns.boxplot(data=df, x='type', y='packet_size', ax=axes[1, 0])
        axes[1, 0].set_title('Packet Size by Traffic Type')
        
        # Bandwidth distribution by traffic type
        sns.boxplot(data=df, x='type', y='bandwidth', ax=axes[1, 1])
        axes[1, 1].set_title('Bandwidth by Traffic Type')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('results/anomaly_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualization saved to results/anomaly_analysis.png")
        
        plt.show()


def main():
    """Main function for training and testing anomaly detection"""
    detector = NetworkAnomalyDetector()
    
    print("Loading traffic data...")
    df = detector.load_traffic_data()
    
    if df is not None:
        print(f"Loaded {len(df)} traffic records")
        print(f"Normal traffic: {len(df[df['type'] == 'normal'])}")
        print(f"Anomalous traffic: {len(df[df['type'] == 'anomaly'])}")
        
        print("\nTraining models...")
        X_test, y_test = detector.train_models(df)
        
        print("\nSaving models...")
        detector.save_models()
        
        print("\nCreating visualizations...")
        detector.visualize_results(df)
        
        # Test real-time detection
        print("\nTesting real-time detection...")
        test_traffic = {
            'timestamp': datetime.now().isoformat(),
            'src': 'h1',
            'dst': 'h2',
            'src_ip': '10.0.0.1',
            'dst_ip': '10.0.0.2',
            'protocol': 'HTTP',
            'type': 'normal',
            'packet_size': 1200,
            'bandwidth': 50.0
        }
        
        results = detector.detect_anomalies(test_traffic)
        print(f"Detection results: {results[0]}")
    
    else:
        print("No traffic data found. Please run traffic generator first.")


if __name__ == '__main__':
    main()