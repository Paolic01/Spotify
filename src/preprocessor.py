import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    """Classe per preprocessing e feature engineering"""
    
    def __init__(self, target='popularity', test_size=0.2, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess(self, df):
        """Esegue il preprocessing completo del dataset"""
        df_clean = self._clean_data(df.copy())
        X, y = self._prepare_features(df_clean)
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler
    
    def _clean_data(self, df):
        """Pulisce il dataset da valori mancanti e duplicati"""
        print("  - Rimozione valori mancanti...")
        df = df.dropna()
        print("  - Rimozione duplicati...")
        df = df.drop_duplicates()
        print(f"  ✓ Dataset pulito: {df.shape[0]} righe rimanenti")
        return df
    
    def _prepare_features(self, df):
        """Prepara le features per il training"""
        numeric_features = [
            'duration_ms', 'danceability', 'energy', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature'
        ]
        
        if 'explicit' in df.columns:
            df['explicit'] = df['explicit'].astype(int)
            numeric_features.append('explicit')
        
        if 'track_genre' in df.columns:
            print("  - Encoding track_genre...")
            self.label_encoders['track_genre'] = LabelEncoder()
            df['track_genre_encoded'] = self.label_encoders['track_genre'].fit_transform(df['track_genre'])
            numeric_features.append('track_genre_encoded')
        
        available_features = [f for f in numeric_features if f in df.columns]
        X = df[available_features].values
        y = df[self.target].values
        print(f"  ✓ Features preparate: {len(available_features)} features")
        return X, y
    
    def _split_data(self, X, y):
        """Divide il dataset in train e test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"  ✓ Split completato: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    def _scale_features(self, X_train, X_test):
        """Normalizza le features usando StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("  ✓ Features normalizzate")
        return X_train_scaled, X_test_scaled