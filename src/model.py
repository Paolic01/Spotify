import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

class NeuralNetworkModel:
    """Classe per la rete neurale di predizione della popolarità"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Costruisce l'architettura della rete neurale"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("\n  ✓ Modello creato")
        print(f"    Architettura: {self.input_dim} -> 128 -> 64 -> 32 -> 16 -> 1")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Allena il modello con callbacks"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        print(f"\n  Training in corso (epochs={epochs}, batch_size={batch_size})...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n  ✓ Training completato!")
        return history
    
    def predict(self, X):
        """Effettua predizioni"""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def save_model(self, filepath):
        """Salva il modello in formato Keras"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"  ✓ Modello salvato: {filepath}")
    
    def load_model(self, filepath):
        """Carica un modello salvato"""
        self.model = keras.models.load_model(filepath)
    
    def get_summary(self):
        """Restituisce il summary del modello"""
        return self.model.summary()