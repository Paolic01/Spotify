import pandas as pd
import numpy as np
from datasets import load_dataset

class DataLoader:
    """Classe per il caricamento e l'analisi preliminare dei dati"""
    
    def __init__(self, dataset_name='maharshipandya/spotify-tracks-dataset'):
        self.dataset_name = dataset_name
        self.data = None
    
    def load_data(self):
        """Carica il dataset da HuggingFace"""
        print(f"Caricamento dataset da HuggingFace: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)
        self.data = pd.DataFrame(dataset['train'])
        print(f"✓ Dataset caricato correttamente")
        self._show_info()
        return self.data
    
    def _show_info(self):
        """Mostra informazioni sul dataset"""
        print(f"\nInfo dataset:")
        print(f"  - Righe: {self.data.shape[0]}")
        print(f"  - Colonne: {self.data.shape[1]}")
        print(f"  - Valori mancanti: {self.data.isnull().sum().sum()}")
        
        if 'popularity' in self.data.columns:
            print(f"\nDistribuzione target (popularity):")
            print(f"  - Media: {self.data['popularity'].mean():.2f}")
            print(f"  - Mediana: {self.data['popularity'].median():.2f}")
            print(f"  - Min: {self.data['popularity'].min()}")
            print(f"  - Max: {self.data['popularity'].max()}")
    
    def get_feature_columns(self):
        """Restituisce le colonne numeriche utilizzabili come features"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Unnamed: 0', 'popularity']
        features = [col for col in numeric_cols if col not in exclude_cols]
        return features