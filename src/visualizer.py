import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, figsize=(12, 8)):
        """Inizializza il visualizzatore dei dati"""
        self.figsize = figsize
        
    def plot_distribution(self, df, column, title=None):
        """Visualizza la distribuzione di una variabile"""
        plt.figure(figsize=self.figsize)
        sns.histplot(df[column], kde=True)
        plt.title(title or f'Distribuzione di {column}')
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, df, target_column=None):
        """Visualizza la matrice di correlazione"""
        plt.figure(figsize=self.figsize)
        
        # Calcola la matrice di correlazione
        corr = df.corr()
        
        # Crea una maschera per il triangolo superiore
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Crea la heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', linewidths=0.5)
        
        plt.title('Matrice di Correlazione')
        plt.tight_layout()
        plt.show()
        
        # Se specificato, mostra le correlazioni con la variabile target
        if target_column and target_column in df.columns:
            target_corr = corr[target_column].sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=target_corr.index, y=target_corr.values)
            plt.title(f'Correlazione con {target_column}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            
    def plot_feature_importance(self, feature_names, importances, title='Feature Importance'):
        """Visualizza l'importanza delle features"""
        indices = np.argsort(importances)
        plt.figure(figsize=self.figsize)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_vs_actual(self, y_true, y_pred, title='Predizioni vs Valori Reali'):
        """Visualizza predizioni vs valori reali"""
        plt.figure(figsize=self.figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Linea di perfetta predizione
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Valori Reali')
        plt.ylabel('Predizioni')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves(self, history):
        """Visualizza le curve di apprendimento di un modello Keras"""
        plt.figure(figsize=self.figsize)
        
        # Plot della loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot della metrica (se disponibile)
        if 'mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epoche')
            plt.ylabel('MAE')
            plt.legend()
        
        plt.tight_layout()
        plt.show()