import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

class Evaluator:
    """Classe per valutazione e visualizzazione dei risultati"""
    
    def evaluate(self, y_true, y_pred):
        """Calcola le metriche di valutazione"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        return metrics
    
    def plot_training_history(self, history):
        """Visualizza l'andamento del training"""
        Path('results').mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("\n  ✓ Grafico training salvato: results/training_history.png")
    
    def plot_predictions(self, y_true, y_pred):
        """Visualizza predizioni vs valori reali"""
        Path('results').mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Actual Popularity', fontsize=12)
        axes[0].set_ylabel('Predicted Popularity', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Popularity', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/predictions.png', dpi=300, bbox_inches='tight')
        print("  ✓ Grafico predizioni salvato: results/predictions.png")
    
    def plot_feature_importance(self, feature_names, importances):
        """Visualizza l'importanza delle features"""
        Path('results').mkdir(exist_ok=True)
        sorted_idx = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances[sorted_idx])
        plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        print("  ✓ Feature importance salvata: results/feature_importance.png")