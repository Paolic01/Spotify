import pandas as pd
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import NeuralNetworkModel
from src.evaluator import Evaluator
import matplotlib.pyplot as plt

def main():
    print("=" * 50)
    print("SPOTIFY TRACK POPULARITY PREDICTION")
    print("=" * 50)
    
    # 1. Caricamento dati
    print("\n[1/5] Caricamento dataset...")
    loader = DataLoader()
    df = loader.load_data()
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # 2. Preprocessing
    print("\n[2/5] Preprocessing dei dati...")
    preprocessor = Preprocessor(target='popularity')
    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess(df)
    print(f"Training set: {X_train.shape[0]} campioni")
    print(f"Test set: {X_test.shape[0]} campioni")
    print(f"Features: {X_train.shape[1]}")
    
    # 3. Creazione e training del modello
    print("\n[3/5] Training della rete neurale...")
    model = NeuralNetworkModel(input_dim=X_train.shape[1])
    history = model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # 4. Predizioni
    print("\n[4/5] Generazione predizioni...")
    y_pred = model.predict(X_test)
    
    # 5. Valutazione
    print("\n[5/5] Valutazione del modello...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("RISULTATI FINALI")
    print("=" * 50)
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Visualizzazione training history
    evaluator.plot_training_history(history)
    
    # Visualizzazione predizioni vs valori reali
    evaluator.plot_predictions(y_test, y_pred)
    
    # Salvataggio modello
    print("\n[INFO] Salvataggio modello...")
    model.save_model('models/spotify_model.h5')
    print("Modello salvato in: models/spotify_model.h5")
    
    plt.show()

if __name__ == "__main__":
    main()