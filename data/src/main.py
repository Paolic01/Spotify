import pandas as pd

df = pd.read_csv("/Users/pasqualepaolicelli/Downloads/Spotify/dataset.csv")

print(df.head())

from data_loader import SpotifyDataLoader
from preprocessor import SpotifyPreprocessor
from model import SpotifyModel

def main():
    # 1️⃣ Carica e pulisci i dati
    loader = SpotifyDataLoader("data/spotify_tracks.csv")
    df = loader.load_data()
    df = loader.clean_data()

    # 2️⃣ Preprocessing
    prep = SpotifyPreprocessor(df, target='popularity')
    X, y = prep.split_features_target()
    X_train, X_test, y_train, y_test = prep.train_test_split(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    # 3️⃣ Crea e allena il modello
    model = SpotifyModel(input_dim=X_train_scaled.shape[1], model_type='regression')
    model.train(X_train_scaled, y_train, epochs=50)
    model.evaluate(X_test_scaled, y_test)

if __name__ == "__main__":
    main()
