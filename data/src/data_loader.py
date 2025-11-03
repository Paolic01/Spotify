import pandas as pd

class SpotifyDataLoader:
    def __init__(self, path: str):
        self.path = path
        self.df = None

    def load_data(self):
        """Carica il dataset locale"""
        self.df = pd.read_csv(self.path)
        print(f"✅ Dataset caricato: {len(self.df)} righe, {len(self.df.columns)} colonne.")
        return self.df

    def clean_data(self):
        """Rimuove colonne inutili e gestisce valori mancanti"""
        if self.df is None:
            raise ValueError("Carica prima i dati con load_data().")

        # Colonne inutili per la predizione
        drop_cols = ['track_id', 'track_name', 'album_name', 'artists']
        self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)

        # Conversione booleana
        if 'explicit' in self.df.columns:
            self.df['explicit'] = self.df['explicit'].astype(int)

        # Rimuove righe con valori NaN
        self.df.dropna(inplace=True)
        print("🧹 Pulizia completata.")
        return self.df
