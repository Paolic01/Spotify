
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SpotifyPreprocessor:
    def __init__(self, df, target='popularity'):
        self.df = df
        self.target = target
        self.scaler = StandardScaler()

    def split_features_target(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return X, y

    def train_test_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
