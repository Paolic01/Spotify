from tensorflow import keras
from tensorflow.keras import layers

class SpotifyModel:
    def __init__(self, input_dim, model_type='regression'):
        self.model_type = model_type
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu')
        ])

        if model_type == 'regression':
            self.model.add(layers.Dense(1))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif model_type == 'classification':
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test)
        print(f"📊 Valutazione modello → {results}")
        return results
