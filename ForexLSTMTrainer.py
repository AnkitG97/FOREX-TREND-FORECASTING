import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN
from sklearn.model_selection import train_test_split

from ForexTechnicalIndicators import ForexTechnicalIndicators
from HyperparameterTuner import HyperparameterTuner
from HybridLSTMCell import HybridLSTMCell
from new_validation_method import new_validation_method

class ForexLSTMTrainer:
    def __init__(self, ohlc_df, timesteps=100, test_size=0.1, val_size=0.1, num_classes=3):
        """
        Initializes the Forex LSTM Trainer.

        Parameters:
        - ohlc_df (DataFrame): Input OHLC forex data.
        - timesteps (int): Number of time steps for LSTM sequences.
        - test_size (float): Proportion of data for testing.
        - val_size (float): Proportion of data for validation.
        - num_classes (int): Number of target classes (default: 3).
        """
        self.ohlc_df = ohlc_df.copy()
        self.timesteps = timesteps
        self.test_size = test_size
        self.val_size = val_size
        self.num_classes = num_classes

    def preprocess_data(self):
        """Prepares the dataset by creating technical indicators, scaling, and splitting."""
        # Create the target variable (5-day indicator)
        self.ohlc_df["5th_day_indicator"] = (
            self.ohlc_df["Close"].shift(-5) - self.ohlc_df["Close"]
        ).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Generate technical indicators
        indicator = ForexTechnicalIndicators(self.ohlc_df)
        df_with_indicators = indicator.calculate_all_indicators()

        # Select relevant features (exclude OHLC and Date columns)
        cols = list(set(df_with_indicators.columns) - {"Open", "High", "Low", "Close", "Date"})
        df_clean = df_with_indicators[cols].dropna()

        # Separate features and target
        target = df_clean["5th_day_indicator"]
        df_clean = df_clean.drop(columns=["5th_day_indicator"])

        # Scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_features = scaler.fit_transform(df_clean)

        # Split into training, validation, and test sets
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = new_validation_method(
            data=self.scaled_features,
            target=target,
            timesteps=self.timesteps,
            test_size=self.test_size,
            val_size=self.val_size
        )

        # Convert targets to categorical (one-hot encoding)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_val = tf.keras.utils.to_categorical(self.y_val, num_classes=self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=self.num_classes)

    def tune_hyperparameters(self):
        """Tunes hyperparameters using HyperparameterTuner."""
        self.tuner = HyperparameterTuner(self.X_train, self.y_train)
        self.best_params = self.tuner.tune_sequentially()

        print("\nFinal Best Parameters:")
        for k, v in self.best_params.items():
            print(f"{k}: {v}")

    def build_model(self):
        """Builds the LSTM model using the best hyperparameters."""
        self.model = Sequential()
        
        # Add LSTM layers
        for _ in range(self.best_params['layers']):
            self.model.add(RNN(
                HybridLSTMCell(self.best_params['neurons'], 
                               ratio=self.best_params['activation_ratio']),
                input_shape=(self.best_params['timesteps'], self.X_train.shape[2])
            ))
        
        # Output layer (Softmax for classification)
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.best_params['learning_rate']
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self):
        """Trains the model using early stopping."""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=100,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=1000,
            batch_size=self.best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )

    def evaluate_model(self):
        """Evaluates the trained model on the test set."""
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    def run_pipeline(self):
        """Runs the complete pipeline: preprocessing, tuning, training, and evaluation."""
        print("\n **Step 1: Preprocessing Data**")
        self.preprocess_data()

        print("\n **Step 2: Hyperparameter Tuning**")
        self.tune_hyperparameters()

        print("\n **Step 3: Building Model**")
        self.build_model()

        print("\n **Step 4: Training Model**")
        self.train_model()

        print("\n **Step 5: Evaluating Model**")
        self.evaluate_model()
