# FOREX-TREND-FORECASTING
It is an implementation of a Brunal University paper, titled FOREX TREND FORECASTING BASED ON LONG SHORT TERM MEMORY AND ITS VARIATIONS WITH HYBRID ACTIVATION FUNCTIONS which investigates the application of Long Short-Term Memory (LSTM) neural networks to foreign exchange (forex) market prediction and proposes Hybrid Activation Function (HAF) based LSTM models to improve forecasting accuracy.

# Important Aspects
1. Hybrid Actiavtion Function LSTM Model: The paper proposes a Hybrid Activation Function (HAF) LSTM model, which modifies the activation functions of the cell state (c_t) and hidden state (h_t) in an LSTM network. Instead of using only tanh, the model introduces a combination of tanh and sigmoid activations.
   The standard LSTM model uses tanh activation for both the cell state (c_t) and hidden state (h_t).
   The HAF-LSTM model changes the activation function of a subset of units from tanh to sigmoid.
   This modification leverages:
   Tanh (tanh(x)), which has steeper curves, allowing for faster training.
   Sigmoid (σ(x)), which enforces stability and helps prevent exploding gradients.
   The paper shows that this hybrid activation mechanism improves the model's ability to capture short-term and long-term dependencies in financial time series data.
   The best ratio observed in experiments was 90% tanh neurons and 10% sigmoid neurons, leading to higher forecasting accuracy.

3. New Validation Method: The paper introduces a New Validation Method for time series forecasting that addresses the limitations of traditional walk-forward validation.
   Problem with Standard Validation:
   Time series models, especially LSTMs, require sequential data for learning.
   Randomized cross-validation is not ideal, as it breaks the sequential nature of financial time series.
   Walk-forward validation, while better, may not generalize well across different time horizons.
   Proposed Validation Method:
   Uses a customized cross-validation approach designed specifically for LSTM-based forecasting.
   Ensures that each training fold contains only past data and validation folds contain only future data.
   Helps reduce overfitting by evaluating the model’s performance on multiple validation segments.
   Enhances model robustness by testing different market conditions.
   Implementation:
   The validation method splits the dataset into train-validation-test sets in a way that mimics real-world trading conditions.
   The test set remains completely unseen until final model evaluation.
   This method ensures that the LSTM model generalizes well across various time frames and forex pairs.

# Note
1. Random generated data is used, you can add data according to you.
2. Macroeconmoic indicators is not added yet.


