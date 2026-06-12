# Tesla Stock Price Prediction using Deep Learning

## Overview

This project focuses on predicting Tesla stock prices using Deep Learning models such as RNN, LSTM, and GRU. The system analyzes historical stock market data and forecasts future stock trends using time-series forecasting techniques.

The project also includes hyperparameter tuning, model evaluation, trend prediction, visualization, and a Streamlit-based web application for user interaction.

---

## Features

* Stock price prediction using:

  * Recurrent Neural Network (RNN)
  * Long Short-Term Memory (LSTM)
  * Gated Recurrent Unit (GRU)

* Hyperparameter Optimization:

  * Random Search
  * Bayesian Optimization

* Time-series forecasting

* Future trend prediction

* Interactive visualizations using Plotly

* Streamlit web application

* Model saving and loading

* Prediction for:

  * Entered date
  * 5-day future trend
  * 10-day future trend

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Plotly
* Streamlit
* Keras Tuner

---

## Dataset

The project uses Tesla historical stock market data containing:

* Open Price
* High Price
* Low Price
* Close Price
* Adjusted Close Price
* Volume

The `Adj Close` column is used as the target variable for prediction.

---

## Data Preprocessing

The following preprocessing techniques were applied:

* Handling missing values
* Date formatting
* Feature scaling using MinMaxScaler
* Sequence generation for time-series forecasting

Sequence length used:

* Previous 60 days → Next day prediction

---

## Deep Learning Models

### 1. RNN Model

* Basic recurrent neural network
* Captures sequential dependencies
* Used as baseline model

### 2. LSTM Model

* Handles long-term dependencies
* Uses memory cells and gates
* Achieved best prediction performance

### 3. GRU Model

* Simplified version of LSTM
* Faster training
* Lower computational complexity

---

## Hyperparameter Tuning

### RNN

* Technique Used: Random Search

### LSTM

* Technique Used: Bayesian Optimization

### GRU

* Technique Used: Random Search

Tuned Parameters:

* Number of units
* Dropout rate
* Optimizer
* Learning rate

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

These metrics help measure prediction accuracy and business impact.

---

## Best Model

LSTM was selected as the final prediction model because:

* It achieved the highest accuracy
* Better captured long-term stock dependencies
* Lower prediction error
* Stronger trend learning capability

---

## Visualizations

The project includes:

* Actual vs Predicted Stock Price Graph
* Training vs Validation Loss Curve
* Trend Prediction Graphs using Plotly

---

## Streamlit Web Application

The Streamlit app allows users to:

* Enter a date
* Predict stock price
* View 5-day trend
* View 10-day trend
* Interact with prediction charts

Run the application:

```bash
streamlit run app.py
```

---

## Business Insights

* Supports automated trading strategies
* Helps investors manage portfolio risk
* Enables long-term investment planning
* Useful for financial forecasting
* Helps analyze competitor stock trends
* Can be extended with sentiment analysis and macroeconomic indicators

---

## Future Enhancements

* Add news sentiment analysis
* Integrate social media trend analysis
* Add Transformer-based models
* Deploy using cloud platforms
* Real-time stock prediction API integration

---

## Conclusion

This project demonstrates how Deep Learning models can effectively predict stock market trends using historical time-series data. Among all models, LSTM achieved the best performance due to its ability to capture long-term sequential dependencies. The project provides valuable insights for stock forecasting, financial analysis, and investment decision-making.

---
