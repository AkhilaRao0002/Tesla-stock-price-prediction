
# ============================================
# STREAMLIT STOCK PRICE PREDICTION APP
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide"
)

st.title("📈 Stock Price Prediction App")

# ============================================
# LOAD MODEL & SCALER
# ============================================

model = load_model("lstm_stock_prediction_model (1).h5")

target_scaler = joblib.load(
    "target_scaler (3).pkl"
)

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv(
    "TSLA.csv"
)

# Convert Date column
data['Date'] = pd.to_datetime(
    data['Date']
)

# Set index
data.set_index(
    'Date',
    inplace=True
)

# ============================================
# PREPARE TARGET DATA
# ============================================

target_data = data[['Adj Close']]

scaled_target = target_scaler.transform(
    target_data
)

sequence_length = 60

# ============================================
# FUNCTION
# ============================================

def predict_stock_behaviour(date_input):

    future_date = pd.to_datetime(
        date_input
    )

    last_date = data.index[-1]

    # ========================================
    # EXISTING / PAST DATE
    # ========================================

    if future_date <= last_date:

        idx = data.index.get_loc(
            future_date
        )

        if idx < sequence_length:

            return None

        past_60_days = scaled_target[
            idx-sequence_length:idx
        ]

        temp_input = list(
            past_60_days
        )

        predictions = []

        total_days = 10

        for i in range(total_days):

            X_input = np.array(
                temp_input[-60:]
            )

            X_input = X_input.reshape(
                1,
                60,
                1
            )

            predicted = model.predict(
                X_input,
                verbose=0
            )

            predictions.append(
                predicted[0][0]
            )

            temp_input.append(
                predicted[0]
            )

        predictions = target_scaler.inverse_transform(

            np.array(
                predictions
            ).reshape(-1,1)

        )

        entered_day_prediction = predictions[
            0
        ][0]

        trend_5_days = predictions[
            :5
        ]

        trend_10_days = predictions[
            :10
        ]

    # ========================================
    # FUTURE DATE
    # ========================================

    else:

        days_ahead = (
            future_date - last_date
        ).days

        last_60_days = scaled_target[-60:]

        temp_input = list(
            last_60_days
        )

        predictions = []

        total_days = days_ahead + 10

        for i in range(total_days):

            X_input = np.array(
                temp_input[-60:]
            )

            X_input = X_input.reshape(
                1,
                60,
                1
            )

            predicted = model.predict(
                X_input,
                verbose=0
            )

            predictions.append(
                predicted[0][0]
            )

            temp_input.append(
                predicted[0]
            )

        predictions = target_scaler.inverse_transform(

            np.array(
                predictions
            ).reshape(-1,1)

        )

        entered_day_prediction = predictions[
            days_ahead - 1
        ][0]

        trend_5_days = predictions[
            days_ahead - 1 :
            days_ahead + 4
        ]

        trend_10_days = predictions[
            days_ahead - 1 :
            days_ahead + 9
        ]

    return (

        entered_day_prediction,

        trend_5_days,

        trend_10_days
    )

# ============================================
# USER INPUT
# ============================================

selected_date = st.date_input(
    "Select Prediction Date"
)

# ============================================
# BUTTON
# ============================================

if st.button("Predict Stock Price"):

    result = predict_stock_behaviour(
        selected_date
    )

    if result is None:

        st.error(
            "Not enough previous data"
        )

    else:

        entered_price, trend_5, trend_10 = result

        # ====================================
        # ENTERED DATE PREDICTION
        # ====================================

        st.subheader(
            "📌 Predicted Price"
        )

        st.success(
            f"₹ {round(entered_price,2)}"
        )

        # ====================================
        # DROPDOWN FOR TREND SELECTION
        # ====================================

        trend_option = st.selectbox(

            "Select Trend Duration",

            (
                "5 Day Trend",
                "10 Day Trend"
            )
        )

        # ====================================
        # 5 DAY TREND
        # ====================================

        if trend_option == "5 Day Trend":

            st.subheader(
                "📈 5 Day Trend"
            )

            trend_df = pd.DataFrame({

                "Day": [
                    f"Day {i}"
                    for i in range(
                        1,
                        6
                    )
                ],

                "Predicted Price": [

                    round(
                        value[0],
                        2
                    )

                    for value in trend_5
                ]
            })

            st.dataframe(
                trend_df,
                use_container_width=True
            )

            st.line_chart(
                trend_df.set_index(
                    "Day"
                )
            )

        # ====================================
        # 10 DAY TREND
        # ====================================

        else:

            st.subheader(
                "📈 10 Day Trend"
            )

            trend_df = pd.DataFrame({

                "Day": [
                    f"Day {i}"
                    for i in range(
                        1,
                        11
                    )
                ],

                "Predicted Price": [

                    round(
                        value[0],
                        2
                    )

                    for value in trend_10
                ]
            })

            st.dataframe(
                trend_df,
                use_container_width=True
            )

            st.line_chart(
                trend_df.set_index(
                    "Day"
                )
            )
