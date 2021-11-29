# package imports
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import linear_model
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

def validate_result(model, model_name, validation_X, validation_y):
    predicted = model.predict(validation_X)
    RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))

    R2_score = r2_score(validation_y, predicted)

    # plt.plot(validation_y.index, predicted, 'r', label='Predict')
    # plt.plot(validation_y.index, validation_y, 'b', label='Actual')
    # plt.ylabel('Price')
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # plt.title(model_name + ' Predict vs Actual')
    # plt.legend(loc='upper right')
    # plt.show()


def stackedLSTM(csv, feature_columns):
    print("Stacked LSTM")
    df_final = pd.read_csv(csv, na_values=[
        'null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

    # df_final.head()

    # df_final.shape

    # df_final.describe()

    # df_final.isnull().valueTs.any()

    test = df_final
    # Target column
    target_adj_close = pd.DataFrame(test['Close'])

    # selecting Feature Columns
    # feature_columns = ['Open', 'High', 'Low', 'Volume']

    """# Normalizing the data"""

    scaler = MinMaxScaler()
    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(
        columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
    feature_minmax_transform.head()

    # Shift target array because we want to predict the n + 1 day value

    target_adj_close = target_adj_close.shift(-1)
    validation_y = target_adj_close[-90:-1]
    target_adj_close = target_adj_close[:-90]

    # Taking last 90 rows of data to be validation set
    validation_X = feature_minmax_transform[-90:-1]
    feature_minmax_transform = feature_minmax_transform[:-90]

    """# Train test Split using Timeseriessplit"""

    ts_split = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(
            train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(
        ), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    # X_train.shape

    # X_test.shape

    # y_train.shape

    # y_test.shape

    """# Benchmark Model"""

    dt = DecisionTreeRegressor(random_state=0)

    benchmark_dt = dt.fit(X_train, y_train)

    validate_result(benchmark_dt, 'Decision Tree Regression',
                    validation_X, validation_y)

    """# Process the data for Stacked LSTM"""

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    """# Model building : Stacked LSTM"""

    K.clear_session()
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(
        1, X_train.shape[1]), activation='relu', return_sequences=True))
    model_lstm.add(LSTM(16, activation='relu', return_sequences=False))
    model_lstm.add(Dense(1, activation='relu'))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history_model_lstm = model_lstm.fit(
        X_tr_t, y_train, epochs=150, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

    """# Evaluation of Model"""

    y_pred_test_lstm = model_lstm.predict(X_tst_t)
    y_train_pred_lstm = model_lstm.predict(X_tr_t)

    r2_train = r2_score(y_train, y_train_pred_lstm)

    r2_test = r2_score(y_test, y_pred_test_lstm)

    """## Predictions made by LSTM"""

    score_lstm = model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

    return {
        "score_lstm": score_lstm,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }


def biDirectionalLSTM(csv, feature_columns):
    print("BiDirectional LSTM")

    df_final = pd.read_csv(csv, na_values=[
                           'null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

    # df_final.head()

    # df_final.shape

    # df_final.describe()

    # df_final.isnull().values.any()

    test = df_final
    # Target column
    target_adj_close = pd.DataFrame(test['Close'])

    # selecting Feature Columns
    # feature_columns = ['Open', 'High', 'Low', 'Volume']

    """# Normalizing the data"""

    scaler = MinMaxScaler()
    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(
        columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
    feature_minmax_transform.head()

    # Shift target array because we want to predict the n + 1 day value

    target_adj_close = target_adj_close.shift(-1)
    validation_y = target_adj_close[-90:-1]
    target_adj_close = target_adj_close[:-90]

    # Taking last 90 rows of data to be validation set
    validation_X = feature_minmax_transform[-90:-1]
    feature_minmax_transform = feature_minmax_transform[:-90]

    """# Train test Split using Timeseriessplit"""

    ts_split = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(
            train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(
        ), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    # X_train.shape

    # X_test.shape

    # y_train.shape

    # y_test.shape

    dt = DecisionTreeRegressor(random_state=0)

    benchmark_dt = dt.fit(X_train, y_train)

    validate_result(benchmark_dt, 'Decision Tree Regression',
                    validation_X, validation_y)

    """# Process the data for Bidirectional LSTM"""

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    """# Model building : Bidirectional LSTM"""

    K.clear_session()
    model_lstm = Sequential()
    model_lstm.add(Bidirectional(LSTM(16, input_shape=(
        1, X_train.shape[1]), activation='relu', return_sequences=False)))
    model_lstm.add(Dense(1, activation='relu'))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history_model_lstm = model_lstm.fit(
        X_tr_t, y_train, epochs=150, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

    """# Evaluation of Model"""

    y_pred_test_lstm = model_lstm.predict(X_tst_t)
    y_train_pred_lstm = model_lstm.predict(X_tr_t)

    r2_train = r2_score(y_train, y_train_pred_lstm)

    r2_test = r2_score(y_test, y_pred_test_lstm)

    """## Predictions made by LSTM"""

    score_lstm = model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

    return {
        "score_lstm": score_lstm,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }


def classicLSTM(csv, feature_columns):
    print("Classic LSTM")

    df_final = pd.read_csv(csv, na_values=[
        'null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

    # df_final.head()

    # df_final.shape

    # df_final.describe()

    # df_final.isnull().values.any()

    test = df_final
    # Target column
    target_adj_close = pd.DataFrame(test['Close'])

    # selecting Feature Columns
    # feature_columns = ['Open', 'High', 'Low', 'Volume']

    """# Normalizing the data"""

    scaler = MinMaxScaler()
    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(
        columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
    feature_minmax_transform.head()

    # Shift target array because we want to predict the n + 1 day value

    target_adj_close = target_adj_close.shift(-1)
    validation_y = target_adj_close[-90:-1]
    target_adj_close = target_adj_close[:-90]

    # Taking last 90 rows of data to be validation set
    validation_X = feature_minmax_transform[-90:-1]
    feature_minmax_transform = feature_minmax_transform[:-90]

    """# Train test Split using Timeseriessplit"""

    ts_split = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(
            train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(
        ), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    # X_train.shape

    # X_test.shape

    # y_train.shape

    # y_test.shape

    """# Benchmark Model"""

    dt = DecisionTreeRegressor(random_state=0)

    benchmark_dt = dt.fit(X_train, y_train)

    validate_result(benchmark_dt, 'Decision Tree Regression',
                    validation_X, validation_y)

    """# Process the data for Classic LSTM"""

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    """# Model building : Classic LSTM"""

    K.clear_session()
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(
        1, X_train.shape[1]), activation='relu', return_sequences=False))
    model_lstm.add(Dense(1, activation='relu'))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history_model_lstm = model_lstm.fit(
        X_tr_t, y_train, epochs=150, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

    """# Evaluation of Model"""

    y_pred_test_lstm = model_lstm.predict(X_tst_t)
    y_train_pred_lstm = model_lstm.predict(X_tr_t)

    r2_train = r2_score(y_train, y_train_pred_lstm)

    r2_test = r2_score(y_test, y_pred_test_lstm)

    """## Predictions made by LSTM"""

    score_lstm = model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

    return {
        "score_lstm": score_lstm,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }
