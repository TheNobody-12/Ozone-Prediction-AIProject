from keras.layers import Bidirectional, LSTM, Dense,LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

def build_lstm_model(input_shape, num_lstm_units, num_hidden_layers, num_units_hidden_layers, activation_function, dropout_rate):
    model = Sequential()
    model.add(LSTM(num_lstm_units, activation='relu', input_shape=input_shape))
    for i in range(num_hidden_layers):
        model.add(Dense(num_units_hidden_layers))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae'])
    return model

def build_bidirectional_lstm_model(input_shape, num_lstm_units, num_dense_units, dense_activation_function, optimizer):
    model = Sequential()
    model.add(Bidirectional(LSTM(num_lstm_units, activation='relu', input_shape=input_shape)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(num_dense_units, activation=dense_activation_function))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model




