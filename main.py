# ---- Created by Adam von Kraemer ----

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import Dense, concatenate
from sklearn.metrics import mean_squared_error

# Load the data
sp500_csv = pd.read_csv(r'GSPC.csv')

# Define which dates and values to include in the data
date = np.asarray(sp500_csv['Date'].values)
close = np.asarray(sp500_csv['Close'].values)
volume = np.asarray(sp500_csv['Volume'].values)
high = np.asarray(sp500_csv['High'].values)
low = np.asarray(sp500_csv['Low'].values)
days_per_sample = 250
days_to_predict = 1
n_features = 4

# Define normalization functions for normalizing the data to (0, 1)
close_max = max(close) + 1
close_min = min(close) - 1
volume_max = max(volume) + 1
volume_min = min(volume) + 1
high_max = max(high) + 1
high_min = min(high) - 1
low_max = max(low) + 1
low_min = min(low) - 1


def normalizeIndex(data, maxvalue, minvalue):
    return (float(data) - minvalue) / (maxvalue - minvalue)


def deNormalizeIndex(data, maxvalue, minvalue):
    return float(data) * (maxvalue - minvalue) + minvalue


# Generate time series sequential data
def generateSequentialData(close, volume):
    data = list(list())
    data_validation = list(list())
    for r in range(0, len(close) - days_per_sample * 2):
        data.append([])
        for c in range(0, days_per_sample):
            data[r].append([normalizeIndex(close[c + r], close_max, close_min),
                            normalizeIndex(volume[c + r], volume_max, volume_min),
                            normalizeIndex(high[c + r], high_max, high_min),
                            normalizeIndex(low[c + r], low_max, low_min)])
    for r in range(len(close) - days_per_sample * 2, len(close) - days_per_sample):
        data_validation.append([])
        for c in range(0, days_per_sample):
            data_validation[r - len(close) + days_per_sample * 2].append(
                [normalizeIndex(close[c + r], close_max, close_min),
                 normalizeIndex(volume[c + r], volume_max, volume_min),
                 normalizeIndex(high[c + r], high_max, high_min),
                 normalizeIndex(low[c + r], low_max, low_min)])
    return np.asarray(data), np.asarray(data_validation)


# Function for calculating correct number of guesses (up or down the next day) for a predicted dataset
def calculateCorrectGuesses(predicted, actual):
    predictedchange = list()
    actualchange = list()
    correctguess = 0
    incorrectguess = 0
    for i in range(1, days_per_sample):
        predictedchange.append(predicted[i] - predicted[i - 1])
        actualchange.append(actual[i] - actual[i - 1])
        if (predictedchange[i - 1] < 0 and actualchange[i - 1] < 0):
            correctguess += 1
        elif (predictedchange[i - 1] > 0 and actualchange[i - 1] > 0):
            correctguess += 1
        else:
            incorrectguess += 1
    return correctguess


# Define train and test datasets
data, data_validation = generateSequentialData(close, volume)
n_train_samples = int(len(data) * 0.8)
np.random.seed(1)
np.random.shuffle(data)
train = data[:n_train_samples, :]
test = data[n_train_samples:, :]
train_X = train[:, :(days_per_sample - days_to_predict)]
train_y = train[:, (days_per_sample - days_to_predict):, 0]
test_X = test[:, :(days_per_sample - days_to_predict)]
test_y = test[:, (days_per_sample - days_to_predict):, 0]
data_validation_X = data_validation[:, :(days_per_sample - days_to_predict)]
data_validation_y = data_validation[:, (days_per_sample - days_to_predict):, 0]

# Reshape data for model
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1])
test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])
data_validation_y = data_validation_y.reshape(data_validation_y.shape[0], data_validation_y.shape[1])

# Separate the datasets of each feature
train_X_close = train_X[:, :, 0]
test_X_close = test_X[:, :, 0]
data_validation_X_close = data_validation_X[:, :, 0]
train_X_volume = train_X[:, :, 1]
test_X_volume = test_X[:, :, 1]
data_validation_X_volume = data_validation_X[:, :, 1]
train_X_high = train_X[:, :, 2]
test_X_high = test_X[:, :, 2]
data_validation_X_high = data_validation_X[:, :, 2]
train_X_low = train_X[:, :, 3]
test_X_low = test_X[:, :, 3]
data_validation_X_low = data_validation_X[:, :, 3]

# Define the model
n_steps = days_per_sample - days_to_predict
hidden_layer_size = 100
n_epochs = 1000
input_shape = [Input(shape=(n_steps,)), Input(shape=(n_steps,)), Input(shape=(n_steps,)), Input(shape=(n_steps,))]
dense = [Dense(hidden_layer_size)(input_shape[0]), Dense(hidden_layer_size)(input_shape[0]),
         Dense(hidden_layer_size)(input_shape[0]), Dense(hidden_layer_size)(input_shape[0])]
merge = concatenate(dense)
output = Dense(days_to_predict)(merge)
model = Model(inputs=input_shape, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit([train_X_close, train_X_volume, train_X_high, train_X_low], train_y, epochs=n_epochs, batch_size=32,
                    validation_data=([test_X_close, test_X_volume, test_X_high, test_X_low], test_y), shuffle=False)

# Plot the error curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

# Make a prediction on a dataset with new data
yhat = model.predict([data_validation_X_close, data_validation_X_volume, data_validation_X_high, data_validation_X_low])
correctguesses = calculateCorrectGuesses(yhat, data_validation_y)

# Denormalize the data
for i in range(0, len(data_validation_y)):
    data_validation_y[i] = deNormalizeIndex(data_validation_y[i], close_max, close_min)
for i in range(0, len(yhat)):
    yhat[i] = deNormalizeIndex(yhat[i], close_max, close_min)

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(data_validation_y, yhat))
print("RMSE: ", rmse)

# Calculate and print correct number of guesses
correctguesses = calculateCorrectGuesses(yhat, data_validation_y)
print("Correct guesses: ", (correctguesses / len(data_validation_y)) * 100, "%")

# Plot curve
x = [x for x in range(len(data_validation_y))]
plt.plot(x, data_validation_y, marker='.', label="actual")
plt.plot(x, yhat, marker='.', label="prediction")
plt.legend(fontsize=14)
plt.show()
