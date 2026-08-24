import tensorflow as tf
from tensorflow import keras #type: ignore

from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D #type: ignore

from tensorflow.keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(13,)))
model.add(Dense(15, activation='tanh'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2,verbose=1)

y_pred = model.predict(X_test)
print("Predicted value:", y_pred[1:10])
print("Actual value:", y_test[1:10])

model.save('boston_house_model.keras')

m1 = keras.models.load_model('boston_house_model.keras')

print(m1.predict(X_test[0].reshape(1, 13)))