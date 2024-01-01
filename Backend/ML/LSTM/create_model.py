import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow.lite as lite
import pickle
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Backend/ML/LSTM/New_Maternal.csv')

x = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].values
y = data['RiskLevel'].values

scaler=MinMaxScaler()
x=scaler.fit_transform(x)
# Save the scaler object to a file
with open('Backend/ML/LSTM/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(y_train)

x_train=x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
x_test=x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
print(x_test.shape[0],1,x_test.shape[1])
# Get the number of unique values in y
num_classes = len(np.unique(y))

model = Sequential()
model.add(LSTM(units=64, input_shape=(1, x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dense(units=num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

_, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)

model.save("Backend/ML/LSTM/model.h5")
m = keras.models.load_model("Backend/ML/LSTM/model.h5")
converter = lite.TFLiteConverter.from_keras_model(m)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
with open('Backend/ML/LSTM/model.tflite', 'wb') as f:
    f.write(tflite_model)
