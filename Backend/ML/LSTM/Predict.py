import numpy as np
import tensorflow as tf
import pickle

# Load the scaler object from the pickle file
with open('Backend/ML/LSTM/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create an input array
x = np.array([[25,140,100,122.39999999999999,36.666666666666664,80]])

# Scale the input data using the loaded scaler object
x = scaler.transform(x)
# Reshape the input array to match the input shape of the TFLite model
x = x.astype(np.float32) 
x = x.reshape((x.shape[0], 1, x.shape[1]))

print(f'Input tensor before invoking TensorFlow Lite model: {x}')

interpreter.set_tensor(input_details[0]['index'], x)

interpreter.invoke()

y_pred = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(y_pred[0])
print(f'Input: {x.tolist()[0]}, Predicted Class: {predicted_class}')
if predicted_class==1:
    print("High Risk")
else:
    print("Low risk")


print(f'Input tensor after invoking TensorFlow Lite model: {interpreter.get_tensor(input_details[0]["index"]).tolist()}')
