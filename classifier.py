import serial
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib


serial_port = '/dev/ttyUSB0'  
baud_rate = 115200

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Load  SVM model and the fitted scaler
clf = joblib.load('svm_emg_model.pkl')  
scaler = joblib.load('scaler.pkl')      

# Feature extraction function
def extract_features(emg_data):
    mean_features = np.mean(emg_data, axis=1)
    var_features = np.var(emg_data, axis=1)
    rms_features = np.sqrt(np.mean(emg_data**2, axis=1))
    features = np.vstack((mean_features, var_features, rms_features)).T
    return features

# Real-time classification function
def real_time_classification(clf, scaler):
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            data = tuple(map(float, line.split(',')))          
            time_stamp, ch1, ch2, ch3 = data           
            emg_data = np.array([[ch1, ch2, ch3]])
            features = extract_features(emg_data)
            features = scaler.transform(features)
            category = clf.predict(features)
            print(f"Time: {time_stamp}, Classified Category: {category[0]}")

        except ValueError:
            print("Received invalid data. Skipping...")
            continue

#  classification loop
real_time_classification(clf, scaler)