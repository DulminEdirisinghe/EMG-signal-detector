import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# feature extraction function
def extract_features(emg_data):
    # emg_data  (n_samples, 3 channels)
    #  features: mean, variance, and RMS for each channel
    mean_features = np.mean(emg_data, axis=1)
    var_features = np.var(emg_data, axis=1)
    rms_features = np.sqrt(np.mean(emg_data**2, axis=1))
    
    features = np.vstack((mean_features, var_features, rms_features)).T
    return features

# load data
n_samples = 1000  # Number of samples
emg_data = []  # Training data
labels = []  # Training labels

# Feature extraction
features = extract_features(emg_data)

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
