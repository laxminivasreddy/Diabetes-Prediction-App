# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
from tensorflow import keras
df = pd.read_csv("diabetes.csv")
df.head()
df.shape # no of rows and columns
#basic info of dataset
df.info()
# Divide the dataset into dependent and independent variables
X = df.drop (columns = "Outcome")
y = df["Outcome"]
X.head()
# normalize the feature
scaler = StandardScaler()
X = scaler.fit_transform(X)
#save the scaler
joblib.dump(scaler,"Scaler.pkl")

['Scaler.pkl']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#build neural network
model = keras.Sequential([
    keras.layers.Dense(16,activation = "relu" , input_shape = (X.shape[1],)),#input layer
    keras.layers.Dense(8, activation = "relu"),#hidden layer
    keras.layers.Dense(1, activation = "sigmoid")#output layer
])
model.summary()
#compiling the model
model.compile(optimizer = "adam", loss = "binary_crossentropy")
# train the model
model.fit(X_train, y_train, epochs = 50,batch_size = 10,validation_data = (X_test,y_test))
#making the predictions and converting it into integer variable
y_predict = model.predict(X_test)
y_predict = (y_predict > 0.5).astype("int32")
# calculate the performance metrics
accuracy = accuracy_score(y_predict,y_test)
recall = recall_score(y_predict, y_test)
precision = precision_score(y_predict, y_test)
f1 = f1_score (y_predict,y_test)
cm = confusion_matrix (y_predict,y_test)
cr = classification_report (y_predict,y_test)
#print all the performance metrices
print("the accuracy of our diabetic prediction model is ",accuracy)
print("the precision of our model",precision)
print("the recall of our model is ",recall)
print("the f1 score of our model is ",f1)
print("the confusion matrix of our model is ",cm)
print("**The classification report**")
print(cr)
model.save("Diabetes_model.h5")
print("the model has been saved")