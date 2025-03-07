import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
exit()


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# Step 2: Reading Data
df1 = pd.read_csv('422 datasets/madrid_2001.csv')
df2 = pd.read_csv('422 datasets/madrid_2002.csv')
df3 = pd.read_csv('422 datasets/madrid_2003.csv')
df4 = pd.read_csv('422 datasets/madrid_2004.csv')
df5 = pd.read_csv('422 datasets/madrid_2005.csv')
df6 = pd.read_csv('422 datasets/madrid_2006.csv')
df7 = pd.read_csv('422 datasets/madrid_2007.csv')
df8 = pd.read_csv('422 datasets/madrid_2008.csv')
df9 = pd.read_csv('422 datasets/madrid_2009.csv')
df10 = pd.read_csv('422 datasets/madrid_2010.csv')
df11 = pd.read_csv('422 datasets/madrid_2011.csv')
df12 = pd.read_csv('422 datasets/madrid_2012.csv')
df13 = pd.read_csv('422 datasets/madrid_2013.csv')
df14 = pd.read_csv('422 datasets/madrid_2014.csv')
df15 = pd.read_csv('422 datasets/madrid_2015.csv')
df16 = pd.read_csv('422 datasets/madrid_2016.csv')
df17 = pd.read_csv('422 datasets/madrid_2017.csv')
df18 = pd.read_csv('422 datasets/madrid_2018.csv')


# Concatenate all dataframes
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18], axis=0)


# Reset the index of the concatenated dataframe
df = df.reset_index(drop=True)


# Convert datetime columns to numerical representations (if any)
for col in df.select_dtypes(include=[object]):
   try:
       df[col] = pd.to_datetime(df[col], errors='raise')  # Attempt to convert any object column to datetime
       df[col] = df[col].astype(int) / 10**9  # Convert datetime to timestamp (seconds since epoch)
   except Exception as e:
       print(f"Column '{col}' could not be converted to datetime: {e}")


# Feature Engineering: Extract useful datetime features (optional but often helpful)
for col in df.select_dtypes(include=[int, float]):
   df[col] = df[col].fillna(df[col].mean())  # Handle missing values by filling with the mean


# Create a copy of the initial dataframe for reference
df_init = df.copy()


# Assuming the last column is the target and others are features
y = df.iloc[:, -1]
X = df.iloc[:, :-1]


# Encoding the target variable (if it's categorical)
le = LabelEncoder()
y = le.fit_transform(y)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define a simple neural network model (KNN-like using dense layers)
model = Sequential([
   Input(shape=(X_train.shape[1],)),
   Dense(64, activation='relu'),
   Dropout(0.2),  # Dropout for regularization to prevent overfitting
   Dense(32, activation='relu'),
   Dropout(0.2),  # Dropout for regularization
   Dense(len(np.unique(y)), activation='softmax')  # Multi-class classification output
])


# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')








