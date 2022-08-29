# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

*A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Deep learning is the development of deep learning algorithms that can be used to train and predict output from complex data.The word “deep” in Deep Learning refers to the number of hidden layers i.e. depth of the neural network. Essentially, every neural network with more than three layers, that is, including the Input Layer and Output Layer can be considered a Deep Learning Model.TensorFlow, an open-source software library for machine learning, offers a robust framework for implementing neural network regression models.The Reluactivation function helps neural networks form deep learning models. Due to the vanishing gradient issues in different layers, you cannot use the hyperbolic tangent and sigmoid activation. You can overcome the gradient problems through the Relu activation function.


## Neural Network Model
![DL-EXP1](https://user-images.githubusercontent.com/75235132/187116816-d04530e6-8b64-41e9-ab1e-12f5612d4130.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model with hidden layer 1 - 7 neurons , hidden layer 2 - 3 neurons and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset1.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(7,activation='relu'),
    Dense(3,activation='relu'),
    Dense(1)
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[25]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)

```

## Dataset Information

![DL-EXP1 B](https://user-images.githubusercontent.com/75235132/187117590-7aad1278-3f0b-43a3-b850-f876a9700827.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![DL-EXP1 c](https://user-images.githubusercontent.com/75235132/187117649-0ac9ff28-bb98-4066-9d56-067d1b237bd7.png)

### Test Data Root Mean Squared Error

![DL-EXP1 d](https://user-images.githubusercontent.com/75235132/187117711-6eea786c-c39f-45d1-9476-36cb510a5fd7.png)

### New Sample Data Prediction

![DL-EXP1 e](https://user-images.githubusercontent.com/75235132/187117736-7f53098f-515f-4688-b9ca-56739f5c6afb.png)

## RESULT

Thus,the neural network regression model for the given dataset is developed.
