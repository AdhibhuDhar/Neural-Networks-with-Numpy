import numpy as np
import matplotlib.pyplot as plt
from cd import create_data
np.random.seed(0)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from layers import Layer_Dense
from activations import Activation_ReLU,Activation_Softmax
from losses import Loss_CategoricalCrossEntropy,Activation_Softmax_Loss_CategoricalCrossEntropy
from data import Dataset,DataLoader,load_dataset,train_test_split
from model import Model
from optimizers import Optimizer_SGD,Optimizer_Adam
from dropout import Layer_Dropout
from batchnorm import Layer_BatchNorm
from gradcheck import gradient_check
from tensor import Tensor

dataset_path=input("Enter dataset path:")
label_column=input("Enter label column name:")
dataset=Dataset(dataset_path,label_column)
X_train,X_test,y_train,y_test = train_test_split(dataset.X,dataset.y)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Convert one-hot labels to class indices
if len(y_train.shape) > 1:
    y_train = np.argmax(y_train, axis=1)

if len(y_test.shape) > 1:
    y_test = np.argmax(y_test, axis=1)

train_dataset=Dataset(X=X_train,y=y_train)
test_dataset=Dataset(X=X_test,y=y_test)
print("Train size:",train_dataset.n_samples)
print("Test size",test_dataset.n_samples)
print("X train:", X_train.shape)
print("y train:", y_train.shape)
num_features=train_dataset.n_features
num_classes=len(np.unique(y_train))

model=Model()
#testing grad check
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001)
)
gradient_check(model,X_train[:5],y_train[:5])
model.train(train_dataset,epochs=10000,batch_size=32)
model.evaluate(test_dataset)
model.add(Layer_Dense(num_features,64))
model.add(Layer_BatchNorm(64))
model.add(Activation_ReLU())#ReLU provides active neuorns
model.add(Layer_Dropout(0.2))#Dropout randomly kills neurons for redundant representation,prevents overfitting
model.add(Layer_Dense(64,num_classes))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001)
)


model.train(train_dataset,epochs=10000,batch_size=32)
model.evaluate(test_dataset)



