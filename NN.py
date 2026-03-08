import numpy as np
import matplotlib.pyplot as plt
from cd import create_data
np.random.seed(0)
#X,y=create_data(100,3)#100 feature set of 3 classes
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities
class Loss:
    def calculate(self,output,y): #y is target values
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1: #passed scalar values
            correct_confidence=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2: #one-hot encoded
            correct_confidence=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood=-np.log(correct_confidence)
        return negative_log_likelihood 
X,y=create_data(100,3)
dense_1=Layer_Dense(2,3)
activation_1=Activation_ReLU()
dense_2=Layer_Dense(3,3)
activation_2=Activation_Softmax()
dense_1.forward(X)
activation_1.forward(dense_1.output)
dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)
print(activation_2.output[:5])
loss_function=Loss_CategoricalCrossEntropy()
loss=loss_function.calculate(activation_2.output,y)
print("Loss:",loss)



