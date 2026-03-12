import numpy as np
import matplotlib.pyplot as plt
from cd import create_data
np.random.seed(0)
import matplotlib.pyplot as plt

class Optimizer_SGD: #drunk downhill
    def __init__(self,learning_rate=1.0,momentum=0.0): #momentum=how much memory we keep(v_t = β * v_(t-1) + (1 - β) * gradient)
        #weight = weight - learning_rate * v_t
        #B is momentum factor,how much momentum to keep
        #momentum helps to escape local minima,smooths updates,accelerates convergence or training,keeps moving when gradients tiny
        #velocity is memory of previous slope
        self.learning_rate=learning_rate
        self.momentum=momentum
    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,"weight_momentums"):#does obh already have this attr
                layer.weight_momentums=np.zeros_like(layer.weights)#dynamically add atr
                layer.bias_momentums=np.zeros_like(layer.biases)
                weight_updates=(self.momentum*layer.weight_momentums-self.learning_rate*layer.dweights)
                layer.weight_momentums=weight_updates
                bias_updates=(self.momentum*layer.bias_momentums-self.learning_rate*layer.dbiases)
                layer.bias_momentums=bias_updates
            else:
                weight_updates=-self.learning_rate*layer.dweights
                bias_updates=-self.learning_rate*layer.dbiases
            layer.weights+=weight_updates
            layer.biases+=bias_updates

        
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(inputs,self.weights)+self.biases
    def backward(self,dvalues):
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs=np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.output<=0]=0

class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities

class Loss:
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidence=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidence=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood=-np.log(correct_confidence)
        return negative_log_likelihood 

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples


X,y=create_data(100,3)
dense_1=Layer_Dense(2,64)
activation_1=Activation_ReLU()
dense_2=Layer_Dense(64,3)
activation_2=Activation_Softmax()
loss_function=Loss_CategoricalCrossEntropy()
loss_activation=Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer=Optimizer_SGD(learning_rate=0.1,momentum=0.9)
for epoch in range(10001):
    dense_1.forward(X)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)
    loss=loss_function.calculate(activation_2.output,y)
    predictions=np.argmax(activation_2.output,axis=1)
    accuracy=np.mean(predictions==y)
    loss_activation.backward(activation_2.output,y)
    dense_2.backward(loss_activation.dinputs)
    activation_1.backward(dense_2.dinputs)
    dense_1.backward(activation_1.dinputs)
    optimizer.update_params(dense_1)
    optimizer.update_params(dense_2)
    if epoch%1000==0:
        print("epoch:",epoch,"loss:",loss,"accuracy:",accuracy)
x_min,x_max=X[:,0].min() -0.5,X[:,0].max() +0.5
y_min,y_max=X[:,1].min() -0.5,X[:,1].max() +0.5
h=0.01 #step
xx,yy=np.meshgrid(
    np.arange(x_min,x_max,h),
    np.arange(y_min,y_max,h)
)
#flatten(sample,features)
grid=np.c_[xx.ravel(),yy.ravel()]
dense_1.forward(grid)
activation_1.forward(dense_1.output)
dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)
#predicted class
predictions=np.argmax(activation_2.output,axis=1)
#reshape tp grid
predictions=predictions.reshape(xx.shape)

#plot
plt.contourf(xx,yy,predictions,cmap="brg",alpha=0.5)
plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
plt.show()      


