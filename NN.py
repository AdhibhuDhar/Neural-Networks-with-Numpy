import numpy as np
import matplotlib.pyplot as plt
from cd import create_data
np.random.seed(0)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Optimizer_Adam: #decay:reduces LR over time,epsilom:prevents div by 0
    #beta_1=how much paast gradients influence direction or curr grad
    #beta_2=how much pass gradient squared influence step size or scaling
    #we add bias to momentum and cache to counteract their initial underestimation->bias correction
    #without correction,optimizer takes very slow steps at the beginning,slowing down initial convergence
    #we need a global learning rate to measure how big a step to take overall
    def __init__(self,learning_rate=0.001,decay=0.0,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
    def update_params(self,layer): #momentum decides where to go and cache decides how big the step should be
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1.0/(1.0+self.decay*self.iterations))
        if not hasattr(layer,"weight_cache"): #weight_cache is memory of past sqrd grad
            #first time optimiser sees layer,create cache arrays
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights) #momentum and cache have same shape as weights
            layer.bias_momentums=np.zeros_like(layer.biases)
            layer.bias_cache=np.zeros_like(layer.biases)
        #update momentum with current gradients
        layer.weight_momentums=self.beta_1*layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1*layer.bias_momentums+(1-self.beta_1)*layer.dbiases #for smooth gradients
        #corrected momentum due to underestimation in initial iterations
        weight_momentums_corrected=layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums/(1-self.beta_1**(self.iterations+1))
        #Second moment->how much past squared gradients influence step size
        layer.weight_cache=self.beta_2*layer.weight_cache+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache=self.beta_2*layer.bias_cache+(1-self.beta_2)*layer.dbiases**2
        #bias correct variance
        weight_cache_corrected=layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected=layer.bias_cache/(1-self.beta_2**(self.iterations+1))
        #final update
        layer.weights+=-self.current_learning_rate*weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        #step size=momentum/variance or sqrt(cache)+epsilon->sqrt cache brings sqrd grad back to same scale as grad
        #so step becomes gradient_mean/gradient_variance->Also epsilon prevents sqrt(cache) to become too small which explodes step size
        #big gradient=smaller step,small gradient=bigger step
        layer.biases+=-self.current_learning_rate*bias_momentums_corrected/np.sqrt(bias_cache_corrected+self.epsilon)
    def post_update_params(self):
        self.iterations+=1
        #Adam=remember dir->measure terrain slope->adjust step size->correct bias->move effiiciently towards minima


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
        data_loss=np.mean(negative_log_likelihood)
        return data_loss

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples

def load_dataset(path,label_column):
    data=pd.read_csv(path)
    X=data.drop(label_column,axis=1)
    y=data[label_column]
    X=pd.get_dummies(X)
    X=(X-X.mean())/(X.std()+1e-7) #standardize features
    X=X.values.astype(float)
    y=y.values
    encoder=LabelEncoder()
    y=encoder.fit_transform(y)
    return X,y
class Dataset:
    def __init__(self,path,label_column):
        data=pd.read_csv(path)
        self.X,self.y=load_dataset(path,label_column)
        self.n_samples=self.X.shape[0]
        self.n_features=self.X.shape[1]
class DataLoader:
    def __init__(self,dataset,batch_size=32,shuffle=True):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
    def __iter__(self):
        self.indices=np.arange(self.dataset.n_samples) #this creates an array of indices from 0 to n_samples-1
        if self.shuffle:
            np.random.shuffle(self.indices)#shuffling the indices to ensure random sampling of data in each epoch
            self.current=0
            return self
    def __next__(self): #fetch next batch
        if self.current>=self.dataset.n_samples:
            raise StopIteration
        start=self.current
        end=self.current+self.batch_size #calculate end index for batch
        batch_indices=self.indices[start:end] #get indices for current batch
        X_batch=self.dataset.X[batch_indices] #fetch batch data using indices
        y_batch=self.dataset.y[batch_indices]
        self.current+=self.batch_size #update current index for next batch
        return X_batch,y_batch
dataset_path=input("Enter dataset path:")
label_column=input("Enter label column name:")
X,y=load_dataset(dataset_path,label_column)
dataset=Dataset(dataset_path,label_column)
loader=DataLoader(dataset,batch_size=32,shuffle=True)
#X,y=create_data(100,3)
input_size=X.shape[1]
dense_1=Layer_Dense(input_size,64)
activation_1=Activation_ReLU()
num_classes=len(np.unique(y))
dense_2=Layer_Dense(64,num_classes)
activation_2=Activation_Softmax()
loss_function=Loss_CategoricalCrossEntropy()
loss_activation=Activation_Softmax_Loss_CategoricalCrossEntropy()
#optimizer=Optimizer_SGD(learning_rate=0.1,momentum=0.9)
optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-7)
for epoch in range(10001):
    loader=DataLoader(dataset,batch_size=32,shuffle=True) #create new loader for each epoch to reshuffle data
    for X_batch,y_batch in loader:
        dense_1.forward(X_batch)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)
        loss=loss_function.calculate(activation_2.output,y_batch)
        predictions=np.argmax(activation_2.output,axis=1)
        accuracy=np.mean(predictions==y_batch)
        loss_activation.backward(activation_2.output,y_batch)
        dense_2.backward(loss_activation.dinputs)
        activation_1.backward(dense_2.dinputs)
        dense_1.backward(activation_1.dinputs)
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
    if epoch%1000==0:
        print("epoch:",epoch,"loss:",loss,"accuracy:",accuracy)


