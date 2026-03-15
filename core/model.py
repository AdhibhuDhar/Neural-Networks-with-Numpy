import numpy as np
from data import Dataset,DataLoader
from losses import Loss_CategoricalCrossEntropy,Activation_Softmax_Loss_CategoricalCrossEntropy
#from optimizers import Optimizer_SGD,Optimizer_Adam

class Model:
    def __init__(self):
        self.layers=[]
        self.training=True
        self.history={
            "loss":[],
            "accuracy":[]
        }
    def train_mode(self):
        self.training=True
    def eval_mode(self):
        self.training

    def add(self,layer):
        #self.layers.append(layer)
        #automatic backprop,layers know their neighbours
        if len(self.layers)>0:
            layer.prev=self.layers[-1]
            self.layers[-1].next=layer

        self.layers.append(layer)

    def set(self,*,loss,optimizer):
        self.loss=loss
        self.optimizer=optimizer
        self.loss_activation=Activation_Softmax_Loss_CategoricalCrossEntropy() #combine softmax and CE for efficiency

    def forward(self,X):
        self.layers[0].forward(X)
        for layer in self.layers[1:]:
            layer.forward(layer.prev.output)
        return self.layers[-1].output

    #add bwd and train functions later
    def backward(self,output,y):
        #softmaxx+CE
        self.loss_activation.backward(output,y)
        self.layers[-1].dinputs=self.loss_activation.dinputs
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.dinputs)
    
    def train(self,dataset,epochs=10000,batch_size=32):
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        for epoch in range(epochs):
           
            for X_batch,y_batch in loader:
                output=self.forward(X_batch)#fwd
                loss=self.loss.calculate(output,y_batch)#loss
                predictions=np.argmax(output,axis=1)
                accuracy=np.mean(predictions==y_batch)
                self.backward(output,y_batch)#bwd
                for layer in self.layers:#update params
                    if hasattr(layer,"weights"):
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params() #update learning rate if needed
            if epoch%1000==0:
                print(
                    "epoch:", epoch,
                    "loss:", loss,
                    "accuracy:", accuracy
                )

    def evaluate(self,dataset):
        X=dataset.X
        y=dataset.y
        if len(X.shape) == 1:
             X = X.reshape(-1, 1)
        output=self.forward(X)
        predictions=np.argmax(output,axis=1)
        accuracy=np.mean(predictions==y)
        print("Test Accuracy:",accuracy)
    def save(self,filepath):
        params={}
        for i,layer in enumerate(self.layers):
            if hasattr(layer,"weights"):
                params[f"W{i}"]=layer.weights
                params[f"b{i}"]=layer.biases
            if hasattr(layer,"gamma"):
                params[f"gamma{i}"]=layer.gamma
                params[f"beta{i}"]=layer.beta
            np.savez(filepath,**params)
    def load(self,filepath):
        data=np.load(filepath)
        for i,layer in enumerate(self.layers):
            if hasattr(layer,"weights"):
                layer.weights=data[f"W{i}"]
                layer.biases=data[f"b{i}"]
            if hasattr(layer,"gamma"):
                layer.gamma=data[f"gamma{i}"]
                layer.beta=data[f"beta{i}"]
    def plot_training(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.history["loss"])
        plt.title("Training Loss")
        plt.xlabel("Checkpoints")
        plt.ylabel("Loss")
        plt.subplot(1,2,2)
        plt.plot(self.history["accuracy"])
        plt.title("Training Accuracy")
        plt.xlabel("Checkpoints")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.show()
            

