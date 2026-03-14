import numpy as np
class Layer_Dropout:
    def __init__(self,rate):
        self.rate=1-rate #eg 0.2 rate meas 20% neuron switched off
    def forward(self,inputs,training=True):
        self.inputs=inputs
        if not training: #training->dropout active,inference->dropout disabled
            self.output=inputs
            return
        #make mask
        self.binary_mask=np.random.binomial(
            1,self.rate,size=inputs.shape
        )/self.rate #without dividing,avg activation becomes smaller,so we scale it
        #this is called inverted dropout
        #apply mask
        self.output=inputs*self.binary_mask
    def backward(self,dvalues):
        self.dinputs=dvalues*self.binary_mask #only backprop through active neurons