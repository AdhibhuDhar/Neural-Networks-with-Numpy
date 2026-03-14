import numpy as np
class Layer_BatchNorm:
    #if one weight becomes really high,the output cascades
    #we normalize output from activation func
    #multiply normalized output with arbitary param gamma
    #then add aother arb param to the product(beta)
    #this sets a new standard deviation
    #makes sure weights in the network dont become imbalanced with extremely high or low value
    def __init__(self,n_features,epsilon=1e-5):#n-features is number of neurons
        self.epsilon=epsilon
        self.gamma=np.ones((1,n_features)) #scale
        self.beta=np.zeros((1,n_features)) #shift
    def forward(self,inputs,training=True):
        self.inputs=inputs
        self.mean=np.mean(inputs,axis=0,keepdims=True)
        self.var=np.var(inputs,axis=0,keepdims=True)
        #normalize
        self.x_hat=(inputs-self.mean)/np.sqrt(self.var+self.epsilon)
        #scale and shift
        self.output=self.gamma*self.x_hat+self.beta
    def backward(self,dvalues):
        N=dvalues.shape[0]#batch size
        #dist from mean
        x_mu=self.inputs-self.mean
        std_inv=1./np.sqrt(self.var+self.epsilon)
        dx_hat=dvalues*self.gamma#gradient thru scale
        #chain rule->op=gamma*normalized
        dvar=np.sum(dx_hat*x_mu*-0.5*std_inv**3,axis=0,keepdims=True)
        dmean = np.sum(dx_hat * -std_inv, axis=0, keepdims=True) + dvar * np.sum(-2. * x_mu, axis=0, keepdims=True) / N
        self.dinputs=dx_hat*std_inv+dvar*2*x_mu/N+dmean/N
        self.dgamma=np.sum(dvalues*self.x_hat,axis=0,keepdims=True)
        self.dbeta=np.sum(dvalues,axis=0,keepdims=True)


