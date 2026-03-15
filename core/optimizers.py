import numpy as np
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

            #update batchnorm parameters
            if hasattr(layer,"gamma"):
                layer.gamma+=-self.learning_rate*layer.dgamma
                layer.beta+=-self.learning_rate*layer.dbeta


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

        #update batchnorm parameters
        if hasattr(layer,"gamma"):
            layer.gamma+=-self.current_learning_rate*layer.dgamma
            layer.beta+=-self.current_learning_rate*layer.dbeta

    def post_update_params(self):
        self.iterations+=1
        #Adam=remember dir->measure terrain slope->adjust step size->correct bias->move effiiciently towards minima