import numpy as np
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
        #return data_loss
        return negative_log_likelihood

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples