#idea=analytical grad if = mathematical grad,the model is working fine
#dL/dw ≈ (L(w+ε) − L(w−ε)) / (2ε)
import numpy as np
def gradient_check(model,X,y,epsilon=1e-5):
    print("Running Gradient Check")
    #fwd pass
    output=model.forward(X)
    loss=model.loss.calculate(output,X)
    #backprop
    model.backward(output,y)
    for layer in model.layers:
        if not hasattr(layer,"weights"):
            continue
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                original=layer.weights[i,j]
                #w+epsilon
                layer.weights[i,j]=original+epsilon
                out_plus=model.forward(X)
                loss_plus=model.loss.calculate(out_plus,y)
                #w-epsilon
                layer.weights[i,j]=original-epsilon
                out_minus=model.forward(X)
                loss_minus=model.loss.calculate(out_minus,y)
                #restore
                layer.weights[i,j]=original
                numerical_grad=(loss_plus-loss_minus)/(2*epsilon)
                backprop_grad=layer.dweights[i,j]
                diff=abs(numerical_grad-backprop_grad)
                if  diff > 1e-4:
                    print("Gradient mismatch", diff)
                    return
        return("Grad check passed")

