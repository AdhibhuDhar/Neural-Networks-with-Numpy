class Trainer:
    def __init(self,model):
        self.model=model
    def fit(self,dataset,epochs=1000,batch_size=32):
        for epoch in range(epochs):
            output=self.model.forward(dataset.X)
            loss=self.loss.calculate(output,dataset.y)
            self.model.backward(output,dataset.y)
            for layer in self.model.layers:
                if hasattr(layer,"weights"):
                    self.model.optimizer.update_params(layer)
            self.model.optimizer.post_update_params()
            if epoch%1000==0:
                print(epoch,loss)