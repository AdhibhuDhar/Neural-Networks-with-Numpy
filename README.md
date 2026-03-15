# NEURAL ENGINE-A LIGHTWEIGHT DEEP LEARNING FRAMEWORK MADE FROM SCRATCH

## FEATURES
• Dense neural network layers
• Automatic gradient verification (gradient checking)
• Adam and SGD optimizers
• Batch normalization
• Dropout regularization
• Training / evaluation mode switching
• Experiment tracking system
• Model persistence (save/load)
• Training visualization

## EXAMPLE USAGE
model = Model()

model.add(Layer_Dense(num_features,64))
model.add(Layer_BatchNorm(64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))

model.add(Layer_Dense(64,num_classes))
model.add(Activation_Softmax())

model.train(dataset)
model.save("model.npz")