import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
def create_data(points,classes):
    X=np.zeros((points*classes,2))
    y=np.zeros(points*classes,dtype='uint8')
    for class_number in range(classes):
        ix=range(points*class_number,points*(class_number+1))
        r=np.linspace(0.0,1,points)
        t=np.linspace(class_number*4,(class_number+1)*4,points)+np.random.randn(points)*0.2
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
        y[ix]=class_number
    return X,y
x=create_data(100,3)[0]
y=create_data(100,3)[1]
#print("Here")
#showing the dataset
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', edgecolors='black')
plt.colorbar(label='Class')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset')
#plt.show()