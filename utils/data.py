import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
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
    def __init__(self,path=None,label_column=None,X=None,y=None):
        if path is not None and label_column is not None:
            self.X,self.y=load_dataset(path,label_column)
        else:
            self.X=X
            self.y=y
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1] if len(self.X.shape) > 1 else 0
class DataLoader:
    def __init__(self,dataset,batch_size=32,shuffle=True):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
    def __iter__(self):
        n = min(self.dataset.n_samples, len(self.dataset.y) if hasattr(self.dataset.y, '__len__') else self.dataset.n_samples)
        self.indices=np.arange(n) 
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current=0
        return self
    def __next__(self): #fetch next batch
        if self.current>=self.dataset.n_samples:
            raise StopIteration
        start=self.current
        end=min(self.current+self.batch_size,self.dataset.n_samples) #calculate end index for batch
        batch_indices=self.indices[start:end] 
        X_batch=self.dataset.X[batch_indices] 
        y_batch=self.dataset.y[batch_indices]
        self.current+=self.batch_size 
        return X_batch,y_batch
def train_test_split(X,y,test_size=0.2,shuffle=True):
    n_samples=X.shape[0]
    indices=np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    split=int(n_samples*(1-test_size))#calculate split index
    train_idx=indices[:split]
    test_idx=indices[split:]
    X_train=X[train_idx]
    y_train=y[train_idx]
    X_test=X[test_idx]
    y_test=y[test_idx]
    return X_train,X_test,y_train,y_test