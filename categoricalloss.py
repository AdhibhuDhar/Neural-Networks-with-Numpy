import math
import numpy as np

softmax_output=np.array([[0.7,0.1,0.2],
                        [0.1,0.5,0.4],
                        [0.02,0.9,0.08]])
class_targets=np.array([0,1,1])
print(softmax_output[[0,1,2],class_targets])
#categorical loss
neg_log=-np.log(softmax_output[range(len(softmax_output)),class_targets])
average_loss=np.mean(neg_log)
#but we encounter a problem when the output is 0 because log(0) is undefined and we get nan as output
#so we clip the op to a small value to avoid log(0)
y_clipped=np.clip(softmax_output,1e-7,1-1e-7)
