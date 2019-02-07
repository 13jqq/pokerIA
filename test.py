import numpy as np
from model import build_model
print(np.std([2,4,8,16]),np.mean([2,4,8,16]))
print(np.std([2,3,4,5,6,7,8,9,10,11,12,13,14]),np.mean([2,3,4,5,6,7,8,9,10,11,12,13,14]))

model=build_model(2)
print(model.summary())