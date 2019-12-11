import torch
import torch.nn as nn
from torch.nn import functional as fn
from torch.autograd import Variable


import numpy as np
#np.random.seed(99)
f = lambda x: 1*x**2 
xgrid = np.linspace(0.,1., 640)
fgrid = f(xgrid)
ygrid = fgrid + 0.1*np.random.normal(size=640)

#%matplotlib inline

#import matplotlib.pyplot as plt
#plt.plot(xgrid, fgrid, lw=2)
#plt.plot(xgrid, ygrid, '.')




xdata = Variable(torch.Tensor(xgrid))
ydata = Variable(torch.Tensor(ygrid))


