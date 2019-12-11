import torch
from qpth.qp import QPFunction

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction, QPSolvers


class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=1, neq=0, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.M = Variable(torch.tril(torch.ones(nCls, nCls)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
        self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
        self.z0 = Parameter(torch.zeros(nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())
        
        print("M=")       
        print(self.M)       
 
        print("L=")       
        print(self.L)  
     
        print("G=")       
        print(self.G)  
        
        print("z0=")       
        print(self.z0)  
        
        print("s0=")       
        print(self.s0)  
     
    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        h = self.G.mv(self.z0)+self.s0
        e = Variable(torch.Tensor())
        x = QPFunction(verbose=False)(Q, x, G, h, e, e)

        return F.log_softmax(x)

def main():
    # params: features, nhidden, nCls, bn  
    net = OptNet(28*28, 5, 10, 1)
    for epoch in range(1,10):
        print('Epoch: ' + str(epoch))
        net.train()     

if __name__=='__main__':
    main()

