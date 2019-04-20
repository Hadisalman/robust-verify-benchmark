import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
# import cvxpy as cp

from . import affine as Aff

def AffineTranspose(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineTransposeLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineTransposeConv2d(l)
    else:
        raise ValueError("AffineTranspose class not found for given layer.")

def Affine(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineConv2d(l)
    else:
        raise ValueError("Affine class not found for given layer.")

def full_bias(l, n=None): 
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified.
    if isinstance(l, nn.Linear): 
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d): 
        b = l.bias.unsqueeze(1).unsqueeze(2)
        k = int((n/(b.numel()))**0.5)
        return b.expand(b.numel(),k,k).contiguous().view(1,-1)
    else:
        raise ValueError("Full bias can't be formed for given layer.")

def batch(A, n):
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])

class DualNetBounds: 
    def __init__(self, net, X, epsilon, alpha_grad=False, scatter_grad=False):
        n = X.size(0)

        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d))]
        self.affine_transpose = [AffineTranspose(l) for l in self.layers]
        self.affine = [Affine(l) for l in self.layers]
        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = X[0].view(1,-1)
        for a in self.affine: 
            _ = a(_)
            
        self.biases = [full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        gamma = [self.biases[0]]
        nu = []
        nu_hat_x = self.affine[0](X)

        eye = Variable(torch.eye(self.affine[0].in_features)).type_as(X)
        nu_hat_1 = self.affine[0](eye).unsqueeze(0)

        l1 = (nu_hat_1).abs().sum(1)
        
        self.zl = [nu_hat_x + gamma[0] - epsilon*l1]
        self.zu = [nu_hat_x + gamma[0] + epsilon*l1]

        self.I = []
        self.I_empty = []
        self.I_neg = []
        self.I_pos = []
        self.X = X
        self.epsilon = epsilon
        I_collapse = []
        I_ind = []

        # set flags
        self.scatter_grad = scatter_grad
        self.alpha_grad = alpha_grad

        for i in range(0,self.k-2):
            # compute sets and activation
            self.I_neg.append((self.zu[-1] <= 0).detach())
            self.I_pos.append((self.zl[-1] > 0).detach())
            self.I.append(((self.zu[-1] > 0) * (self.zl[-1] < 0)).detach())
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            
            I_nonzero = ((self.zu[-1]!=self.zl[-1])*self.I[-1]).detach()
            d = self.I_pos[-1].type_as(X).clone()

            # Avoid division by zero by indexing with I_nonzero
            if I_nonzero.data.sum() > 0:
                d[I_nonzero] += self.zu[-1][I_nonzero]/(self.zu[-1][I_nonzero] - self.zl[-1][I_nonzero])

            # indices of [example idx, origin crossing feature idx]
            I_ind.append(Variable(self.I[-1].data.nonzero()))
            
            # initialize new terms
            if not self.I_empty[-1]:
                out_features = self.affine[i+1].out_features

                subset_eye = Variable(X.data.new(self.I[-1].data.sum(), d.size(1)).zero_())
                subset_eye.scatter_(1, I_ind[-1][:,1,None], d[self.I[-1]][:,None])

                if not scatter_grad: 
                    subset_eye = subset_eye.detach()
                nu.append(self.affine[i+1](subset_eye))

                # create a matrix that collapses the minibatch of origin-crossing indices 
                # back to the sum of each minibatch
                I_collapse.append(Variable(X.data.new(I_ind[-1].size(0), X.size(0)).zero_()))
                I_collapse[-1].scatter_(1, I_ind[-1][:,0][:,None], 1)
            else:
                nu.append(None)         
                I_collapse.append(None)
            gamma.append(self.biases[i+1])
            # propagate terms
            gamma[0] = self.affine[i+1](d * gamma[0])
            for j in range(1,i+1):
                gamma[j] = self.affine[i+1](d * gamma[j])
                if not self.I_empty[j-1]: 
                    nu[j-1] = self.affine[i+1](d[I_ind[j-1][:,0]] * nu[j-1])

            nu_hat_x = self.affine[i+1](d*nu_hat_x)
            nu_hat_1 = batch(self.affine[i+1](unbatch(d.unsqueeze(1)*nu_hat_1)), n)
            
            l1 = (nu_hat_1).abs().sum(1)
            
            # compute bounds
            self.zl.append(nu_hat_x + sum(gamma) - epsilon*l1 + 
                           sum([(self.zl[j][self.I[j]] * (-nu[j].t()).clamp(min=0)).mm(I_collapse[j]).t()
                                for j in range(i+1) if not self.I_empty[j]]))
            self.zu.append(nu_hat_x + sum(gamma) + epsilon*l1 - 
                           sum([(self.zl[j][self.I[j]] * nu[j].t().clamp(min=0)).mm(I_collapse[j]).t()
                                for j in range(i+1) if not self.I_empty[j]]))
        
        self.s = [torch.zeros_like(u) for l,u in zip(self.zl, self.zu)]

        for (s,l,u) in zip(self.s,self.zl, self.zu): 
            I_nonzero = (u != l).detach()
            if I_nonzero.data.sum() > 0: 
                s[I_nonzero] = u[I_nonzero]/(u[I_nonzero]-l[I_nonzero])
        
    def g(self, c):
        n = c.size(0)
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = batch(self.affine_transpose[i](unbatch(nu[i+1])),n)
            if i > 0:
                # avoid in place operation
                out = nu[i].clone()
                out[self.I_neg[i-1].unsqueeze(1).expand_as(out)] = 0
                if not self.I_empty[i-1]:
                    if self.alpha_grad: 
                        out[self.I[i-1].unsqueeze(1).expand_as(out)] = (self.s[i-1].unsqueeze(1).expand_as(nu[i])[self.I[i-1].unsqueeze(1).expand_as(nu[i])] * 
                                                               nu[i][self.I[i-1].unsqueeze(1).expand_as(nu[i])])
                    else:
                        out[self.I[i-1].unsqueeze(1).expand_as(out)] = ((self.s[i-1].unsqueeze(1).expand_as(nu[i])[self.I[i-1].unsqueeze(1).expand_as(nu[i])] * 
                                                                                   torch.clamp(nu[i], min=0)[self.I[i-1].unsqueeze(1).expand_as(nu[i])])
                                                         + (self.s[i-1].detach().unsqueeze(1).expand_as(nu[i])[self.I[i-1].unsqueeze(1).expand_as(nu[i])] * 
                                                                                   torch.clamp(nu[i], max=0)[self.I[i-1].unsqueeze(1).expand_as(nu[i])]))
                nu[i] = out

        f = (-sum(nu[i+1].matmul(self.biases[i].view(-1)) for i in range(self.k-1))
             -nu[0].matmul(self.X.view(self.X.size(0),-1).unsqueeze(2)).squeeze(2)
             -self.epsilon*nu[0].abs().sum(2)
             + sum((nu[i].clamp(min=0)*self.zl[i-1].unsqueeze(1)).matmul(self.I[i-1].type_as(self.X).unsqueeze(2)).squeeze(2) 
                    for i in range(1, self.k-1) if not self.I_empty[i-1]))

        return f

def robust_loss(net, epsilon, X, y, 
                size_average=True, alpha_grad=False, scatter_grad=False):
    num_classes = net[-1].out_features
    dual = DualNetBounds(net, X, epsilon, alpha_grad, scatter_grad)
    c = Variable(torch.eye(num_classes).type_as(X.data)[y.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()
    f = -dual.g(c)
    err = (f.data.max(1)[1] != y.data)
    if size_average: 
        err = err.sum()/X.size(0)
        ce_loss = nn.CrossEntropyLoss(reduction='elementwise_mean')(f, y)
    else:
        ce_loss = nn.CrossEntropyLoss(size_average=size_average)(f, y)

    return ce_loss, err