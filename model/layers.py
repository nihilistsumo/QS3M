import torch
import torch.nn as nn
import numpy as np

class CATS(nn.Module):
    def __init__(self, emb_size):
        super(CATS, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(3 * emb_size, 1)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL1(self.Xp1))
        self.zp2 = torch.relu(self.LL1(self.Xp2))
        self.zql = torch.relu(self.LL1(self.Xq))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL2(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred


class CATS_Scaled(nn.Module):
    def __init__(self, emb_size):
        super(CATS_Scaled, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(5 * emb_size, 1)
        self.A = torch.tensor(torch.randn(emb_size), requires_grad=True).cuda()

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.zp1 = torch.mul(self.Xp1, self.A)
        self.zp2 = torch.mul(self.Xp2, self.A)
        self.zql = torch.mul(self.Xq, self.A)
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL1(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred


class CATS_QueryScaler(nn.Module):
    def __init__(self, emb_size):
        super(CATS_QueryScaler, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(5 * emb_size, 1)
        self.A = torch.tensor(torch.randn(emb_size), requires_grad=True).cuda()

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.zp1 = torch.relu(self.LL1(self.Xp1))
        self.zp2 = torch.relu(self.LL1(self.Xp2))
        self.zql = torch.relu(self.LL1(self.Xq))
        self.zp1a = torch.mul(self.zp1, self.A)
        self.zp2a = torch.mul(self.zp2, self.A)
        self.zqla = torch.mul(self.zql, self.A)
        self.zd = torch.abs(self.zp1a - self.zp2a)
        self.zdqp1 = torch.abs(self.zp1a - self.zqla)
        self.zdqp2 = torch.abs(self.zp2a - self.zqla)
        self.z = torch.cat((self.zp1a, self.zp2a, self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL2(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred