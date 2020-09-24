import torch
torch.manual_seed(42)
import torch.nn as nn

class CATS(nn.Module):
    def __init__(self, emb_size):
        super(CATS, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(5 * emb_size, 1)

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
        self.zp1 = torch.relu(self.LL2(self.LL1(self.Xp1)))
        self.zp2 = torch.relu(self.LL2(self.LL1(self.Xp2)))
        self.zql = torch.relu(self.LL2(self.LL1(self.Xq)))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL3(self.z))
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

class CATS_Attention(nn.Module):
    def __init__(self, emb_size, n):
        super(CATS_Attention, self).__init__()
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.emb_size = emb_size
        self.n = n
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(5 * emb_size, 1)
        self.Wa = torch.tensor(torch.randn(self.n, 2*emb_size), requires_grad=True).to(device)
        self.va = torch.tensor(torch.randn(1,self.n), requires_grad=True).to(device)
        self.tanh = nn.Tanh()

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (m X (3*vec size + 2) X N) where m = batch size, N = max seq len
        extra two values in dim 1 are the valid bits for p1 and p2 in the current sample
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        xnumpy = X.numpy()
        self.Xq = X[:, :self.emb_size, :]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size+1, :]
        self.Xp2 = X[:, 2 * self.emb_size+1:]
        self.Xp1valid = self.Xp1[:, -1, :]
        self.Xp2valid = self.Xp2[:, -1, :]
        self.Xp1 = self.Xp1[:, :self.emb_size, :]
        self.Xp2 = self.Xp2[:, :self.emb_size, :]
        self.Xp1score = self.Xp1valid * (torch.bmm(self.va, self.tanh(torch.bmm(self.Wa, torch.cat((self.Xq, self.Xp1), 1)))))
        self.Xp2score = self.Xp2valid * (torch.bmm(self.va, self.tanh(torch.bmm(self.Wa, torch.cat((self.Xq, self.Xp2), 1)))))
        self.Xp1beta = (torch.exp(self.Xp1score) / torch.sum(torch.exp(self.Xp1score), 2)).reshape((-1,1,self.n))
        self.Xp2beta = (torch.exp(self.Xp2score) / torch.sum(torch.exp(self.Xp2score), 2)).reshape((-1,1,self.n))
        self.Xp1dash = torch.sum(torch.mul(self.Xp1beta, self.Xp1), 2)
        self.Xp2dash = torch.sum(torch.mul(self.Xp2beta, self.Xp2), 2)

        self.z1 = torch.abs(self.Xp1dash - self.Xq)
        self.z2 = torch.abs(self.Xp2dash - self.Xq)
        self.zdiff = torch.abs(self.Xp1dash - self.Xp2dash)
        self.zp1 = torch.relu(self.LL2(self.LL1(self.Xp1dash)))
        self.zp2 = torch.relu(self.LL2(self.LL1(self.Xp2dash)))
        self.zql = torch.relu(self.LL2(self.LL1(self.Xq)))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL3(self.z))
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
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(emb_size, emb_size)
        self.cos = nn.CosineSimilarity()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.zql = torch.relu(self.LL2(self.LL1(self.Xq)))
        self.zp1 = torch.mul(self.zql, self.Xp1)
        self.zp2 = torch.mul(self.zql, self.Xp2)
        o = self.cos(self.zp1, self.zp2)
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

class CATS_manhattan(nn.Module):
    def __init__(self, emb_size):
        super(CATS_manhattan, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)

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
        self.zp1 = torch.relu(self.LL2(self.LL1(self.Xp1)))
        self.zp2 = torch.relu(self.LL2(self.LL1(self.Xp2)))
        self.zql = torch.relu(self.LL2(self.LL1(self.Xq)))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.p1tr = torch.cat((self.zp1, self.zdqp1), dim=1)
        self.p2tr = torch.cat((self.zp2, self.zdqp2), dim=1)
        o = torch.exp(-torch.sum(torch.abs(self.p1tr-self.p2tr), dim=1))
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