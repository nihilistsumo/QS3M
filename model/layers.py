import torch
torch.manual_seed(42)
import torch.nn as nn

class CATS(nn.Module): # CATS
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

class CATS_Ablation(nn.Module):
    def __init__(self, emb_size):
        super(CATS_Ablation, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(3 * emb_size, 1)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        #self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        #self.z1 = torch.abs(self.Xp1 - self.Xq)
        #self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL2(self.LL1(self.Xp1)))
        self.zp2 = torch.relu(self.LL2(self.LL1(self.Xp2)))
        #self.zql = torch.relu(self.LL2(self.LL1(self.Xq)))
        self.zd = torch.abs(self.zp1 - self.zp2)
        #self.zdqp1 = torch.abs(self.zp1 - self.zql)
        #self.zdqp2 = torch.abs(self.zp2 - self.zql)
        #self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        self.z = torch.cat((self.zp1, self.zp2, self.zd), dim=1)
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

class Sent_Attention(nn.Module):
    def __init__(self, emb_size, n):
        super(Sent_Attention, self).__init__()
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.emb_size = emb_size
        self.n = n
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(5 * emb_size, 1)
        self.Wa = nn.Parameter(torch.tensor(torch.randn(2*emb_size, self.n), requires_grad=True).to(device))
        self.va = nn.Parameter(torch.tensor(torch.randn(self.n, 1), requires_grad=True).to(device))
        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity()

    def forward(self, Xq, Xp):
        '''

        :param Xq: context vec of shape (m X vec size)
        :param Xp: para sent vecs of shape (m X 2*vec size + 2 X max seq len)
        :return: Pairwise CATS scores of shape (mC2 X 1)
        '''
        b = Xq.shape[0]
        seq = Xp.shape[2]
        self.Xq = Xq
        self.Xp1 = Xp[:, :self.emb_size + 1, :]
        self.Xp2 = Xp[:, self.emb_size + 1:, :]
        self.Xp1valid = self.Xp1[:, -1, :]
        self.Xp2valid = self.Xp2[:, -1, :]
        self.Xp1 = self.Xp1[:, :self.emb_size, :]
        self.Xp2 = self.Xp2[:, :self.emb_size, :]

        self.Xqp1 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp1), 1)
        self.S1 = torch.mul(self.Xp1valid, torch.mm(self.tanh(
            torch.mm(self.Xqp1.permute(0,2,1).reshape(-1, 2*self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta1 = torch.exp(self.S1) / torch.sum(torch.exp(self.S1), 1).unsqueeze(1).repeat(1, seq)
        self.Xp1dash = torch.sum(torch.mul(self.beta1.reshape(b, 1, seq), self.Xp1), 2)

        self.Xqp2 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp2), 1)
        self.S2 = torch.mul(self.Xp2valid, torch.mm(self.tanh(
            torch.mm(self.Xqp2.permute(0, 2, 1).reshape(-1, 2 * self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta2 = torch.exp(self.S2) / torch.sum(torch.exp(self.S2), 1).unsqueeze(1).repeat(1, seq)
        self.Xp2dash = torch.sum(torch.mul(self.beta2.reshape(b, 1, seq), self.Xp2), 2)

        o = self.cos(self.Xp1dash, self.Xp2dash)
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
        self.Wa = nn.Parameter(torch.tensor(torch.randn(2 * emb_size, self.n), requires_grad=True).to(device))
        self.va = nn.Parameter(torch.tensor(torch.randn(self.n, 1), requires_grad=True).to(device))
        self.tanh = nn.Tanh()

    def forward(self, Xq, Xp):
        '''

        :param Xq: context vec of shape (m X vec size)
        :param Xp: para sent vecs of shape (m X 2*vec size + 2 X max seq len)
        :return: Pairwise CATS scores of shape (mC2 X 1)
        '''
        b = Xq.shape[0]
        seq = Xp.shape[2]
        self.Xq = Xq
        self.Xp1 = Xp[:, :self.emb_size + 1, :]
        self.Xp2 = Xp[:, self.emb_size + 1:, :]
        self.Xp1valid = self.Xp1[:, -1, :]
        self.Xp2valid = self.Xp2[:, -1, :]
        self.Xp1 = self.Xp1[:, :self.emb_size, :]
        self.Xp2 = self.Xp2[:, :self.emb_size, :]

        self.Xqp1 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp1), 1)
        self.S1 = torch.mul(self.Xp1valid, torch.mm(self.tanh(
            torch.mm(self.Xqp1.permute(0, 2, 1).reshape(-1, 2 * self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta1 = torch.exp(self.S1) / torch.sum(torch.exp(self.S1), 1).unsqueeze(1).repeat(1, seq)
        self.Xp1dash = torch.sum(torch.mul(self.beta1.reshape(b, 1, seq), self.Xp1), 2)

        self.Xqp2 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp2), 1)
        self.S2 = torch.mul(self.Xp2valid, torch.mm(self.tanh(
            torch.mm(self.Xqp2.permute(0, 2, 1).reshape(-1, 2 * self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta2 = torch.exp(self.S2) / torch.sum(torch.exp(self.S2), 1).unsqueeze(1).repeat(1, seq)
        self.Xp2dash = torch.sum(torch.mul(self.beta2.reshape(b, 1, seq), self.Xp2), 2)

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

class Sent_FixedCATS_Attention(nn.Module):
    def __init__(self, emb_size, n, cats_model):
        super(Sent_FixedCATS_Attention, self).__init__()
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.emb_size = emb_size
        self.n = n
        self.cats = cats_model
        self.cats.eval()
        self.Wa = nn.Parameter(torch.tensor(torch.randn(2*emb_size, self.n), requires_grad=True).to(device))
        self.va = nn.Parameter(torch.tensor(torch.randn(self.n, 1), requires_grad=True).to(device))
        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity()

    def forward(self, Xq, Xp):
        '''

        :param Xq: context vec of shape (m X vec size)
        :param Xp: para sent vecs of shape (m X 2*vec size + 2 X max seq len)
        :return: Pairwise CATS scores of shape (mC2 X 1)
        '''
        b = Xq.shape[0]
        seq = Xp.shape[2]
        self.Xq = Xq
        self.origXq = Xq
        self.Xp1 = Xp[:, :self.emb_size + 1, :]
        self.Xp2 = Xp[:, self.emb_size + 1:, :]
        self.Xp1valid = self.Xp1[:, -1, :]
        self.Xp2valid = self.Xp2[:, -1, :]
        self.Xp1 = self.Xp1[:, :self.emb_size, :]
        self.Xp2 = self.Xp2[:, :self.emb_size, :]

        self.Xqp1 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp1), 1)
        self.S1 = torch.mul(self.Xp1valid, torch.mm(self.tanh(
            torch.mm(self.Xqp1.permute(0,2,1).reshape(-1, 2*self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta1 = torch.exp(self.S1) / torch.sum(torch.exp(self.S1), 1).unsqueeze(1).repeat(1, seq)
        self.Xp1dash = torch.sum(torch.mul(self.beta1.reshape(b, 1, seq), self.Xp1), 2)

        self.Xqp2 = torch.cat((self.Xq.reshape(b, self.emb_size, 1).expand(-1, -1, seq), self.Xp2), 1)
        self.S2 = torch.mul(self.Xp2valid, torch.mm(self.tanh(
            torch.mm(self.Xqp2.permute(0, 2, 1).reshape(-1, 2 * self.emb_size), self.Wa)), self.va).reshape(b, seq))
        self.beta2 = torch.exp(self.S2) / torch.sum(torch.exp(self.S2), 1).unsqueeze(1).repeat(1, seq)
        self.Xp2dash = torch.sum(torch.mul(self.beta2.reshape(b, 1, seq), self.Xp2), 2)
        X = torch.cat((self.origXq, self.Xp1dash, self.Xp2dash), 1)

        o = self.cats(X)
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


class CATS_Scaled(nn.Module): # CAVS
    def __init__(self, emb_size):
        super(CATS_Scaled, self).__init__()
        self.emb_size = emb_size
        self.n = 32
        self.LL1 = nn.Linear(emb_size, self.n)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.A = nn.Parameter(torch.tensor(torch.randn(self.n, emb_size), requires_grad=True).to(device))
        self.cos = nn.CosineSimilarity()

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.Xlq = torch.relu(self.LL1(self.Xq))
        self.scale = torch.mm(self.Xlq, self.A)
        self.zp1 = torch.mul(self.Xp1, self.scale)
        self.zp2 = torch.mul(self.Xp2, self.scale)

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