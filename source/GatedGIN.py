import torch
import torch.nn as nn
from MLP import MLP
# from torch_geometric.nn import GINConv
# from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from utils import printCurrentProcessMemory, printItemMemory
import copy
import gc

class GatedGIN_pyg(torch.nn.Module):
    def __init__(self, nfeat, nhid, nlayer, nclasses, mlpPos, loss = "kl", useDropout = False, keepProb = 0.5, useBatchNorm = False, layerNorm = False, detach = False):
        super(GatedGIN_pyg, self).__init__()
        self.nlayer = nlayer
        self.mlpPos = mlpPos
        self.useDropout = useDropout
        self.keepProb = keepProb
        self.useBatchNorm = useBatchNorm
        self.loss = loss
        self.layerNorm = layerNorm
        self.detach = detach
        
        self.GinMlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.rnns = nn.ModuleList()
        for i in range(nlayer):
            self.GinMlps.append(MLP([nhid, nhid, nhid], useDropout, keepProb, useBatchNorm))
            if self.useBatchNorm == True:
                self.bns.append(nn.BatchNorm1d(nhid))
            self.rnns.append(torch.nn.GRUCell(nhid, nhid, bias=True))
        
        self.mlp1 = MLP([nfeat, nhid], useDropout, keepProb, useBatchNorm)
        if self.useBatchNorm == True:
            self.bn1 = nn.BatchNorm1d(nhid)
        self.mlps = nn.ModuleList()
        for i in range(len(mlpPos)):
            self.mlps.append(MLP([nhid, nhid, nclasses[i]], useDropout, keepProb, useBatchNorm))

        
    def forward(self, features, adj):
        _outputs = []
        outputs = []
        
        output = self.mlp1(features)
        if self.useBatchNorm:
            output = self.bn1(output)
        output = F.relu(output)
        if self.useDropout:
            output = F.dropout(output, p = self.keepProb, training=self.training)
        layerInfo = output.clone()
        for i in range(self.nlayer):
            gc.collect()
            output = torch.spmm(adj, output)
            output = self.rnns[i](output, layerInfo)
            output = self.GinMlps[i](output)
            
            if self.useBatchNorm:
                output = self.bns[i](output)
            output = F.relu(output)
            if self.useDropout:
                output = F.dropout(output, p = self.keepProb, training=self.training)
            if self.layerNorm == True:
                output = output / torch.sqrt(1 + torch.sum(torch.square(output), dim = 1, keepdim = True))
            for j in range(len(self.mlpPos)):
                if (i == self.mlpPos[j]):
                    tmp = self.mlps[j](output)
                    if self.loss == "kl":
                        preds = F.log_softmax(tmp, dim = 1)
                        _preds = F.softmax(tmp, dim = 1)
                        _outputs.append(_preds)
                        outputs.append(preds)
                    else:
                        preds = F.softmax(tmp, dim = 1)
                        outputs.append(preds)
                    if self.detach == True:
                        output = output.detach()
            layerInfo = output.clone()
                        
        if self.loss == "kl":
            return outputs, _outputs
        else:
            return outputs