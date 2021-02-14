from GIN import GIN_pyg
from GCN import GCN
from Graph import Graph
from GatedGIN import GatedGIN_pyg
# from GatedGIN_LayerNorm import GatedGIN_LayerNorm
# from GIN_LayerNorm import GIN_LayerNorm
from RMSE import RMSE
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
# import networkx as nx
import pandas as pd
# from torch_geometric.utils import to_undirected
from utils import printCurrentProcessMemory, printItemMemory, readEdgeList, split_between_last_char

device = torch.device('cuda')
cpu = torch.device('cpu')



class PipeLine:
    def __init__(self, trainData = None, trainInfo = None, valData = None, valInfo = None, resultWritter = "", args = {}, deviceID = 1, writeInfo = 1):
        self.resultWritter = resultWritter
        self.args = {}
        
        #Set manually
        self.args['numLayer'] = 3
        self.args['nclasses'] = [2, 6, 21]
        self.args['mlpPos'] = [0, 1, 2]
        self.args['baseModel'] = "GatedGIN"
        self.args['numIterator'] = 1000
        
        self.args['useRandomFeature'] = False
        self.args['useKTupleFeature'] = True
        self.args['use3Feature'] = False
        
        self.args['useDropout'] = True
        self.args['useBatchNorm'] = True
        self.args['layerNorm'] = True
        self.args['detach'] = False
        
        #Grid Search
        self.args['learningRate'] = 0.001
        self.args['weightDecay'] = 0
        self.args['keepProb'] = 0.5
        
        #nearly fixed parameters
        self.args['aggregator'] = "GCN"
        self.args['activateFunc'] = "relu"
        self.args['hiddenDim'] = 128
        self.args['nodeFeatureDim'] = 1
        self.args['loss'] = "kl"
        for key, value in args.items():
            self.args[key] = value
        
        if self.args['useRandomFeature'] == True:
            self.args['nodeFeatureDim'] = 1
        if self.args['useKTupleFeature'] == True:
            self.args['nodeFeatureDim'] = 29
        if self.args['use3Feature'] == True:
            self.args['nodeFeatureDim'] = 3
        
        print("featureDim: ", self.args['nodeFeatureDim'])
        if self.args['baseModel'] == "GatedGIN":
            self.model = GatedGIN_pyg(nfeat = self.args['nodeFeatureDim'], nhid = self.args['hiddenDim'], nlayer = self.args['numLayer'], nclasses = self.args['nclasses'], mlpPos = self.args['mlpPos'], useDropout = self.args['useDropout'], keepProb =self.args['keepProb'], useBatchNorm = self.args['useBatchNorm'], layerNorm = self.args['layerNorm'], detach = self.args['detach'])
        elif self.args['baseModel'] == "GIN":
            self.model = GIN_pyg(nfeat = self.args['nodeFeatureDim'], nhid = self.args['hiddenDim'], nlayer = self.args['numLayer'], nclasses = self.args['nclasses'], mlpPos = self.args['mlpPos'], useDropout = self.args['useDropout'], keepProb =self.args['keepProb'], useBatchNorm = self.args['useBatchNorm'], layerNorm = self.args['layerNorm'], detach = self.args['detach'])
        elif self.args['baseModel'] == "GCN":
            self.model = GCN(nfeat = self.args['nodeFeatureDim'], nhid = self.args['hiddenDim'], nlayer = self.args['numLayer'], nclasses = self.args['nclasses'], mlpPos = self.args['mlpPos'], useDropout = self.args['useDropout'], keepProb =self.args['keepProb'], useBatchNorm = self.args['useBatchNorm'], layerNorm = self.args['layerNorm'], detach = self.args['detach'])
            
        self.writeInfo = str(writeInfo)
        
        torch.cuda.set_device(deviceID)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args['learningRate'], weight_decay = self.args['weightDecay'])
            
        self.trainData = {}
        self.trainInfo = {}
        self.valData = {}
        self.valInfo = {}
        if trainData != None:
            self.trainData = trainData
            self.trainInfo = trainInfo
            self.valData = valData
            self.valInfo = valInfo

        if self.args['loss'] == "kl":
            self.criterion = nn.KLDivLoss(reduction = 'batchmean')
        elif self.args['loss'] == "mse":
            self.criterion = RMSE()
        self.model.to(device)
        

    def SaveModel(self, modelPath):
        torch.save(self.model.state_dict(), modelPath)
        
    def LoadModel(self, modelPath):
        self.model.load_state_dict(torch.load(modelPath))
        
    
    def GenerateLabel(self, nodeCnt, path):
        fileName = path.split('/')[-1].split('.')[0]
        items = path.split('/')
        prefix = ""
        for i in range(len(items) - 1):
            prefix += items[i] + "/"
        outFile = prefix + fileName + ".out"
        df = pd.read_csv(outFile, sep=' ',header = None).iloc[:, :-1]
        rawLabel = np.array(df)
        
#         labels = []
#         labels.append(rawLabel[:, 1 : 4])
#         labels.append(rawLabel[:, 4 : 15])
#         labels.append(rawLabel[:, 15 : 73])
#         for i in range(len(labels)):
#             labels[i] += 1e-10
#             labels[i] = labels[i] / np.sum(labels[i], axis = 1).reshape((labels[i].shape[0], 1))
#         return labels
    
        
        orbits = [1,    2, 1,   2, 2, 1, 3, 2, 1,    3, 4, 2, 3, 4, 3, 1, 4, 4, 2, 4, 2, 3, 2, 3, 3, 3, 3, 2, 2, 1]
        motifs = np.zeros((nodeCnt, 30), dtype = np.float64)
        cnt = 0
        st = 0
        for length in orbits:
            motifs[:, cnt] = np.sum(rawLabel[:, st : st + orbits[cnt]], axis = 1)
            st += orbits[cnt]
            cnt += 1
        labels = []
        labels.append(motifs[:, 1 : 3])
        labels.append(motifs[:, 3 : 9])
        labels.append(motifs[:, 9 : 30])
        for i in range(len(labels)):
            labels[i] += 1e-10
            labels[i] = labels[i] / np.sum(labels[i], axis = 1).reshape((labels[i].shape[0], 1))
        return labels

    def GenerateFeedDict(self, fileName = "test", needLabel = True):
        
        #Do this edge have been cleaned ?
        nodeCnt, edgeCnt, edgeList = readEdgeList(fileName)
        nodeFeature = np.ones((nodeCnt, self.args['nodeFeatureDim']))
        if self.args['useRandomFeature'] == True:
            nodeFeature[:, 1] = np.random.rand(nodeCnt)
        if self.args['useKTupleFeature'] == True:
            prefix, _ = split_between_last_char(fileName, '.')
            KTupleFeature3 = np.loadtxt(prefix + ".edges_features3")
            KTupleFeature3 = KTupleFeature3 / np.sum(KTupleFeature3, axis = 1, keepdims = True)
            KTupleFeature4 = np.loadtxt(prefix + ".edges_features4")
            KTupleFeature4 = KTupleFeature4 / np.sum(KTupleFeature4, axis = 1, keepdims = True)
            KTupleFeature5 = np.loadtxt(prefix + ".edges_features5")
            KTupleFeature5 = KTupleFeature5 / np.sum(KTupleFeature5, axis = 1, keepdims = True)
            nodeFeature = np.hstack((KTupleFeature3, KTupleFeature4, KTupleFeature5))
            print(KTupleFeature3.shape, KTupleFeature4.shape, KTupleFeature5.shape, nodeFeature.shape)
            
        indices = np.zeros((2, edgeCnt * 2), dtype = np.int64)
        values = np.zeros((edgeCnt * 2), dtype = np.float32)
        edgeCnt = 0
        deg = []
        if self.args['baseModel'] == "GCN" or self.args['aggregator'] == 'GCN' or self.args['aggregator'] == "mean":
            deg = [0 for i in range(nodeCnt)]
            for edge in edgeList:
                u, v = edge
                deg[int(u)] += 1
                deg[int(v)] += 1
            for i in range(nodeCnt):
                if deg[i] == 0:
                    deg[i] = 1
#         jishu = 0
        for edge in edgeList:
            u, v = edge
            indices[0, edgeCnt], indices[1, edgeCnt] = u, v
            if self.args['aggregator'] == "sum":
                values[edgeCnt] = 1.0
            elif self.args['aggregator'] == "mean":
                values[edgeCnt] = 1.0 / deg[int(u)]
            if self.args['baseModel'] == "GCN" or self.args['aggregator'] == "GCN":
#                 val = deg[int(u)]
                values[edgeCnt] = 1.0 / deg[int(u)] / deg[int(v)]
            edgeCnt += 1
            
            v, u = edge
            indices[0, edgeCnt], indices[1, edgeCnt] = u, v
            if self.args['aggregator'] == "sum":
                values[edgeCnt] = 1.0
            elif self.args['aggregator'] == "mean":
                values[edgeCnt] = 1.0 / deg[int(u)]
            if self.args['baseModel'] == "GCN" or self.args['aggregator'] == "GCN":
                values[edgeCnt] = 1.0 / deg[int(u)] / deg[int(v)]
#             print(deg[int(u)], deg[int(v)])
#             print(values[edgeCnt])
#             jishu += 1
#             if jishu == 10:
#                 break
            edgeCnt += 1
            
        sparseAggregator = torch.sparse.FloatTensor(torch.from_numpy(indices), torch.from_numpy(values), torch.Size([nodeCnt, nodeCnt]))
        labels = []
        if needLabel == True:
            labels = self.GenerateLabel(nodeCnt, fileName)
            for i in range(len(labels)):
                labels[i] = torch.from_numpy(labels[i]).float()
        
        
        nodeFeature = torch.from_numpy(nodeFeature).float()
        if self.args['use3Feature'] == True:
            nodeFeature = labels[0].clone()
            
        edgeIndex = (torch.from_numpy(np.array(edgeList).T.astype(int))).type(torch.LongTensor)
        sparseAggregator = sparseAggregator.float()
        return nodeFeature, edgeIndex, sparseAggregator, labels, nodeCnt, edgeCnt
    
    
    
    def loadData(self, filePath, needLabel = True):
        idx = 0
        dataDic = {}
        infoDic = {}
        for path in filePath:
            features, edgeIndex, adj, labels, nodeCnt, edgeCnt = self.GenerateFeedDict(fileName = path, needLabel = needLabel)
            dataDic[idx] = (features, edgeIndex, adj, labels)
            infoDic[idx] = (nodeCnt, edgeCnt)
            idx += 1
        return dataDic, infoDic


#     def writeResult(self, resultPath, x):#Has this been used?
#         info = open(resultPath, "a+")
#         info.write(x + "\n")
#         info.close()
    
#     def saveResult(self, info, loss, preds, labels):#Has this been used?
#         for tmp in loss:
#             item = tmp.cpu().detach().numpy()
#             self.writeResult(info, str(item))
#         for i in range(len(preds)):
#             item1 = preds[i]
#             item2 = labels[i]
#             pred = item1.cpu().detach().numpy()
#             label = item2.cpu().detach().numpy()
#             for i in range(pred.shape[0]):
#                 self.writeResult(info, str(pred[i]) + " " + str(label[i]))
    
        
    def train(self, features, adj, labels, idx = -1):
        self.model.train()
        self.optimizer.zero_grad()
        
        preds = self.model(features, adj)
        if self.args['loss'] == "kl":
            preds = preds[0]
        
            
        losses = []
        loss = 0
#         print(len(preds))
        for i in range(len(preds)):
            if self.args['nclasses'][i] == 2:
                idx = 0
            elif self.args['nclasses'][i] == 6:
                idx = 1
            elif self.args['nclasses'][i] == 21:
                idx = 2
#             print(i, self.args['nclasses'][i], idx)
#             print(preds[i].shape, labels[idx].shape)
            if i == 0:
                loss = self.criterion(preds[i], labels[idx])
            else:
                loss = loss + self.criterion(preds[i], labels[idx])
            losses.append(self.criterion(preds[i], labels[idx]))
        loss.backward()
        self.optimizer.step()
        
        return losses, preds, labels
    
    
    def eval(self, features, adj, labels, criterion, idx = -1):
        with torch.no_grad():
            self.model.eval()
            
            preds = self.model(features, adj)
         
            if len(preds) == 2 and idx != -1:
                preds = preds[idx]
            losses = []
            loss = 0
            for i in range(len(preds)):
                if self.args['nclasses'][i] == 2:
                    idx = 0
                elif self.args['nclasses'][i] == 6:
                    idx = 1
                elif self.args['nclasses'][i] == 21:
                    idx = 2
                if i == 0:
                    loss = criterion(preds[i], labels[idx])
                else:
                    loss = loss + criterion(preds[i], labels[idx])
                losses.append(criterion(preds[i], labels[idx]))

        return losses, preds, labels

    def trainVal(self, synDir):
        precent = 10
        filePath = []
        valPath = []
        for path in synDir:
            filenames = os.listdir(path)
            filenames = [(path + "/" + name) for name in filenames]
            fileNames = []
            for name in filenames:
                if name.split('.')[-1] == "edges":
                    fileNames.append(name)
            np.random.shuffle(fileNames)
            trainNum = int(len(fileNames) / 10 * 7)
            filePath.extend(fileNames[:trainNum])
            valPath.extend(fileNames[trainNum:])
        return filePath, valPath
    
    def torchList2floatList(self, losses):
        results = []
        for loss in losses:
            results.append(float(loss.to(cpu)))
        return results
                
                
    def trainSynGraph(self, synDir = ""):
#         trainRealInfo = "../result/trainSyn" + self.writeInfo + ".txt"
#         os.system("rm " + trainRealInfo)
        if len(self.trainData) == 0:
            print('trainVal split')
            filePath, valPath = self.trainVal(synDir)
            self.trainData, self.trainInfo = self.loadData(filePath)
            self.valData, self.valInfo = self.loadData(valPath)
            print(len(self.trainData), len(self.valData))
        print(len(self.trainData))
        print(self.args)
        modelPath = "../model/real1/realBest.ckpt" + self.writeInfo
        print(modelPath)
        bestValScore = 1e10
        bestValLosses = []
        index = [i for i in range(len(self.trainData))]
        avgLoss = [0 for i in range(len(self.args['nclasses']))]
        step = len(self.valData)
        for i in range(self.args['numIterator']):
            if i % len(self.trainData) == 0:
                np.random.shuffle(index)
            idx = index[i % len(self.trainData)]
            features, edgeIndex, adj, labels = self.trainData[idx]
            features, adj = features.to(device), adj.to(device)

            for j in range(len(labels)):
                 labels[j] = labels[j].to(device)
#             print("enter train")
            result = self.train(features, adj, labels)
#             print("exit train")
            if (i % step == 0) or (i == self.args['numIterator'] - 1):
                print('------------------------------------------------------------------------')
                print(i)
                print("train")
                print(self.trainInfo[idx])
#                 self.resultWritter.writeResult('test.txt', str(result[0]))
                print(result[0])
                valScore, valLosses = self.testRealGraph(printInfo = False)
#                 if (i // step == self.args['numIterator'] // step) or (i // step == self.args['numIterator'] // step - 1):
#                     valScore, valLosses = self.testRealGraph(printInfo = False)
#                     self.result
                print(valScore)
                print(valLosses)
                if valScore < bestValScore:
                    bestValScore = valScore
                    bestValLosses = valLosses
                    self.SaveModel(modelPath)
            
            #is this trans needed???
            features, adj = features.to(cpu), adj.to(cpu)
            for j in range(len(labels)):
                 labels[j] = labels[j].to(cpu)
        print("best valScore: ", bestValScore)
        print("best valLosses: ", bestValLosses)
        self.resultWritter.writeResult('summary.txt', '------------------------------------------------------------------------')
        self.resultWritter.writeResult('summary.txt', "trainDataNum: " + str(len(self.trainData)))
        self.resultWritter.writeResult('summary.txt', "valDataNum: " + str(len(self.valData)))
        self.resultWritter.saveDic('summary.txt', self.args)
        self.resultWritter.writeResult('summary.txt', "bestValScore: " + str(float(bestValScore.to(cpu))))
        self.resultWritter.writeResult('summary.txt', "bestValLosses: ")
        self.resultWritter.saveListLine('summary.txt', self.torchList2floatList(bestValLosses))
#         self.saveResult(trainRealInfo, result[0], result[1], result[2])
        return bestValScore
    
    def testRealGraph(self, filePath = "", printInfo = True):
        if len(self.valData) == 0:
            self.valData, self.valInfo = self.loadData(filePath)
        valScore = 0
        cnt = 0
        totalResult = None
        lossResult = None
        for i in range(len(self.valData)):
            if printInfo:
                self.resultWritter.saveList('test.txt', self.valInfo[i])
                print(self.valInfo[i])
            
            features, edgeIndex, adj, labels = self.valData[i]
            features, adj = features.to(device), adj.to(device)
            for j in range(len(labels)):
                labels[j] = labels[j].to(device)
            result = self.eval(features, adj, labels, self.criterion, 0)
            if printInfo:
                self.resultWritter.writeResult('test.txt', str(result[0]))
                print(result[0])
            if lossResult == None:
                lossResult = list(result[0])
            else:
                for i in range(len(lossResult)):
                    lossResult[i] += result[0][i]
            for item in result[0]:
                valScore += item
                    
            if self.args['loss'] == "kl":
                result = self.eval(features, adj, labels, RMSE(), 1)
                if printInfo:
                    self.resultWritter.writeResult('test.txt', str(result[0]))
                    print(result[0])
                if totalResult == None:
                    totalResult = list(result[0])
                else:
                    for i in range(len(totalResult)):
                        totalResult[i] += result[0][i]
                        
#         for i in range(len(totalResult)):
#             totalResult[i] /= len(self.valData)
        for i in range(len(lossResult)):
            lossResult[i] /= len(self.valData)
        return valScore / len(self.valData), lossResult
    
    def inferRealGraph(self, filePath = "", printInfo = True, needLabel = False, write = False):
        print("Enter LoadData")
        self.model.to(cpu)
        self.valData, self.valInfo = self.loadData(filePath, needLabel = needLabel)
        self.resultWritter.saveList('summary.txt', filePath)
        for i in range(len(self.valData)):
            if printInfo:
                self.resultWritter.saveList('summary.txt', self.valInfo[i])
                print(self.valInfo[i])
            features, edgeIndex, adj, labels = self.valData[i]
            features, adj = features, adj
        with torch.no_grad():
            self.model.eval()
            preds = self.model(features, adj)
        with torch.no_grad():
            self.model.eval()
            
            preds = self.model(features, adj)
            if needLabel == True:
                #kl
                losses = []
                loss = 0
                if self.args['loss'] == "kl":
                    pred = preds[0]
                else:
                    pred = preds
                for i in range(len(pred)):
                    if self.args['nclasses'][i] == 2:
                        idx = 0
                    elif self.args['nclasses'][i] == 6:
                        idx = 1
                    elif self.args['nclasses'][i] == 21:
                        idx = 2
                    if i == 0:
                        loss = self.criterion(pred[i], labels[idx])
                    else:
                        loss = loss + self.criterion(pred[i], labels[idx])
                    losses.append(self.criterion(pred[i], labels[idx]))
                print(self.args['loss'] + ": ", losses)
                self.resultWritter.writeResult('summary.txt', self.args['loss'] + ": ")
                self.resultWritter.saveListLine('summary.txt', self.torchList2floatList(losses))
                
                
                #mse
                if self.args['loss'] == "kl":
                    losses = []
                    loss = 0
                    for i in range(len(preds[1])):
                        if self.args['nclasses'][i] == 2:
                            idx = 0
                        elif self.args['nclasses'][i] == 6:
                            idx = 1
                        elif self.args['nclasses'][i] == 21:
                            idx = 2
                        if i == 0:
                            loss = RMSE()(preds[1][i], labels[idx])
                        else:
                            loss = loss + RMSE()(preds[1][i], labels[idx])
                        losses.append(RMSE()(preds[1][i], labels[idx]))
                    print("RMSE: ", losses)
                    self.resultWritter.writeResult('summary.txt', "RMSE: ")
                    self.resultWritter.saveListLine('summary.txt', self.torchList2floatList(losses))
###write the result to files
        if write == True:
            for i in range(len(preds[1])):
                pred = preds[1][i].detach().numpy()
                print(pred.shape)
                for j in range(pred.shape[0]):
                    self.resultWritter.writeResult(str(i) + ".txt", str(labels[i][j]))
                    self.resultWritter.writeResult(str(i) + ".txt", str(pred[j]))