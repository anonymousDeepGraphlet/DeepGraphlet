from GraphletGenerator import GraphletGenerator
from utils import readEdgeList, split_between_last_char
from Graph import Graph
import numpy as np
import time
from multiprocessing import Pool
import os

class KTupleFeatureGenerator:
    def __init__(self, path):
        self.K = 5
        self.sampleNum = 100
        self.value = 1. / self.sampleNum
        self.path = path
        self.graphletGenerator = GraphletGenerator(self.K)
        self.graphletGenerator.generateGraphlet()
        
    def chooseNeighbor(self, node):
        neighborIndex = self.listGraph.alias[node].draw()
#         print(self.listGraph.adj[node])
#         print(neighborIndex)
        neighbor = self.listGraph.adj[node][neighborIndex]
        return neighbor
    
    def sampleKTuple(self, u):
        KTuple = [u]
        KTupleDeg = [self.listGraph.deg[u]]
#         print(self.listGraph.deg[u])
        sumDeg = self.listGraph.deg[u]
#         self.listGraph.constructNeighborProb()
        for i in range(1, self.K):
            probs = []
            for deg in KTupleDeg:
                probs.append(deg / sumDeg)
            node = np.random.choice(KTuple, p = probs)
#             nodeIndex = np.random.choice([j - 1 for j in range(i)], p = probs)
#             node = KTuple[nodeIndex]
#             print(nodeIndex, len(KTuple), node)
            neighbor = self.chooseNeighbor(node)
#             print(node, neighbor)
            KTuple.append(neighbor)
            KTupleDeg.append(self.listGraph.deg[neighbor])
            sumDeg += self.listGraph.deg[neighbor]
            #define KTuple
        return KTuple
    
    def generateGraphFromTuple(self, KTuple):
        kGraph = Graph(len(KTuple))
        kGraph.constructListAdj()
        #what if this contains the same node IDs
        for i in range(self.K):
            for j in range(i + 1, self.K):
                if KTuple[j] in self.setGraph.adj[KTuple[i]]:
                    kGraph.addUnDirectedEdgeList(i, j)
        return kGraph
    
    def ops(self, i):
        time_start = time.time()
        result = np.zeros((1, len(self.graphletGenerator.Graphlets)))
        for j in range(self.sampleNum):
#             print(j)
            kTuple = self.sampleKTuple(i)
            kGraph = self.generateGraphFromTuple(kTuple)
#                 kGraph.print()
            kTupleID = self.graphletGenerator.getIsomorphismID(kGraph)
#                 self.kTuples[i].append(kTupleID)
            result[0][kTupleID] += self.value
#                 break
#             break
        time_end = time.time()
        if i % 100 == 0:
            print(i, " ", time_end - time_start)
        return result

    def generateKTupleFeature(self, path):
        nodeCnt, edgeCnt, edgeList = readEdgeList(path)
        self.prefix, tmp = split_between_last_char(path, "/")
        self.fileName, _ = split_between_last_char(tmp, ".")
        print(self.prefix, self.fileName)
        self.nodeCnt = nodeCnt
        self.setGraph = Graph(nodeCnt = nodeCnt, edgeList = edgeList)
        self.setGraph.constructSetAdj()
        self.listGraph = Graph(nodeCnt = nodeCnt, edgeList = edgeList)
        self.listGraph.constructListAdj()
        self.listGraph.constructNeighborProb()
        
        time_start = time.time()
        stepCnt = 0
        self.kTupleFeature = np.zeros((self.nodeCnt, len(self.graphletGenerator.Graphlets)))
        
#         print('enter op')
#         self.ops(0)
#         print('end op')
        print('enter pool')
        p = Pool(50)
        results = p.map(self.ops, [i for i in range(self.nodeCnt)])
        p.close()
        p.join()
        
        for i in range(self.nodeCnt):
            self.kTupleFeature[i] = results[i]
#         print(self.kTupleFeature)
        np.savetxt(self.prefix + "/" + self.fileName + ".fea", self.kTupleFeature)
        return self.kTupleFeature

    def generateDataFeature(self):
        print(self.path)
        self.generateKTupleFeature(self.path)
        prefix, _ = split_between_last_char(self.path, '.')
        print(prefix)
        filenames = os.listdir(prefix)
        filenames = [(prefix + "/" + name) for name in filenames]
        fileNames = []
        for name in filenames:
            if name.split('.')[-1] == "edges":
                print(name)
                self.generateKTupleFeature(name)
                
if __name__ == '__main__':
    time_start = time.time()
    KTupleFeatureGenerator(path = "../newData/com-lj.edges").generateDataFeature()
    time_end = time.time()
    print("end: time cost" + str(time_end - time_start) + "s")
    