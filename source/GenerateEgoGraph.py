from GenerateLabel import GenerateLabel
import queue
from utils import split_between_last_char, saveEdgeList, saveSet, readEdgeList
import os
import time
import numpy as np

class EgoGraphGenerator:
    def __init__(self, filePath, sampleNum):
        self.filePath = filePath
        self.sampleNum = sampleNum
        
    def sampling(self):
        start = np.random.randint(len(self.adj))
        
        sampledNodeList = set()
        sampledNodeList.add(start)
        
        q = queue.Queue()
        q.put((start, 0))
        while q.empty() == False:
            u, depU = q.get()
            if depU < 4:
                for v in self.adj[u]:
                    if v in sampledNodeList:
                        continue
                    q.put((v, depU + 1))
                    sampledNodeList.add(v)
                    
        sampledEdgeList = []
        for u in sampledNodeList:
            for v in self.adj[u]:
                if v in sampledNodeList:
                    sampledEdgeList.append([u, v])
        return start, sampledNodeList, sampledEdgeList

    def edgeList2adj(self, n, edgeList):
        self.adj = []
        for i in range(n):
            self.adj.append([])
        for edge in edgeList:
            u, v = edge
            self.adj[u].append(v)
            self.adj[v].append(u)

    def sampleEgoGraph(self):
        n, m, edgeList = readEdgeList(self.filePath)
        self.edgeList2adj(n, edgeList)
        print("--------------------------------------------------------------------------------------------")
        print(n, m)
        prePath, fileName = split_between_last_char(filePath, '/')
        prefix, suffix = split_between_last_char(fileName, '.')
        savePath = prePath + '/' + prefix + "_ego"
        if os.path.exists(savePath):
            print(savePath)
            os.system('rm -rf ' + savePath)
        os.system('mkdir ' + savePath)
        print(prePath, fileName)
        print(prefix, suffix)
        print(savePath)
        for i in range(self.sampleNum):
            startNode, sampledNodeList, sampledEdgeList = self.sampling()
            saveEdgeList(savePath + '/' + str(i) + '.edge', sampledEdgeList)
            print('node|edgeList length', len(sampledNodeList), len(sampledEdgeList))
            #santize??? edge/edges
            #GenerateLabel(savePath + '/' + str(i) + '.edge')

    #     saveSet(savePath + '/' + 'nonSampled.nodes', nodeList)
    
if __name__ == '__main__':
    time_start = time.time()
    filePath = "../newData/com-friendster.edges"
    egoGraphGenerator = EgoGraphGenerator(filePath, 10)
    egoGraphGenerator.sampleEgoGraph()
    time_end = time.time()
    print("end: time cost" + str(time_end - time_start) + "s")
    