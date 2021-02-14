from GenerateLabel import GenerateLabel
import queue
from utils import split_between_last_char, saveEdgeList, saveSet, readEdgeList
import os
import time
from KTupleFeatureGeneratorCPP import KTupleFeatureGenerator

def sampling(nodeList, adj, expectedNodeNum):
    sampledNodeList = set()
    while len(sampledNodeList) < expectedNodeNum:
        q = queue.Queue()
        for node in nodeList:
            q.put(node)
            sampledNodeList.add(node)
            nodeList.remove(node)
            break
#         print(len(sampledNodeList))
        while q.empty() == False:
            u = q.get()
            for v in adj[u]:
                if len(sampledNodeList) >= expectedNodeNum:
                    break
                if v in nodeList:
                    sampledNodeList.add(v)
                    nodeList.remove(v)
                    q.put(v)
            if len(sampledNodeList) >= expectedNodeNum:
                break
    sampledEdgeList = []
    for u in sampledNodeList:
        for v in adj[u]:
            if v in sampledNodeList:
                sampledEdgeList.append([u, v])
    return sampledNodeList, sampledEdgeList

def edgeList2adj(n, edgeList):
    adj = []
    for i in range(n):
        adj.append([])
    for edge in edgeList:
        u, v = edge
        adj[u].append(v)
        adj[v].append(u)
    return adj

def partition(filePath, splitDen, splitNum):
    n, m, edgeList = readEdgeList(filePath)
    print("--------------------------------------------------------------------------------------------")
    print(n, m)
    prePath, fileName = split_between_last_char(filePath, '/')
    prefix, suffix = split_between_last_char(fileName, '.')
    savePath = prePath + '/' + prefix
    if os.path.exists(savePath):
        print(savePath)
        os.system('rm -rf ' + savePath)
    os.system('mkdir ' + savePath)
    print(prePath, fileName)
    print(prefix, suffix)
    print(savePath)
    nodeList = set()
    for i in range(n):
        nodeList.add(i)
    expectedNodeNum = len(nodeList) / splitDen
    adj = edgeList2adj(n, edgeList)
    for i in range(splitNum):
        print(len(nodeList))
        sampledNodeList, sampledEdgeList = sampling(nodeList, adj, expectedNodeNum)
        print(len(nodeList))
        saveEdgeList(savePath + '/' + str(i) + '.edge', sampledEdgeList)
        print('node|edgeList length', len(sampledNodeList), len(sampledEdgeList))
        GenerateLabel(savePath + '/' + str(i) + '.edge')
#         for v in sampledNodeList:
#             nodeList.remove(v)

    saveSet(savePath + '/' + 'nonSampled.nodes', nodeList)

def partitionSize(filePath, nodeSize):
    n, m, edgeList = readEdgeList(filePath)
    print("--------------------------------------------------------------------------------------------")
    print(n, m)
    if nodeSize >= n / 2:
        return
    prePath, fileName = split_between_last_char(filePath, '/')
    prefix, suffix = split_between_last_char(fileName, '.')
    prefix += str(nodeSize)
    savePath = prePath + '/' + prefix
    if os.path.exists(savePath):
        print(savePath)
        os.system('rm -rf ' + savePath)
    os.system('mkdir ' + savePath)
    print(prePath, fileName)
    print(prefix, suffix)
    print(savePath)
    nodeList = set()
    for i in range(n):
        nodeList.add(i)
    expectedNodeNum = nodeSize
    adj = edgeList2adj(n, edgeList)
    accNodeNum = 0
    for i in range(100000):
        accNodeNum += nodeSize
        if accNodeNum >= n / 2:
            break
        print(len(nodeList))
        sampledNodeList, sampledEdgeList = sampling(nodeList, adj, expectedNodeNum)
        print(len(nodeList))
        saveEdgeList(savePath + '/' + str(i) + '.edge', sampledEdgeList)
        print('node|edgeList length', len(sampledNodeList), len(sampledEdgeList))
        GenerateLabel(savePath + '/' + str(i) + '.edge')
#         for v in sampledNodeList:
#             nodeList.remove(v)

    saveSet(savePath + '/' + 'nonSampled.nodes', nodeList)
    
def partitionNumber(filePath, graphNum):
    n, m, edgeList = readEdgeList(filePath)
    print(n, m)
    prePath, fileName = split_between_last_char(filePath, '/')
    prefix, suffix = split_between_last_char(fileName, '.')
    savePath = prePath + '/' + prefix + str(graphNum)
    print(savePath)
    if os.path.exists(savePath):
        print(savePath)
        os.system('rm -rf ' + savePath)
    os.system('mkdir ' + savePath)
    print(prePath, fileName)
    print(prefix, suffix)
    print(savePath)
    nodeList = set()
    for i in range(n):
        nodeList.add(i)
    expectedNodeNum = len(nodeList) / 30
    adj = edgeList2adj(n, edgeList)
    for i in range(graphNum):
        print(len(nodeList))
        sampledNodeList, sampledEdgeList = sampling(nodeList, adj, expectedNodeNum)
        print(len(nodeList))
        saveEdgeList(savePath + '/' + str(i) + '.edge', sampledEdgeList)
        print('node|edgeList length', len(sampledNodeList), len(sampledEdgeList))
        GenerateLabel(savePath + '/' + str(i) + '.edge')
#         for v in sampledNodeList:
#             nodeList.remove(v)

    saveSet(savePath + '/' + 'nonSampled.nodes', nodeList)
    
def partitionSize(filePath, graphSize):
    n, m, edgeList = readEdgeList(filePath)
    print(n, m)
    prePath, fileName = split_between_last_char(filePath, '/')
    prefix, suffix = split_between_last_char(fileName, '.')
    savePath = prePath + '/' + prefix + str(graphSize) + "_"
    print(savePath)
    if os.path.exists(savePath):
        print(savePath)
        os.system('rm -rf ' + savePath)
    os.system('mkdir ' + savePath)
    print(prePath, fileName)
    print(prefix, suffix)
    print(savePath)
    nodeList = set()
    for i in range(n):
        nodeList.add(i)
    expectedNodeNum = len(nodeList) / graphSize
    adj = edgeList2adj(n, edgeList)
    for i in range(15):
        print(len(nodeList))
        sampledNodeList, sampledEdgeList = sampling(nodeList, adj, expectedNodeNum)
        print(len(nodeList))
        saveEdgeList(savePath + '/' + str(i) + '.edge', sampledEdgeList)
        print('node|edgeList length', len(sampledNodeList), len(sampledEdgeList))
        GenerateLabel(savePath + '/' + str(i) + '.edge')
#         for v in sampledNodeList:
#             nodeList.remove(v)

    saveSet(savePath + '/' + 'nonSampled.nodes', nodeList)
    
if __name__ == '__main__':
    time_start = time.time()
#     filePath = "/data/sujintao/motif/newData/com-orkut.edges"
#     partition(filePath, 30, 15)
    filePaths = ["../newData/artist_edges.edges", "../newData/web-BerkStan.edges", "../newData/com-lj.edges", "../newData/com-orkut.edges"]
# #     nodeSizes = [1000, 5000, 10000, 50000, 100000]
#     nodeSizes = [100000]
#     for filePath in filePaths:
#         for nodeSize in nodeSizes:
#             partitionSize(filePath, nodeSize)


    graphNums = [5, 10, 20]
    for filePath in filePaths:
        for graphNum in graphNums:
            print('------------------------------------------------------------------------')
            print(filePath)
            print(graphNum)
            partitionNumber(filePath, graphNum)

    graphSizes = [60, 40, 20]
    for filePath in filePaths:
        for graphSize in graphSizes:
            print('------------------------------------------------------------------------')
            print(filePath)
            print(graphSize)
            partitionSize(filePath, graphSize)
            
    for path in ["../newData/artist_edges.edges", "../newData/web-BerkStan.edges", "../newData/com-lj.edges", "../newData/com-orkut.edges"]:
        KTupleFeatureGenerator(path = path).generateDataFeature()
    for path in ["../newData/artist_edges.edges", "../newData/web-BerkStan.edges", "../newData/com-lj.edges", "../newData/com-orkut.edges"]:
        KTupleFeatureGenerator(path = path).generateDataFeature2()
    time_end = time.time()
    print("end: time cost" + str(time_end - time_start) + "s")
    