from MLP import MLP
from PipeLine import PipeLine
from ResultWritter import ResultWritter
import time
import os

def experimentsS(deviceID, writeID):
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
    mlpPos = [0, 1, 2]
    nclasses = [2, 6, 21]
    for dataName in ["com-lj"]:
#     for dataName in ["com-orkut"]:
        for i in range(5):#5
            args = {}
            for k in [0, 1, 2]:#3
                for j in range(3):#3
                    args['numLayer'] = mlpPos[j] + k + 1
                    args['mlpPos'] = [mlpPos[j] + k]
                    args['nclasses'] = [nclasses[j]]
                    
                    if writeID == 4:
                        args['baseModel'] = "GatedGIN"
                        args['useKTupleFeature'] = True
                    elif writeID == 5:
                        args['baseModel'] = "GIN"
                        args['useKTupleFeature'] = False
                    elif writeID == 6:
                        args['baseModel'] = "GCN"
                        args['useKTupleFeature'] = False
                        
                    args['numIterator'] = 200
                    args['use3Feature'] = False
                    args['useRandomFeature'] = False
                    args['layerNorm'] = False
                    args['learningRate'] = 0.001
                    args['weightDecay'] = 0
                    args['useDropout'] = True
                    args['keepProb'] = 0.5
                    args['useBatchNorm'] = True
                    args['detach'] = False
                    args['aggregator'] = "mean"
                    if writeID == 5:
                        args['aggregator'] = "sum"

                    pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = deviceID)
                    pipeLine.trainSynGraph(["../newData/" + dataName])
                    valData = ["../newData/" + dataName + ".edges"]
                    model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = deviceID)
                    model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                    model.inferRealGraph(valData, needLabel = True)
                    time_end = time.time()
                    print("time cost",time_end - time_start,'s')
                    resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                
def experimentsM(deviceID, writeID):
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-orkut", "web-BerkStan", "com-lj"]:
    for dataName in ["artist_edges"]:
        for i in range(5):#5
            args = {}
            args['numLayer'] = 3
            args['mlpPos'] = [0, 1, 2]
            for j in [0, 1, 2]:#3
                
                args['numLayer'] = 3 + j
                args['mlpPos'][0] = 0 + j
                args['mlpPos'][1] = 1 + j
                args['mlpPos'][2] = 2 + j
                if writeID == 1:
                    args['baseModel'] = "GatedGIN"
                    args['useKTupleFeature'] = True
                elif writeID == 2:
                    args['baseModel'] = "GatedGIN"
                    args['useKTupleFeature'] = False
                elif writeID == 3:
                    args['baseModel'] = "GIN"
                    args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "mean"

                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = deviceID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = deviceID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                
def experiments(deviceID, writeID):
    if writeID == 1 or writeID == 2 or writeID == 3:
        experimentsM(deviceID, writeID)
    elif writeID == 4 or writeID == 5 or writeID == 6:
        experimentsS(deviceID, writeID)
#     if ID == 1 or ID == 2 or ID == 3:
#         experimentsM(ID)
#     elif ID == 4 or ID == 5 or ID == 6:
#         experimentsS(ID)

def runPerformanceExperiments():
#     experiments(0, 2)
    experiments(1, 1)
#     experiments(1, 3)
#     experiments(0, 4)
#     experiments(1, 2)
#     experiments(1, 5)
#     experiments(6, 3)
#     experiments(6, 6)


def runDeepLGC_Parameters(deviceID, writeID):
    
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
    
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj"]:
    for dataName in ["com-orkut"]:
         for i in range(1):#5
            for layerNorm in [Fal]:
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['nclasses'] = [2, 6, 21]

                args['layerNorm'] = False
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = False
                args['numIterator'] = 200
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['aggregator'] = "GCN"

                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')

def runDeepLGC_M_Parametes(deviceID, writeID):
    
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
    
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj"]:
            
            
    mlpPos = [0, 1, 2]
    nclasses = [2, 6, 21]
    for dataName in ["com-orkut"]:
        for i in range(5):#5
            for j in range(3):
                args = {}
                args['numLayer'] = mlpPos[j] + 1
                args['mlpPos'] = [mlpPos[j]]
                args['nclasses'] = [nclasses[j]]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['aggregator'] = "GCN"

                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')

def runTransferExperiments(deviceID):
    writeID = deviceID
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-lj", "com-orkut"]:
#     for dataName in ["artist_edges"]:
#     for dataName in ["com-lj"]:
    for dataName in ["com-orkut"]:
        for i in range(5):#5
            args = {}
            args['numLayer'] = 3
            args['mlpPos'] = [0, 1, 2]
            for j in range(1):#3
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "mean"
                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                
                for testName in ["artist_edges", "web-BerkStan", "com-lj", "com-orkut"]:
                    valData = ["../newData/" + testName + ".edges"]
                    model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                    model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                    model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                resultWritter.writeResult('summary.txt', dataName)
                
                args['numLayer'] += 1
                args['mlpPos'][0] += 1
                args['mlpPos'][1] += 1
                args['mlpPos'][2] += 1
                
def runMultiNumber(deviceID):
    writeID = deviceID
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-lj", "com-orkut"]:
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj"]:
    for dataName in ["com-orkut"]:
        for nodeSize in [5, 10, 20]:
            for i in range(5):#5
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "mean"
                if os.path.exists("../newData/" + dataName + str(nodeSize)) == False:
                    break
                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName + str(nodeSize)])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                resultWritter.writeResult('summary.txt', dataName + str(nodeSize))
                

                
def runLayer(deviceID, writeID):
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
    for dataName in ["web-BerkStan"]:
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj", "com-orkut"]:
        for i in range(5):#5
            args = {}
            args['numLayer'] = 3
            args['mlpPos'] = [0, 1, 2]
            for j in range(3):
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "mean"

                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                
                args['numLayer'] += 1
                args['mlpPos'][0] += 1
                args['mlpPos'][1] += 1
                args['mlpPos'][2] += 1    
                
def runAggregator(deviceID):
    writeID = deviceID
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
    for dataName in ["com-orkut"]:
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj"]:
#     for dataName in ["com-lj", "com-orkut", "artist_edges", "web-BerkStan"]:
#         for nodeSize in [1000, 5000, 10000, 50000, 100000]:
        for aggregator in ["sum"]:
            for i in range(5):#5
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = aggregator

                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                
def runGraphNum(deviceID, writeID):
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-lj", "com-orkut"]:
#     for dataName in ["artist_edges", "web-BerkStan"]:
    for dataName in ["artist_edges", "web-BerkStan", "com-lj", "com-orkut"]:
        for graphNum in ['5', '10', '20']:
            for i in range(5):#5
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "GCN"
                if os.path.exists("../newData/" + dataName + str(graphNum)) == False:
                    break
                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName + str(graphNum)])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                resultWritter.writeResult('summary.txt', dataName + str(nodeSize))  
                
def runGraphSize(deviceID, writeID):
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-lj", "com-orkut"]:
#     for dataName in ["artist_edges", "web-BerkStan"]:
    for dataName in ["artist_edges", "web-BerkStan", "com-lj", "com-orkut"]:
        for graphSize in ['60', '40', '20']:
            for i in range(5):#5
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "GCN"
                if os.path.exists("../newData/" + dataName + str(graphSize) + "_") == False:
                    break
                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName + str(graphSize) + "_"])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                resultWritter.writeResult('summary.txt', dataName + str(nodeSize))   
def runMultiSize(deviceID):
    writeID = deviceID
    time_start = time.time()
    nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    resultWritter = ResultWritter("../result/" + nowTime)
#     for dataName in ["com-lj", "com-orkut"]:
#     for dataName in ["artist_edges", "web-BerkStan", "com-lj"]:
    for dataName in ["com-orkut"]:
        for nodeSize in ['60_', '40_', '20_']:
            for i in range(5):#5
                args = {}
                args['numLayer'] = 3
                args['mlpPos'] = [0, 1, 2]
                args['baseModel'] = "GatedGIN"
                args['useKTupleFeature'] = True
                args['numIterator'] = 200
                args['nclasses'] = [2, 6, 21]
                args['use3Feature'] = False
                args['useRandomFeature'] = False
                args['layerNorm'] = False
                args['learningRate'] = 0.001
                args['weightDecay'] = 0
                args['useDropout'] = True
                args['keepProb'] = 0.5
                args['useBatchNorm'] = True
                args['detach'] = False
                args['aggregator'] = "mean"
                if os.path.exists("../newData/" + dataName + str(nodeSize)) == False:
                    break
                pipeLine = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                pipeLine.trainSynGraph(["../newData/" + dataName + str(nodeSize)])
                valData = ["../newData/" + dataName + ".edges"]
                model = PipeLine(resultWritter = resultWritter, args = args, deviceID = deviceID, writeInfo = writeID)
                model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
                model.inferRealGraph(valData, needLabel = True)
                time_end = time.time()
                print("time cost",time_end - time_start,'s')
                resultWritter.writeResult('summary.txt', "time cost" + str(time_end - time_start) + 's')
                resultWritter.writeResult('summary.txt', dataName + str(nodeSize))                
if __name__ == '__main__':
#     runPerformanceExperiments()
#     runTransferExperiments(1)
#     runDeepLGC_M_Parametes(5, 5)
#     runMultiNumber(1)
    runMultiSize(1)
#     runAggregator(0)
#     runLayer(6, 6)
