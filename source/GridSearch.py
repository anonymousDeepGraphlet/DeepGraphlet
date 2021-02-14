class GridSearch:
    def __init__(self):
        self.args = {}
        nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.resultWritter = ResultWritter("../result/" + nowTime)
        self.GSWritter = ResultWritter("../result/" + "GridSearch" + nowTime)
        self.GSWritter.saveDic('summary.txt', self.args)
#         self.GSWritter.saveList('summary.txt', synDir)
        self.GSWritter.writeResult('summary.txt', "--------------------------------------------------------------------------------------")

    def trainStep(self, newArgs):
        model = Model(trainData, trainInfo, valData, valInfo, self.resultWritter, gsWritter = self.GSWritter, args = newArgs)
        model.model.cuda()
        return model.trainSynGraph()

    def testStep(self, newArgs):
        model = Model(resultWritter = self.resultWritter, gsWritter = self.GSWritter, args = newArgs)
        valData = ["../data/processed/artist_edges.edges"]
        model.LoadModel("../model/real1/realBest.ckpt" + model.writeInfo)
        model.testRealGraph(valData, real = True, gsWrite = True)

    def run(self):
        newArgs = {}
        bestValScore = 1e10
        bestArgs = {}
        
#         self.args['learningRate'] = 0.001
#         self.args['weightDecay'] = 1e-3
#         self.args['useDropout'] = True
#         self.args['keepProb'] = 0.5
#         self.args['useBatchNorm'] = True
        for learningRate in self.args['learningRate']:
            newArgs['learningRate'] = learningRate
            for weightDecay in self.args['weightDecay']:
                newArgs['weightDecay'] = weightDecay
                for useBatchNorm in self.args['useBatchNorm']:
                    newArgs['useBatchNorm'] = useBatchNorm
                    for useDropout in self.args['useDropout']:
                        newArgs['useDropout'] = useDropout
                        if useDropout == False:
                            nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                            self.resultWritter = ResultWritter("../result/" + nowTime)
                            valScore = self.trainStep(newArgs)
                            if valScore < bestValScore:
                                bestValScore = valScore
                                bestArgs = copy.deepcopy(newArgs)
                            self.testStep(newArgs)
                            self.GSWritter.writeResult('summary.txt', "--------------------------------------------------------------------------------------")
                            continue
                        for keepProb in self.args['keepProb']:
                            newArgs['keepProb'] = keepProb
                            nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                            self.resultWritter = ResultWritter("../result/" + nowTime)
                            valScore = self.trainStep(newArgs)
                            if valScore < bestValScore:
                                bestValScore = valScore
                                bestArgs = copy.deepcopy(newArgs)
                            self.testStep(newArgs)
                            self.GSWritter.writeResult('summary.txt', "--------------------------------------------------------------------------------------")
        self.GSWritter.writeResult('summary.txt', "--------------------------------------------------------------------------------------")
        self.GSWritter.writeResult('summary.txt', "best model")
        nowTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.resultWritter = ResultWritter("../result/" + nowTime)
        valScore = self.trainStep(bestArgs)
        self.testStep(bestArgs)