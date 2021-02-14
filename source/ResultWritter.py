import os
class ResultWritter:
    def __init__(self, prefix):
        if os.path.exists(prefix) == False:
            os.system("mkdir " + prefix)
        self.prefix = prefix
        self.paths = {}

    def writeResult(self, resultPath, s):
        if resultPath not in self.paths:
            self.paths[resultPath] = 1
            os.system("rm -rf " + self.prefix + "/" + resultPath)
        info = open(self.prefix + "/" + resultPath, "a+")
        info.write(s + "\n")
        info.close()

    def saveDic(self, resultPath, dic):
        print(type(dic))
        for k, v in dic.items():
            line = str(k) + "      "
            if type(v) is list:
                for item in v:
                    line += "," + str(item)
            else:
                line += str(v)
            print(line)
            self.writeResult(resultPath, line)

    def saveList(self, resultPath, dic):
        print(type(dic))
        line = ""
        for item in dic:
            line += "," + str(item)
        print(line)
        self.writeResult(resultPath, line)
        
    def saveListLine(self, resultPath, dic):
        for item in dic:
            self.writeResult(resultPath, str(item))