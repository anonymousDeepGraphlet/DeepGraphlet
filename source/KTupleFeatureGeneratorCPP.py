from GraphletGenerator import GraphletGenerator
from utils import readEdgeList, split_between_last_char
from Graph import Graph
import numpy as np
import time
from multiprocessing import Pool
from ResultWritter import ResultWritter
import os

class KTupleFeatureGenerator:
    def __init__(self, path):
        self.path = path

    def generateKTupleFeature(self, path):
#         os.system('rm ' + path + ".feature5")
        os.system('./run ' + path + " 3")
        os.system('./run ' + path + " 4")
        os.system('./run ' + path + " 5")

    def generateDataFeature(self):
        print(self.path)
#         self.generateKTupleFeature(self.path)
#         for suffix in ['1000', '5000', '10000', '50000', '100000']:
        for suffix in ['5', '10', '20']:
#         for suffix in ['5', '10', '20', '60_', '40_', '20_']:
            prefix, _ = split_between_last_char(self.path, '.')
            prefix += suffix
            print(prefix)
            if os.path.exists(prefix):
                filenames = os.listdir(prefix)
                filenames = [(prefix + "/" + name) for name in filenames]
                fileNames = []
                for name in filenames:
                    if name.split('.')[-1] == "edges":
                        print(name)
                        self.generateKTupleFeature(name)
    def generateDataFeature2(self):
        print(self.path)
#         self.generateKTupleFeature(self.path)
#         for suffix in ['1000', '5000', '10000', '50000', '100000']:
        for suffix in ['60_', '40_', '20_']:
#         for suffix in ['5', '10', '20', '60_', '40_', '20_']:
            prefix, _ = split_between_last_char(self.path, '.')
            prefix += suffix
            print(prefix)
            if os.path.exists(prefix):
                filenames = os.listdir(prefix)
                filenames = [(prefix + "/" + name) for name in filenames]
                fileNames = []
                for name in filenames:
                    if name.split('.')[-1] == "edges":
                        print(name)
                        self.generateKTupleFeature(name)
                
if __name__ == '__main__':
    time_start = time.time()
#     for path in ["../newData/com-orkut.edges"]:
#     for path in ["../newData/artist_edges.edges", "../newData/web-BerkStan.edges", "../newData/com-lj.edges"]:
#         KTupleFeatureGenerator(path = path).generateDataFeature()
    for path in ["../newData/artist_edges.edges", "../newData/web-BerkStan.edges", "../newData/com-lj.edges", "../newData/com-orkut.edges"]:
        KTupleFeatureGenerator(path = path).generateDataFeature2()
    time_end = time.time()
    print("end: time cost" + str(time_end - time_start) + "s")
    