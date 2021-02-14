import os
import sys
import psutil

def printCurrentProcessMemory():
    process = psutil.Process(os.getpid())
    print('Current Used Memory: ', process.memory_info().rss / 1024 / 1024, 'MB')

def printItemMemory(item):
    print(sys.getsizeof(item) / 1024 / 1024, 'MB')
    
def split_between_last_char(path, ch):
    items = path.split(ch)
    ans = ""
    for i in range(len(items) - 1):
        ans += items[i]
        if i != len(items) - 2:
            ans += ch
    return ans, items[-1]

def saveEdgeList(path, edgeList):
    f = open(path, "w")
    for edge in edgeList: 
        u, v = edge
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

def saveSet(path, items):
    f = open(path, "w")
    for item in items:
        f.write(str(item) + '\n')
    f.close()
    
def readEdgeList(filePath):
    f = open(filePath)
    line = f.readline()
    strN, strM = line.strip('\n').split(' ')
    n, m = int(strN), int(strM.split('.')[0])
#     print(n, m)
    line = f.readline()
    edgeList = []
    while line:
#         print(line)
        strU, strV = line.split(' ')
        u, v = int(strU), int(strV)
        edgeList.append([u, v])
        line = f.readline()
    return n, m, edgeList