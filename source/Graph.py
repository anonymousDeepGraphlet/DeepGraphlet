import queue
from Alias import Alias

class Graph:
    def __init__(self, nodeCnt, edgeCnt = 0, edgeList = []):
        self.nodeCnt = nodeCnt
        self.edgeCnt = edgeCnt
        self.edgeList = edgeList
        self.deg = [0 for i in range(nodeCnt)]
#         self.constructListAdj()
    
    def constructListAdj(self):
        self.adj = []
        for i in range(self.nodeCnt):
            self.adj.append([])
        for edge in self.edgeList:
            self.adj[edge[0]].append(edge[1])
            self.edgeCnt += 1
            self.adj[edge[1]].append(edge[0])
            self.deg[edge[0]] += 1
            self.deg[edge[1]] += 1
            
    def constructSetAdj(self):
        self.adj = []
        for i in range(self.nodeCnt):
            self.adj.append(set())
        for edge in self.edgeList:
            self.adj[edge[0]].add(edge[1])
            self.edgeCnt += 1
            self.adj[edge[1]].add(edge[0])
            self.deg[edge[0]] += 1
            self.deg[edge[1]] += 1
    
    def constructNeighborProb(self):
        self.alias = []
        for u in range(self.nodeCnt):
            sumDeg = 0
            for v in self.adj[u]:
                sumDeg += self.deg[v]
            probs = []
            for v in self.adj[u]:
                probs.append(self.deg[v] / sumDeg)
            self.alias.append(Alias(probs))
                
    def addUnDirectedEdgeList(self, u, v):
        #!!! can be optimized
#         self.nodeCnt = self.max(self.nodeCnt, v + 1)
        
        #??? should add edge v to u
#         print(len(self.adj), u, v)
        self.adj[u].append(v)
        self.edgeCnt += 1
        self.adj[v].append(u)
        self.deg[u] += 1
        self.deg[v] += 1
        
    def addUnDirectedEdgeSet(self, u, v):
        #!!! can be optimized
#         self.nodeCnt = self.max(self.nodeCnt, v + 1)
        
        #??? should add edge v to u
#         print(len(self.adj), u, v)
        self.adj[u].add(v)
        self.edgeCnt += 1
        self.adj[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1
        
    def nextPermutation(self, nums):
        pos = -1;
        for i in range(len(nums) - 1, 0, -1):
            if nums[i - 1] < nums[i]:
                pos = i - 1
                break
        if pos != -1:
            pos2 = 0
            for i in range(len(nums) - 1, pos, -1):
                if nums[i] > nums[pos]:
                    pos2 = i
                    break
            t = nums[pos]
            nums[pos] = nums[pos2];
            nums[pos2] = t
        # print(pos)
        # print(nums[pos+1:])
        i = pos + 1
        j = len(nums) - 1
        while i < j:
            t = nums[i]
            nums[i] = nums[j]
            nums[j] = t
            i += 1
            j -= 1
        return pos

    def isIsomorphism(self, g):
        if self.nodeCnt != g.nodeCnt:
            return False
        indexs = [i for i in range(self.nodeCnt)]
        flag = True
#         indexs = [0, 2, 3, 1, 4]
        while flag:
#             print(indexs)
            same = True
            for i in range(self.nodeCnt):
#                 print("step: ", i)
                if len(self.adj[i]) != len(g.adj[indexs[i]]):
                    same = False
                    break
                tmpAdj = []
                for j in range(len(self.adj[i])):
                    tmpAdj.append(indexs[self.adj[i][j]])
                tmpAdj.sort()
#                 print(tmpAdj, g.adj[i])
                for j in range(len(tmpAdj)):
                    if tmpAdj[j] != g.adj[indexs[i]][j]:
#                     if indexs[self.adj[i][j]] != g.adj[indexs[i]][j]:
                        same = False
                        break
                if same == False:
                    break
            if same == True:
                return True
#             break
            flag = (self.nextPermutation(indexs) != -1)
        return False
    
    def isConnected(self):
        q = queue.Queue()
        vis = [0 for i in range(self.nodeCnt)]
        q.put(0)
        vis[0] = 1
        cnt = 1
        while q.empty() != True:
            u = q.get()
            for v in self.adj[u]:
                if vis[v] == True:
                    continue
                vis[v] = 1
                q.put(v)
                cnt += 1
        return cnt == self.nodeCnt
    
    def print(self):
        print('nodeCnt: ', self.nodeCnt)
        print('edgeCnt: ', self.edgeCnt)
        for i in range(len(self.adj)):
            print(i, ": ", self.adj[i])
    
if __name__ == '__main__':
    g1 = Graph(nodeCnt = 5, edgeList = [[0, 1], [0, 2], [0, 3], [0, 4]])
    g2 = Graph(nodeCnt = 5, edgeList = [[0, 1], [1, 2], [1, 3], [1, 4]])
    g1.print()
    g2.print()
    print(g1.isIsomorphism(g2))