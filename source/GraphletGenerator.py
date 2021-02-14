from Graph import Graph

class GraphletGenerator:
    def __init__(self, K):
        self.K = K
        self.Graphlets = []
    
    def nonExists(self, g):
        for graphlet in self.Graphlets:
            if graphlet.isIsomorphism(g):
                return False
        return True
    
    def getIsomorphismID(self, g):
        for i in range(len(self.Graphlets)):
            if self.Graphlets[i].isIsomorphism(g):
                return i
        return -1
    
    def generateGraphlet(self):
        maxEdgeCnt = self.K * (self.K - 1) // 2
        maxEdgeState = 1 << maxEdgeCnt

        edges = []
        for i in range(self.K - 1):
            for j in range(i + 1, self.K):
                edges.append([i, j])
        
        for i in range(maxEdgeState):
#             print(i)
            g = Graph(nodeCnt = self.K)
            g.constructListAdj()
            for j in range(maxEdgeCnt):
                if (i & (1 << j)):
                    g.addUnDirectedEdgeList(edges[j][0], edges[j][1])
            if g.isConnected() == False:
                continue
            if self.nonExists(g):
                self.Graphlets.append(g)
#                 g.print()
        print(len(self.Graphlets))
#         for i in range(len(self.Graphlets)):
#             print(i)

            
if __name__ == '__main__':
    gen = GraphletGenerator(5)
    gen.generateGraphlet()