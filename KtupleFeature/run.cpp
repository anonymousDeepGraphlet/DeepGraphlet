#include "graph.h"

using namespace std;

int main(int args, char * argv[])
{
    Graph* G = new Graph(string(argv[1]));
    if(args == 3)
        G->local_transitivity(atoi(argv[2]));
    else
        G->getKtuples(atoi(argv[2]), atoi(argv[4]), atoi(argv[3]));
    delete G;
}