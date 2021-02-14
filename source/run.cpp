#include "graph.h"
#include<cstdio>
#include <igraph/igraph.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>
using namespace std;

std::vector<std::string> csv_read_row(std::istream &in, char delimiter);
std::vector<std::string> csv_read_row(std::string &in, char delimiter);

int u[50000000], v[50000000];

int nodecnt, edgecnt;
int main(int args, char * argv[])
{
    //std::ifstream in("./data/artist_edges.edges");
    std::ifstream in(argv[1]);
    if (in.fail()) return (cout << "File not found" << endl) && 0;
    int count = 0;
    while(in.good())
    {
        std::vector<std::string> row = csv_read_row(in, ' ');
        if(count)
        {
            //cout<<row[0]<<" "<<row[1]<<endl;
            u[count-1] = atoi(row[0].c_str());
            v[count-1] = atoi(row[1].c_str());
        }
        else
        {
            nodecnt = atoi(row[0].c_str());
            edgecnt = atoi(row[1].c_str());
        }
        count++;
    }
    in.close();
    //printf("%d\n%d\n", nodecnt, edgecnt);
    Graph* G = new Graph(nodecnt, edgecnt, u, v, 0);
    //printf("%d\n",igraph_vcount(&G->g));
    G->getKtuples(5, 100, 56);

    delete G;
}

std::vector<std::string> csv_read_row(std::string &line, char delimiter)
{
    std::stringstream ss(line);
    return csv_read_row(ss, delimiter);
}
 
std::vector<std::string> csv_read_row(std::istream &in, char delimiter)
{
    std::stringstream ss;
    bool inquotes = false;
    std::vector<std::string> row;//relying on RVO
    while(in.good())
    {
        char c = in.get();
        if (!inquotes && c=='"') //beginquotechar
        {
            inquotes=true;
        }
        else if (inquotes && c=='"') //quotechar
        {
            if ( in.peek() == '"')//2 consecutive quotes resolve to 1
            {
                ss << (char)in.get();
            }
            else //endquotechar
            {
                inquotes=false;
            }
        }
        else if (!inquotes && c==delimiter) //end of field
        {
            row.push_back( ss.str() );
            ss.str("");
        }
        else if (!inquotes && (c=='\r' || c=='\n') )
        {
            if(in.peek()=='\n') { in.get(); }
            row.push_back( ss.str() );
            return row;
        }
        else
        {
            ss << c;
        }
    }
}
