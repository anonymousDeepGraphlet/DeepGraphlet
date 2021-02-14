#include "graph.h"

std::vector<std::string> csv_read_row(int* count, char delimiter, char * data)
{
    std::stringstream ss;
    std::vector<std::string> row;
    while(true)
    {
        char c = data[*count];
        
        if(c == delimiter)
        {
            row.push_back(ss.str());
            ss.str("");
        }
        else if(c=='\r' || c=='\n')
        {
            if(c == '\n')(*count) = (*count) + 1;
            row.push_back( ss.str() );
            return row;
        }
        else
        {
            ss << c;
        }
        (*count) = (*count) + 1;
    }
}

Graph::Graph(string now_path)
{
    file_path = now_path;
    char *data = NULL;
    int fd=open(now_path.c_str(), O_RDONLY); 
    int size = lseek(fd, 0, SEEK_END);
    data = (char *) mmap( NULL,  size ,PROT_READ, MAP_PRIVATE, fd, 0 );
    close(fd);
    int count = 0;
    int flag = 0;
    int idx = 0;
    while(count<size-1)
    {
        std::vector<std::string> row = csv_read_row(&count, ' ', data);
        if(flag)
        {
            int u = atoi(row[0].c_str());
            int v = atoi(row[1].c_str());
            judge_edge[u].insert(v);
            judge_edge[v].insert(u);
        }
        else
        {
            N = atoi(row[0].c_str());
            M = atoi(row[1].c_str());
            judge_edge.resize(N);
        }
        idx ++;
        flag = 1;
    }
    munmap(data, size);
}

myGraph::myGraph(int nodes)
{
    N = nodes;
    for(int i = 0 ; i < N; i++)
    {
        vector<int> now;
        edges.push_back(now);
    }
}

myGraph::~myGraph()
{
    for(int i = 0 ;i < N ; i++)
        vector<int>().swap(edges[i]);
    vector<vector<int>>().swap(edges);
}

Alias::Alias(vector<double> * weight)
{
    len = weight->size();
    Q.resize(len);
    J.resize(len);
    

    queue<int> smaller;
    queue<int> larger;

    for(int i = 0 ;i < len; i++)
    {
        Q[i] = (*weight)[i] * len;
        if(Q[i] < 1)
            smaller.push(i);
        else
            larger.push(i);
    }
    while(!smaller.empty() && !larger.empty())
    {
        int small = smaller.front();
        smaller.pop();
        int large = larger.front();
        larger.pop();
        J[small] = large;
        Q[large] = Q[large] - (1 - Q[small]);

        if(Q[large] < 1.0)
            smaller.push(large);
        else
            larger.push(large);
    }

}

Alias::~Alias()
{
    vector<double>().swap(Q);
    vector<int>().swap(J);
}

Graph::~Graph()
{
    vector<set<int>>().swap(judge_edge);
}

void myGraph::add_edge(int u, int v)
{
    edges[u].push_back(v);
    edges[v].push_back(u);
}

int GetRandomNumWithWeight(vector<double> * weight)
{
	int size = weight->size();
    double accumulateVal = 0;
    for(int i = 0; i < weight->size(); i++)
        accumulateVal += (*weight)[i];

    double tempSum = 0;
    double ranIndex = accumulateVal * (double)rand()/ double(RAND_MAX);
	for (int j = 0; j < size; j++)
	{
		if (ranIndex <= tempSum + (*weight)[j])
			return j;
        tempSum += (*weight)[j];
    }
}

int Alias::draw()
{
    if(len == 0)
        return -1;
    int now = rand()%len;
    double value = (double)rand() / double(RAND_MAX);
    if(value < Q[now])
        return now;
    else
        return J[now];
}

/*Judge if the two graphs are isomorphism*/
bool judge_isomorphism(myGraph *graph1, const myGraph *graph2)
{
    if(graph1->N != graph2->N)
        return false;

    int N = graph1->N;
    int index[N];
    for(int i = 0;i < N; i++)
        index[i] = i;
    int flag = 0;
    do{
        int judge = 0;
        for(int i = 0 ; i < N; i++)
        {
            if(graph1->edges[i].size() != (graph2->edges[index[i]]).size())
            {
                judge = 1;
                break;
            }

            int len = graph1->edges[i].size();
            unordered_set<int> single;
            for(int j = 0; j < len; j++)
                single.insert(index[graph1->edges[i][j]]);
            
            for(int j = 0; j < len; j++)
                if(single.find(graph2->edges[index[i]][j]) == single.end())
                {
                    judge = 1;
                    break;
                }
        }
        if(judge) continue;
        else 
            return true;
    }
    while(next_permutation(index, index + N));

    
    return flag;
}

bool is_connected(myGraph* g)
{
    int vis[g->N];
    for(int i = 0; i < g->N; i++)
        vis[i] = 0;
    queue<int> q;
    q.push(0);
    vis[0] = 1;
    int sum = 1;
    while(!q.empty())
    {
        int u = q.front();
        q.pop();
        for(int i = 0; i < (g->edges)[u].size(); i++ )
        {
            int v = (g->edges)[u][i];
            if(vis[v] == 0)
            {
                q.push(v);
                vis[v] = 1;
                sum++;
            }
        }
    }
    if(sum == g->N)
        return true;
    else
        return false;
}

/*approximately count the K-tuples*/
void Graph::getKtuples(int K, int num_sample, int thread)
{
    vector<int> degree(N);
    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int i = 0; i < N ; i++)
        degree[i] = judge_edge[i].size();

    vector<vector<int>> adj(N); 
    vector<Alias *> alias(N);
    
    vector<myGraph> mygraphlet;

    int maxEdgeCnt = K * (K - 1) / 2;
    int maxEdgeState = 1 << maxEdgeCnt;
    
    vector<pair<int, int>> alledges;
    for(int i = 0;i < K-1; i++)
        for(int j = i + 1 ; j < K; j++ )
            alledges.push_back(make_pair(i, j));

    for(int i = 0; i < maxEdgeState; i++)
    {
        myGraph * my_now_g = new myGraph(K);
        for(int j = 0; j< maxEdgeCnt; j++)
            if(i & (1<<j)) 
                my_now_g->add_edge(alledges[j].first, alledges[j].second);

        if(!is_connected(my_now_g))
            continue;   

        int flag = 0;
        for(int j = 0;j < mygraphlet.size(); j++)
            if(judge_isomorphism(my_now_g, &mygraphlet[j]))
                flag = 1;

        if(!flag)
            mygraphlet.push_back(*my_now_g);
    }

    int length = mygraphlet.size();
    vector<pair<int,int>>().swap(alledges);

    vector<vector<int>> feature(N, vector<int>(length, 0));

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for(int i = 0;i < N; i++ )
    {
        int len = judge_edge[i].size();
        adj[i] = *(new vector<int>(len));
        vector<double> prob(len);
        int count = 0;
        for (int v : judge_edge[i])
        {
            adj[i][count] = v;
            prob[count] = degree[v];
            count++;
        }
        alias[i] = new Alias(&prob);
    }
    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for(int i = 0; i < N;i++)
    {
        for(int j = 0; j<num_sample; j++)
        {
            vector<int> node_list = {i};
            vector<double> degree_prob = {degree[i]};
            for(int iter = 1; iter < K; iter++)
            {
                int target = node_list[GetRandomNumWithWeight(&degree_prob)];
                int present = alias[target]->draw();
                int next;
                if(present == -1)
                    next = target;
                else
                    next = adj[target][present];
                
                node_list.push_back(next);
                degree_prob.push_back(degree[next]);
            }
            myGraph * now_g = new myGraph(K);

            for(int u = 0 ; u < K; u++)
                for(int v = u + 1; v < K; v++)
                {              
                    if(judge_edge[node_list[u]].find(node_list[v])!=judge_edge[node_list[u]].end())
                        now_g->add_edge(u, v);
                }
            
            for(int idx = 0 ;idx < length; idx++)
                if(judge_isomorphism(now_g, &mygraphlet[idx]))
                {
                    feature[i][idx]++;
                    break;
                }
            vector<int>().swap(node_list);
            vector<double>().swap(degree_prob);
            delete now_g;
        }
    }
    stringstream ss;
    ss <<file_path<<"_"<< "features" << K;
    string now_path = ss.str();
    printf("%s\n", now_path.c_str());

    freopen(now_path.c_str(),"w",stdout);
    for(int i= 0;i < N;i ++)
    {   
        printf("%d", feature[i][0]);
        for(int j=1;j < length;j++)
            printf(" %d", feature[i][j]);
        printf("\n");
    }
    fclose(stdout);

    vector<myGraph>().swap(mygraphlet);
    for(int i = 0; i < N; i++)
    {
        delete alias[i];
        vector<int>().swap(adj[i]);
    }
    vector<Alias *>().swap(alias);
    vector<int>().swap(degree);
    vector<vector<int>>().swap(adj);
}

/*count the node orbits in parttens of 3*/
int Graph::local_transitivity(int thread)
{
    vector<vector<long long>> feature(N, vector<long long>(3, 0));
    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int u = 0; u < N; u++)
    {
        long long nei1 = judge_edge[u].size();
        feature[u][0] = nei1 * (nei1 - 1)/2;

        for (int v : judge_edge[u])
        {
            int nei2 = judge_edge[v].size();
            feature[u][2] = feature[u][2] + nei2;
            for (int w : judge_edge[v])
            {
                if(w==u || w==v)
                    feature[u][2] = feature[u][2] - 1;
                if(w > v)
                {
                    auto iterator = judge_edge[w].find(u);
                    if (iterator != judge_edge[w].end())
                        feature[u][1] = feature[u][1] + 1;
                }
            }   
        }
        //feature[u][0] = feature[u][0] - feature[u][1];
        //feature[u][2] = feature[u][2] - 2 * feature[u][1];
    }

    stringstream ss;
    ss <<file_path<<"_"<< "out3";
    string now_path = ss.str();
    printf("%s\n", now_path.c_str());

    freopen(now_path.c_str(),"w",stdout);
    for(int i= 0;i < N;i ++)
        printf("%lld %lld %lld\n", feature[i][0], feature[i][1], feature[i][2]);
    fclose(stdout);
    return 0;
}