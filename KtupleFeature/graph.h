#ifndef _GRAPH_H_
#define _GRAPH_H_
#include <vector>
#include <omp.h>
#include <chrono>
#include <set>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <vector>
#include <queue>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
using namespace std;

class myGraph{
    public:
        vector<vector<int>> edges;
        int N;
        myGraph(int nodes);
        ~myGraph();
        void add_edge(int u, int v);
};

class Alias{
    public:
        int len;
        vector<double> Q;
        vector<int> J;
        Alias(vector<double> * weight);
        int draw();
        ~Alias();
};

class Graph{
    public:
        string file_path;
        vector<set<int>> judge_edge;
        long long N;
        long long M;
        Graph(string now_path);
        ~Graph();
        void getKtuples(int K, int num_sample, int thread);
        int local_transitivity(int thread);
};
#endif