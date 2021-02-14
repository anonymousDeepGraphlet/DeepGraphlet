//#include <vector>
#include <igraph/igraph.h>
#include "graph.h"
boost::mt19937 gen;

static int igraph_i_create_start(
        igraph_vector_t *res, igraph_vector_t *el,
        igraph_vector_t *iindex, igraph_integer_t nodes) {

# define EDGE(i) (VECTOR(*el)[ (long int) VECTOR(*iindex)[(i)] ])

    long int no_of_nodes;
    long int no_of_edges;
    long int i, j, idx;

    no_of_nodes = nodes;
    no_of_edges = igraph_vector_size(el);

    /* result */

    igraph_vector_resize(res, nodes + 1);

    /* create the index */

    if (igraph_vector_size(el) == 0) {
        /* empty graph */
        igraph_vector_null(res);
    } else {
        idx = -1;
        for (i = 0; i <= EDGE(0); i++) {
            idx++; VECTOR(*res)[idx] = 0;
        }
        for (i = 1; i < no_of_edges; i++) {
            long int n = (long int) (EDGE(i) - EDGE((long int)VECTOR(*res)[idx]));
            for (j = 0; j < n; j++) {
                idx++; VECTOR(*res)[idx] = i;
            }
        }
        j = (long int) EDGE((long int)VECTOR(*res)[idx]);
        for (i = 0; i < no_of_nodes - j; i++) {
            idx++; VECTOR(*res)[idx] = no_of_edges;
        }
    }

    /* clean */

# undef EDGE
    return 0;
}

/*Graph::Graph(int _num_nodes, int _num_edges, py::array_t<int> edges_from, py::array_t<int> edges_to, int directed)
{
    py::buffer_info buf1 = edges_from.request(), buf2 = edges_to.request();
    int *ptr1 = (int *)buf1.ptr,
        *ptr2 = (int *)buf2.ptr;
    
    igraph_empty(&g, _num_nodes, directed);
    igraph_t * graph = &g;

    long int no_of_edges = igraph_vector_size(&graph->from);
    long int edges_to_add = _num_edges;
    long int i = 0;
    igraph_error_handler_t *oldhandler;
    int ret1, ret2;
    igraph_vector_t newoi, newii;

    //IGRAPH_CHECK(igraph_vector_reserve(&graph->from, no_of_edges + edges_to_add));
    //IGRAPH_CHECK(igraph_vector_reserve(&g->to, no_of_edges + edges_to_add));

    while (i < edges_to_add) {
        if (directed || ptr1[i] > ptr2[i]) {
            igraph_vector_push_back(&graph->from, ptr1[i]); 
            igraph_vector_push_back(&graph->to,  ptr2[i]); 
        } else {
            igraph_vector_push_back(&graph->to,   ptr1[i]);
            igraph_vector_push_back(&graph->from, ptr2[i]);
        }
        i++;
    }
  
    oldhandler = igraph_set_error_handler(igraph_error_handler_ignore);

    ret1 = igraph_vector_init(&newoi, no_of_edges);
    ret2 = igraph_vector_init(&newii, no_of_edges);
    if (ret1 != 0 || ret2 != 0) {
        igraph_vector_resize(&graph->from, no_of_edges); 
        igraph_vector_resize(&graph->to, no_of_edges);   
        igraph_set_error_handler(oldhandler);
        //IGRAPH_ERROR("cannot add edges", IGRAPH_ERROR_SELECT_2(ret1, ret2));
    }
    ret1 = igraph_vector_order(&graph->from, &graph->to, &newoi, graph->n);
    ret2 = igraph_vector_order(&graph->to, &graph->from, &newii, graph->n);
    if (ret1 != 0 || ret2 != 0) {
        igraph_vector_resize(&graph->from, no_of_edges);
        igraph_vector_resize(&graph->to, no_of_edges);
        igraph_vector_destroy(&newoi);
        igraph_vector_destroy(&newii);
        igraph_set_error_handler(oldhandler);
        //IGRAPH_ERROR("cannot add edges", IGRAPH_ERROR_SELECT_2(ret1, ret2));
    }

    
    igraph_i_create_start(&graph->os, &graph->from, &newoi, graph->n);
    igraph_i_create_start(&graph->is, &graph->to, &newii, graph->n);

    igraph_vector_destroy(&graph->oi);
    igraph_vector_destroy(&graph->ii);
    graph->oi = newoi;
    graph->ii = newii;
    igraph_set_error_handler(oldhandler);
}*/

Graph::Graph(int _num_nodes, int _num_edges, int *edges_from, int *edges_to, int directed)
{
    /*igraph_vector_t v1;
    igraph_vector_init(&v1, 2 * _num_edges);
    for (int i = 0; i < _num_edges; ++i)
    {
        VECTOR(v1)
        [2 * i] = edges_from[i];
        VECTOR(v1)
        [2 * i + 1] = edges_to[i];
    }
    igraph_create(&g, &v1, 0, directed);
    igraph_vector_destroy(&v1);*/
    for(int i = 0; i < _num_nodes; i++)
    {
        judge_edge.insert({edges_from[i], edges_to[i]});
        judge_edge.insert({edges_to[i], edges_from[i]});
    }
    igraph_empty(&g, _num_nodes, directed);
    igraph_t * graph = &g;

    long int no_of_edges = igraph_vector_size(&graph->from);
    long int edges_to_add = _num_edges;
    long int i = 0;
    igraph_error_handler_t *oldhandler;
    int ret1, ret2;
    igraph_vector_t newoi, newii;

    //IGRAPH_CHECK(igraph_vector_reserve(&graph->from, no_of_edges + edges_to_add));
    //IGRAPH_CHECK(igraph_vector_reserve(&g->to, no_of_edges + edges_to_add));
     
    while (i < edges_to_add) {
        if (directed || edges_from[i] > edges_to[i]) {
            igraph_vector_push_back(&graph->from, edges_from[i]); 
            igraph_vector_push_back(&graph->to, edges_to[i]); 
        } else {
            igraph_vector_push_back(&graph->to,   edges_from[i]);
            igraph_vector_push_back(&graph->from, edges_to[i]);
        }
        i++;
    }
  
    oldhandler = igraph_set_error_handler(igraph_error_handler_ignore);

    ret1 = igraph_vector_init(&newoi, no_of_edges);
    ret2 = igraph_vector_init(&newii, no_of_edges);
    if (ret1 != 0 || ret2 != 0) {
        igraph_vector_resize(&graph->from, no_of_edges); 
        igraph_vector_resize(&graph->to, no_of_edges);   
        igraph_set_error_handler(oldhandler);
        //IGRAPH_ERROR("cannot add edges", IGRAPH_ERROR_SELECT_2(ret1, ret2));
    }
    ret1 = igraph_vector_order(&graph->from, &graph->to, &newoi, graph->n);
    ret2 = igraph_vector_order(&graph->to, &graph->from, &newii, graph->n);
    if (ret1 != 0 || ret2 != 0) {
        igraph_vector_resize(&graph->from, no_of_edges);
        igraph_vector_resize(&graph->to, no_of_edges);
        igraph_vector_destroy(&newoi);
        igraph_vector_destroy(&newii);
        igraph_set_error_handler(oldhandler);
        //IGRAPH_ERROR("cannot add edges", IGRAPH_ERROR_SELECT_2(ret1, ret2));
    }

    
    igraph_i_create_start(&graph->os, &graph->from, &newoi, graph->n);
    igraph_i_create_start(&graph->is, &graph->to, &newii, graph->n);

    igraph_vector_destroy(&graph->oi);
    igraph_vector_destroy(&graph->ii);
    graph->oi = newoi;
    graph->ii = newii;
    igraph_set_error_handler(oldhandler);
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
    igraph_destroy(&g);
}

void myGraph::add_edge(int u, int v)
{
    edges[u].push_back(v);
    edges[v].push_back(u);
}

igraph_vs_t *Graph::neighbor(int v)
{
    igraph_vs_t vs;
    igraph_vit_t vit;
    return &vs;
}

void Graph::nodes()
{
    igraph_vs_t vs;
    igraph_vit_t vit;
    igraph_vs_all(&vs);
    igraph_vit_create(&g, vs, &vit);
    while (!IGRAPH_VIT_END(vit))
    {
        printf("%li ", (long int)IGRAPH_VIT_GET(vit));
        IGRAPH_VIT_NEXT(vit);
    }
}

void Graph::edges()
{
    igraph_es_t es;
    igraph_eit_t eit;
    igraph_es_all(&es, IGRAPH_EDGEORDER_FROM);
    igraph_eit_create(&g, es, &eit);
    while (!IGRAPH_EIT_END(eit))
    {
        igraph_integer_t from, to;
        igraph_edge(&g, IGRAPH_EIT_GET(eit), &from, &to);
        printf("%li %li\n", from, to);
        IGRAPH_EIT_NEXT(eit);
    }
}

int GetRandomNumWithWeight(vector<double> * weight)
{
	int size = weight->size();

	//计算权重的总和
	double accumulateVal = accumulate(weight->begin(), weight->end(), 0);
	
    double tempSum = 0;
    
    int ranIndex = accumulateVal * rand() / double(RAND_MAX);

		//0 ~ weight[0]为1，weight[0]+1 ~ weight[1]为2，依次类推
	for (int j = 0; j < size; j++)
	{
		tempSum += (*weight)[j];
		if (ranIndex <= tempSum + (*weight)[j])
			return j;
	}
}

int Alias::draw()
{
    int now = rand()%len;
    double value = (double)rand() / double(RAND_MAX);
    if(value < Q[now])
        return now;
    else
        return J[now];
}



bool judge_isomorphism(igraph_t *graph1, const igraph_t *graph2)
{
    igraph_bool_t res;
    //igraph_isomorphic(graph1, graph2, &res);
    igraph_isomorphic_bliss(graph1, graph2, NULL, NULL, &res, 0, 0, /*sh=*/ IGRAPH_BLISS_F, 0, 0);
    if(res)
        return true;
    else
        return false;
}

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


void Graph::getKtuples(int K, int num_sample, int thread)
{
    long int N = igraph_vcount(&g);

    igraph_vector_t degree;
    igraph_vector_init(&degree, 0);
    igraph_degree(&g, &degree, igraph_vss_all(), IGRAPH_OUT, IGRAPH_NO_LOOPS);

    igraph_adjlist_t allneis;
    igraph_adjlist_init(&g, &allneis, IGRAPH_ALL);
    igraph_adjlist_simplify(&allneis);

    vector<vector<int>> adj(N);
    vector<vector<double>> prob(N); 
    vector<Alias *> alias(N);

    vector<igraph_t> graphlet;
    vector<myGraph> mygraphlet;

    int maxEdgeCnt = K * (K - 1) / 2;
    int maxEdgeState = 1 << maxEdgeCnt;
    
    vector<pair<int, int>> alledges;
    for(int i = 0;i < K-1; i++)
        for(int j = i + 1 ; j < K; j++ )
            alledges.push_back(make_pair(i, j));

    for(int i = 0; i < maxEdgeState; i++)
    {
        igraph_t now_g;
        igraph_empty(&now_g, K, IGRAPH_UNDIRECTED);
        myGraph * my_now_g = new myGraph(K);
        for(int j = 0; j< maxEdgeCnt; j++)
            if(i & (1<<j)) 
            {
                igraph_add_edge(&now_g, alledges[j].first, alledges[j].second);
                my_now_g->add_edge(alledges[j].first, alledges[j].second);
            }
        igraph_integer_t no;
        igraph_clusters(&now_g, 0, 0, &no, IGRAPH_STRONG);

        if(!(no==1))
            continue;
        //printf("%d\n", i);       

        int flag = 0;
        for(int j = 0;j < graphlet.size(); j++)
            if(judge_isomorphism(my_now_g, &mygraphlet[j]))
                flag = 1;

        if(!flag)
        {
            graphlet.push_back(now_g);
            mygraphlet.push_back(*my_now_g);
            //printf("%d\n", i);
        }
    }

    int length = mygraphlet.size();
    //printf("%d\n", length);
    for(int i = 0; i < length; i++)
        igraph_destroy(&graphlet[i]);
    vector<igraph_t>().swap(graphlet);
    
    //printf("%d", length);

    vector<vector<int>> feature(N, vector<int>(length, 0));

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for(int i = 0;i < N; i++)
    {
        igraph_vector_int_t *neis;
        neis = igraph_adjlist_get(&allneis, i);
        int neilen = igraph_vector_int_size(neis);
        for(int j = 0; j < neilen; j++)
        {
            int v = VECTOR(*neis)[j];
            if(j == 0)
            {
                adj[i] = *(new vector<int>);
                prob[i] = *(new vector<double>);
            }
            adj[i].push_back(v);
            prob[i].push_back(VECTOR(degree)[v]);
        }
        alias[i] = new Alias(&prob[i]);
        //igraph_vector_destroy(neis);
    }

    //for (auto i = 0; i != adj[0].size(); ++i) cout << prob[0][i] << " ";
    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for(int i = 0; i < N;i++)
    {
        printf("%d\n", i);
        for(int j = 0; j<num_sample; j++)
        {
            vector<int> node_list = {i};
            vector<double> degree_prob = {VECTOR(degree)[i]};
            for(int iter = 1; iter < K; iter++)
            {
                int target = node_list[GetRandomNumWithWeight(&degree_prob)];
                //printf("%d %d\n", k, target);
                int next = alias[target]->draw();
                node_list.push_back(next);
                degree_prob.push_back(VECTOR(degree)[next]);
            }
            myGraph * now_g = new myGraph(K);

            for(int u = 0 ; u < K; u++)
                for(int v = u + 1; v < K; v++)
                {              
                    if(judge_edge.find({node_list[u], node_list[v]})!=judge_edge.end())
                        now_g->add_edge(u, v);
                }
            
            for(int idx = 0 ;idx < length; idx++)
                if(judge_isomorphism(now_g, &mygraphlet[idx]))
                {
                    feature[i][idx]++;
                    //printf("%d\n", idx);
                    break;
                }
            delete now_g;
        }
    }

    vector<myGraph>().swap(mygraphlet);
    for(int i = 0; i < N; i++)
    {
        delete alias[i];
        vector<int>().swap(adj[i]);
        vector<double>().swap(prob[i]);
    }
    vector<Alias *>().swap(alias);
    vector<vector<int>>().swap(adj);
    vector<vector<double>>().swap(prob);
    igraph_vector_destroy(&degree);
}

/*py::array_t<int> Graph::degree(int thread, int directed)
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    igraph_degree(&g, &res, igraph_vss_all(), IGRAPH_OUT, IGRAPH_NO_LOOPS);

    long int N = igraph_vcount(&g);

    auto result = py::array_t<int>(N);
    py::buffer_info buf = result.request();
    int* ptr = (int*)buf.ptr;
    for (int i = 0; i < N; i++)
        ptr[i] = VECTOR(res)[i];
    
    return result;
}

py::array_t<double> Graph::PageRank()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    igraph_pagerank_power_options_t power_options;
    power_options.niter = 10000;
    power_options.eps = 0.00001;
    igraph_pagerank(&g, IGRAPH_PAGERANK_ALGO_PRPACK, &res, 0,
                    igraph_vss_all(), 0, 0.85, 0, &power_options);

    long int N = igraph_vcount(&g);

    auto result = py::array_t<double>(N);
    py::buffer_info buf = result.request();
    double* ptr = (double*)buf.ptr;
    for (int i = 0; i < N; i++)
        ptr[i] = VECTOR(res)[i];

    return result;
}*/

int Graph::diameter(int thread, int directed)
{
    igraph_integer_t result = 0;
    auto t0 = std::chrono::steady_clock::now();
    //igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 1);
    my_igraph_diameter(&g, &result, 0, 0, 0, directed, thread, 1);
    //igraph_diameter(&g, &result, 0, 0, 0, directed, 1);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    return result;
}

/*void Graph::closeness()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    //igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 1);
    my_igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_ALL, -1, 1);
}*/

int Graph::my_igraph_diameter(const igraph_t *graph, igraph_integer_t *pres,
                    igraph_integer_t *pfrom, igraph_integer_t *pto,
                    igraph_vector_t *path,
                    igraph_bool_t directed, int thread, igraph_bool_t unconn)
{
    long int no_of_nodes = igraph_vcount(graph);
    long int from = 0, to = 0;
    long int res = 0;

    igraph_neimode_t dirmode;
    igraph_adjlist_t allneis;

    if (directed) {
        dirmode = IGRAPH_OUT;
    } else {
        dirmode = IGRAPH_ALL;
    }

    IGRAPH_CHECK(igraph_adjlist_init(graph, &allneis, dirmode));
    IGRAPH_FINALLY(igraph_adjlist_destroy, &allneis);

    igraph_vector_t result;
    igraph_vector_init(&result, no_of_nodes);

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int i = 0; i < no_of_nodes; i++)
    {
        igraph_vector_t already_counted;
        igraph_vector_init(&already_counted, no_of_nodes);

        igraph_dqueue_t q;
        igraph_dqueue_init(&q, 100);
        igraph_dqueue_clear(&q);

        igraph_dqueue_push(&q, i);
        igraph_dqueue_push(&q, 0);

        long int nodes_reached = 1;
        VECTOR(already_counted)[(long int)i] = i + 1;

        //IGRAPH_ALLOW_INTERRUPTION();

        while (!igraph_dqueue_empty(&q))
        {
            long int act = (long int)igraph_dqueue_pop(&q);
            long int actdist = (long int)igraph_dqueue_pop(&q);
            //printf("%d %li %li\n", now, act, actdist);
            //sum += actdist;
            if(actdist>VECTOR(result)[i])
            {
                VECTOR(result)[i] = actdist;
            }
            /* check the neighbors */
            igraph_vector_int_t *neis;
            neis = igraph_adjlist_get(&allneis, act);
            //printf("299 %li %d\n", act, igraph_vector_int_size(neis));
            for (int j = 0; j < igraph_vector_int_size(neis); j++)
            {
                long int neighbor = (long int)VECTOR(*neis)[j];
                if (VECTOR(already_counted)[neighbor] == i + 1)
                {
                    continue;
                }
                VECTOR(already_counted)[neighbor] = i + 1;
                igraph_dqueue_push(&q, neighbor);
                igraph_dqueue_push(&q, actdist + 1);
            }
        }
        //printf("%li\n", (long int)VECTOR(result)[i]);
        igraph_vector_destroy(&already_counted);
        igraph_dqueue_destroy(&q);
    }

    for(int i = 0; i<no_of_nodes; i++)
        if(VECTOR(result)[i]>(*pres))
            *pres = (igraph_integer_t) VECTOR(result)[i];
    if (pfrom != 0) {
        *pfrom = (igraph_integer_t) from;
    }
    if (pto != 0) {
        *pto = (igraph_integer_t) to;
    }
    if (path != 0) {
        if (res == no_of_nodes) {
            igraph_vector_clear(path);
        } else {
            igraph_vector_ptr_t tmpptr;
            igraph_vector_ptr_init(&tmpptr, 1);
            IGRAPH_FINALLY(igraph_vector_ptr_destroy, &tmpptr);
            VECTOR(tmpptr)[0] = path;
            IGRAPH_CHECK(igraph_get_shortest_paths(graph, &tmpptr, 0,
                                                   (igraph_integer_t) from,
                                                   igraph_vss_1((igraph_integer_t)to),
                                                   dirmode, 0, 0));
            igraph_vector_ptr_destroy(&tmpptr);
            IGRAPH_FINALLY_CLEAN(1);
        }
    }
    /* clean */
    igraph_vector_destroy(&result);
    igraph_adjlist_destroy(&allneis);
    IGRAPH_FINALLY_CLEAN(3);

    return 0;
}

igraph_real_t Graph::transitivity(int thread)
{
    igraph_real_t res;
    auto t0 = std::chrono::steady_clock::now();
    my_igraph_transitivity_undirected(&g, &res, IGRAPH_TRANSITIVITY_ZERO, thread);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    return res;
}

int Graph::my_igraph_transitivity_undirected(const igraph_t *graph,
                                             igraph_real_t *res,
                                             igraph_transitivity_mode_t mode,
                                             int thread)
{
    long int no_of_nodes = igraph_vcount(graph);

    igraph_real_t triples = 0, triangles = 0;
    long int maxdegree;
    igraph_vector_t order;
    igraph_vector_t rank;
    igraph_vector_t degree;

    igraph_adjlist_t allneis;

    IGRAPH_VECTOR_INIT_FINALLY(&order, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&degree, no_of_nodes);

    IGRAPH_CHECK(igraph_degree(graph, &degree, igraph_vss_all(), IGRAPH_ALL,
                               IGRAPH_LOOPS));
    maxdegree = (long int)igraph_vector_max(&degree) + 1;
    igraph_vector_order1(&degree, &order, maxdegree);
    igraph_vector_destroy(&degree);
    IGRAPH_FINALLY_CLEAN(1);
    IGRAPH_VECTOR_INIT_FINALLY(&rank, no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
    {
        VECTOR(rank)
        [(long int)VECTOR(order)[i]] = no_of_nodes - i - 1;
    }

    IGRAPH_CHECK(igraph_adjlist_init(graph, &allneis, IGRAPH_ALL));
    IGRAPH_FINALLY(igraph_adjlist_destroy, &allneis);
    IGRAPH_CHECK(igraph_adjlist_simplify(&allneis));

    //igraph_set_error_handler(igraph_error_handler_ignore);

#pragma omp parallel for num_threads(thread) reduction(+ \
                                                   : triangles, triples) schedule(dynamic, 1)
    for (int nn = no_of_nodes - 1; nn >= 0; nn--)
    {
        igraph_vector_int_t *neis1, *neis2, *neis3;
        long int neilen1, neilen2, neilen3, node;
        node = (long int)VECTOR(order)[nn];

        //IGRAPH_ALLOW_INTERRUPTION();

        neis1 = igraph_adjlist_get(&allneis, node);
        neilen1 = igraph_vector_int_size(neis1);
        triples += (double)neilen1 * (neilen1 - 1);

        std::unordered_map<long int, int> judge(neilen1);
        for (int i = 0; i < neilen1; i++)
            judge.insert(std::pair<long int, int>{(long int)VECTOR(*neis1)[i], 1});

        for (int i = 0; i < neilen1; i++)
        {
            //printf("%d\n", igraph_vector_int_size(neis1));
            long int nei = (long int)VECTOR(*neis1)[i];
            //printf("%d %d\n", node, nei);
            if (VECTOR(rank)[nei] > VECTOR(rank)[node])
            {
                neis2 = igraph_adjlist_get(&allneis, nei);
                neilen2 = igraph_vector_int_size(neis2);
                for (int j = 0; j < neilen2; j++)
                {
                    long int nei2 = (long int)VECTOR(*neis2)[j];
                    if (VECTOR(rank)[nei2] > VECTOR(rank)[nei])
                    {
                        //igraph_set_error_handler(igraph_error_handler_ignore);
                        //igraph_integer_t eid = -1;
                        //FIND_UNDIRECTED_EDGE(graph, node, nei2, &eid);
                        auto iterator = judge.find(nei2);
                        if (iterator != judge.end())
                            triangles += 1.0;
                        //if (eid != -1) {triangles += 1.0;}
                    }
                }
            }
        }
    }

    igraph_adjlist_destroy(&allneis);
    igraph_vector_destroy(&rank);
    igraph_vector_destroy(&order);
    IGRAPH_FINALLY_CLEAN(4);

    if (triples == 0 && mode == IGRAPH_TRANSITIVITY_ZERO)
    {
        *res = 0;
    }
    else
    {
        *res = triangles / triples * 6.0;
    }
    return 0;
}

/*py::array_t<double> Graph::local_transitivity(int thread)
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    auto t0 = std::chrono::steady_clock::now();
    my_igraph_local_transitivity_undirected(&g, &res, IGRAPH_TRANSITIVITY_ZERO, thread);
    //igraph_transitivity_local_undirected(&g, &res, igraph_vss_all(),IGRAPH_TRANSITIVITY_NAN);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    long int no_of_nodes = igraph_vcount(&g);
    auto result = py::array_t<double>(no_of_nodes);
    py::buffer_info buf = result.request();
    double *ptr = (double *)buf.ptr;
    for (size_t idx = 0; idx < no_of_nodes; idx++)
        ptr[idx] = VECTOR(res)[idx];
    return result;
}*/

int Graph::my_igraph_local_transitivity_undirected(const igraph_t *graph,
                                             igraph_vector_t *res,
                                             igraph_transitivity_mode_t mode,
                                             int thread)
{
    long int no_of_nodes = igraph_vcount(graph);

    long int maxdegree;
    igraph_vector_t order;
    igraph_vector_t rank;
    igraph_vector_t degree;

    igraph_adjlist_t allneis;

    IGRAPH_VECTOR_INIT_FINALLY(&order, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&degree, no_of_nodes);

    IGRAPH_CHECK(igraph_degree(graph, &degree, igraph_vss_all(), IGRAPH_ALL,
                               IGRAPH_LOOPS));
    maxdegree = (long int)igraph_vector_max(&degree) + 1;
    igraph_vector_order1(&degree, &order, maxdegree);
    igraph_vector_destroy(&degree);
    IGRAPH_FINALLY_CLEAN(1);
    IGRAPH_VECTOR_INIT_FINALLY(&rank, no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
    {
        VECTOR(rank)[(long int)VECTOR(order)[i]] = no_of_nodes - i - 1;
    }

    IGRAPH_CHECK(igraph_vector_resize(res, no_of_nodes));
    igraph_vector_null(res);

    IGRAPH_CHECK(igraph_adjlist_init(graph, &allneis, IGRAPH_ALL));
    IGRAPH_FINALLY(igraph_adjlist_destroy, &allneis);
    IGRAPH_CHECK(igraph_adjlist_simplify(&allneis));

    //igraph_set_error_handler(igraph_error_handler_ignore);

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int nn = no_of_nodes - 1; nn >= 0; nn--)
    {
        igraph_vector_int_t *neis1, *neis2, *neis3;
        long int neilen1, neilen2, neilen3, node;
        node = (long int)VECTOR(order)[nn];
        neis1 = igraph_adjlist_get(&allneis, node);
        neilen1 = igraph_vector_int_size(neis1);

        std::unordered_map<long int, int> judge(neilen1);
        for (int i = 0; i < neilen1; i++)
            judge.insert(std::pair<long int, int>{(long int)VECTOR(*neis1)[i], 1});

        for (int i = 0; i < neilen1; i++)
        {
            //printf("%d\n", igraph_vector_int_size(neis1));
            long int nei = (long int)VECTOR(*neis1)[i];
            //printf("%d %d\n", node, nei);
            if (VECTOR(rank)[nei] > VECTOR(rank)[node])
            {
                neis2 = igraph_adjlist_get(&allneis, nei);
                neilen2 = igraph_vector_int_size(neis2);
                for (int j = 0; j < neilen2; j++)
                {
                    long int nei2 = (long int)VECTOR(*neis2)[j];
                    if (VECTOR(rank)[nei2] > VECTOR(rank)[nei])
                    {
                        //igraph_set_error_handler(igraph_error_handler_ignore);
                        //igraph_integer_t eid = -1;
                        //FIND_UNDIRECTED_EDGE(graph, node, nei2, &eid);
                        auto iterator = judge.find(nei2);
                        if (iterator != judge.end())
                        {
                            #pragma omp atomic
                            VECTOR(*res)[node] = VECTOR(*res)[node] + 1;
                            #pragma omp atomic
                            VECTOR(*res)[nei] = VECTOR(*res)[nei] + 1;
                            #pragma omp atomic
                            VECTOR(*res)[nei2] = VECTOR(*res)[nei2] + 1;
                        }
                        //if (eid != -1) {triangles += 1.0;}
                    }
                }
            }
        }
    }

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int nn = no_of_nodes - 1; nn >= 0; nn--)
    {
        igraph_real_t triples = 0;
        long int neilen1, node;
        igraph_vector_int_t *neis1;
        
        node = (long int)VECTOR(order)[nn];
        neis1 = igraph_adjlist_get(&allneis, node);
        neilen1 = igraph_vector_int_size(neis1);

        triples = (double)neilen1 * (neilen1 - 1)/2;
        if(triples)
            VECTOR(*res)[node] = (double)VECTOR(*res)[node]/triples;
        else
            VECTOR(*res)[node] = 0;
    }

    igraph_adjlist_destroy(&allneis);
    igraph_vector_destroy(&rank);
    igraph_vector_destroy(&order);
    IGRAPH_FINALLY_CLEAN(4);
    
    return 0;
}

/*py::array_t<int> Graph::triadic()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    igraph_triad_census(&g, &res);
    auto result = py::array_t<int>(16);
    py::buffer_info buf = result.request();
    int *ptr = (int *) buf.ptr;
    for (size_t idx = 0; idx < 16; idx++)
        ptr[idx] = VECTOR(res)[idx];
    return result;
}*/

/*py::array_t<double> Graph::closeness(int thread, int directed)
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    auto t0 = std::chrono::steady_clock::now();
    //igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 1);
    my_igraph_closeness_estimate(&g, &res, igraph_vss_all(), directed, thread, -1, 1);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    long int no_of_nodes = igraph_vcount(&g);
    auto result = py::array_t<double>(no_of_nodes);
    py::buffer_info buf = result.request();
    double *ptr = (double *)buf.ptr;
    for (size_t idx = 0; idx < no_of_nodes; idx++)
        ptr[idx] = VECTOR(res)[idx];
    return result;
}*/

/*void Graph::closeness()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    //igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 1);
    my_igraph_closeness_estimate(&g, &res, igraph_vss_all(), IGRAPH_ALL, -1, 1);
}*/

int Graph::my_igraph_closeness_estimate(const igraph_t *graph, igraph_vector_t *res,
                                        const igraph_vs_t vids, int mode,
                                        int thread,
                                        igraph_real_t cutoff,
                                        igraph_bool_t normalized)
{

    long int no_of_nodes = igraph_vcount(graph);

    igraph_adjlist_t allneis;

    long int nodes_to_calc;
    igraph_vit_t vit;

    igraph_bool_t warning_shown = 0;

    IGRAPH_CHECK(igraph_vit_create(graph, vids, &vit));
    IGRAPH_FINALLY(igraph_vit_destroy, &vit);

    nodes_to_calc = IGRAPH_VIT_SIZE(vit);

    if(mode == 0)
        igraph_adjlist_init(graph, &allneis, IGRAPH_ALL);
    if(mode>0)
        igraph_adjlist_init(graph, &allneis, IGRAPH_OUT);
    //IGRAPH_CHECK(igraph_adjlist_init(graph, &allneis, mode));

    IGRAPH_CHECK(igraph_vector_resize(res, nodes_to_calc));
    igraph_vector_null(res);

    igraph_vector_t node_list;
    igraph_vector_init(&node_list, nodes_to_calc);
    IGRAPH_VIT_RESET(vit);
    for (int i = 0; !IGRAPH_VIT_END(vit); IGRAPH_VIT_NEXT(vit), i++)
        VECTOR(node_list)[i] = IGRAPH_VIT_GET(vit);

    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int i = 0; i < nodes_to_calc; i++)
    {
        //double sum = 0;

        long int actdist;

        igraph_vector_t already_counted;
        igraph_vector_init(&already_counted, no_of_nodes);

        igraph_dqueue_t q;
        igraph_dqueue_init(&q, 100);
        igraph_dqueue_clear(&q);

        int now = VECTOR(node_list)[i];
        igraph_dqueue_push(&q, now);
        igraph_dqueue_push(&q, 0);

        long int nodes_reached = 1;
        VECTOR(already_counted)[(long int)now] = i + 1;

        //IGRAPH_ALLOW_INTERRUPTION();

        while (!igraph_dqueue_empty(&q))
        {
            long int act = (long int)igraph_dqueue_pop(&q);
            actdist = (long int)igraph_dqueue_pop(&q);
            //printf("%d %li %li\n", now, act, actdist);
            //sum += actdist;
            VECTOR(*res)[i] += actdist;

            if (cutoff > 0 && actdist >= cutoff)
            {
                continue; /* NOT break!!! */
            }

            /* check the neighbors */
            igraph_vector_int_t *neis;
            neis = igraph_adjlist_get(&allneis, act);
            //printf("299 %li %d\n", act, igraph_vector_int_size(neis));
            for (int j = 0; j < igraph_vector_int_size(neis); j++)
            {
                long int neighbor = (long int)VECTOR(*neis)[j];
                if (VECTOR(already_counted)[neighbor] == i + 1)
                {
                    continue;
                }
                VECTOR(already_counted)
                [neighbor] = i + 1;
                nodes_reached++;
                igraph_dqueue_push(&q, neighbor);
                igraph_dqueue_push(&q, actdist + 1);
            }
        }

        /* using igraph_real_t here instead of igraph_integer_t to avoid overflow */
        VECTOR(*res)[i] += ((igraph_real_t)no_of_nodes * (no_of_nodes - nodes_reached));
        VECTOR(*res)[i] = (no_of_nodes - 1) / VECTOR(*res)[i];

        igraph_vector_destroy(&already_counted);
        igraph_dqueue_destroy(&q);
    }

    if (!normalized)
    {
        for (int i = 0; i < nodes_to_calc; i++)
        {
            VECTOR(*res)
            [i] /= (no_of_nodes - 1);
            //printf("%lf\n", VECTOR(*res)[i]);
        }
    }

    /* Clean */
    igraph_vit_destroy(&vit);
    igraph_adjlist_destroy(&allneis);
    IGRAPH_FINALLY_CLEAN(4);

    return 0;
}

/*igraph_vector_t* Graph::triadic()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    igraph_triad_census(&g, &res);
    return 
}*/

/*py::array_t<double> Graph::harmonic(int thread, int directed)
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    auto t0 = std::chrono::steady_clock::now();
    //igraph_harmonic_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 1);
    my_igraph_harmonic_estimate(&g, &res, igraph_vss_all(),directed, thread, -1);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    long int no_of_nodes = igraph_vcount(&g);
    auto result = py::array_t<double>(no_of_nodes);
    py::buffer_info buf = result.request();
    double *ptr = (double *)buf.ptr;
    for (size_t idx = 0; idx < no_of_nodes; idx++)
        ptr[idx] = VECTOR(res)[idx];
    return result;
}*/

int Graph::my_igraph_harmonic_estimate(const igraph_t *graph, igraph_vector_t *res,
                                        const igraph_vs_t vids, int mode,
                                        int thread,
                                        igraph_real_t cutoff)
{

    long int no_of_nodes = igraph_vcount(graph);

    igraph_adjlist_t allneis;

    long int nodes_to_calc;
    igraph_vit_t vit;

    igraph_bool_t warning_shown = 0;

    IGRAPH_CHECK(igraph_vit_create(graph, vids, &vit));
    IGRAPH_FINALLY(igraph_vit_destroy, &vit);

    nodes_to_calc = IGRAPH_VIT_SIZE(vit);

    if(mode == 0)
        igraph_adjlist_init(graph, &allneis, IGRAPH_ALL);
    if(mode>0)
        igraph_adjlist_init(graph, &allneis, IGRAPH_OUT);
    

    IGRAPH_CHECK(igraph_vector_resize(res, nodes_to_calc));
    igraph_vector_null(res);

    igraph_vector_t node_list;
    igraph_vector_init(&node_list, nodes_to_calc);
    IGRAPH_VIT_RESET(vit);
    for (int i = 0; !IGRAPH_VIT_END(vit); IGRAPH_VIT_NEXT(vit), i++)
    {
        VECTOR(node_list)[i] = IGRAPH_VIT_GET(vit);
        //printf("%d %li\n", i, IGRAPH_VIT_GET(vit));
    }
    #pragma omp parallel for num_threads(thread) schedule(dynamic, 1)
    for (int i = 0; i < nodes_to_calc; i++)
    {
        //double sum = 0;

        long int actdist;

        igraph_vector_t already_counted;
        igraph_vector_init(&already_counted, no_of_nodes);

        igraph_dqueue_t q;
        igraph_dqueue_init(&q, 100);
        igraph_dqueue_clear(&q);

        int now = VECTOR(node_list)[i];
        igraph_dqueue_push(&q, now);
        igraph_dqueue_push(&q, 0);

        long int nodes_reached = 1;
        VECTOR(already_counted)
        [(long int)now] = i + 1;

        //IGRAPH_ALLOW_INTERRUPTION();

        while (!igraph_dqueue_empty(&q))
        {
            long int act = (long int)igraph_dqueue_pop(&q);
            actdist = (long int)igraph_dqueue_pop(&q);
            //printf("%d %li %li\n", now, act, actdist);
            //sum += actdist;
            if(actdist!=0)
                VECTOR(*res)[i] += (double)1/(double)actdist;

            if (cutoff > 0 && actdist >= cutoff)
            {
                continue; /* NOT break!!! */
            }

            /* check the neighbors */
            igraph_vector_int_t *neis;
            neis = igraph_adjlist_get(&allneis, act);
            //printf("299 %li %d\n", act, igraph_vector_int_size(neis));
            for (int j = 0; j < igraph_vector_int_size(neis); j++)
            {
                long int neighbor = (long int)VECTOR(*neis)[j];
                if (VECTOR(already_counted)[neighbor] == i + 1)
                {
                    continue;
                }
                VECTOR(already_counted)[neighbor] = i + 1;
                nodes_reached++;
                igraph_dqueue_push(&q, neighbor);
                igraph_dqueue_push(&q, actdist + 1);
            }
        }

        /* using igraph_real_t here instead of igraph_integer_t to avoid overflow */
        //VECTOR(*res)[i] += ((igraph_real_t)no_of_nodes * (no_of_nodes - nodes_reached));
        //VECTOR(*res)[i] = (no_of_nodes - 1) / VECTOR(*res)[i];

        igraph_vector_destroy(&already_counted);
        igraph_dqueue_destroy(&q);
    }

    /* Clean */
    igraph_vit_destroy(&vit);
    igraph_adjlist_destroy(&allneis);
    IGRAPH_FINALLY_CLEAN(4);

    return 0;
}

/*py::array_t<double> Graph::betweenness(int thread, int directed)
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    auto t0 = std::chrono::steady_clock::now();
    //igraph_betweenness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 0);
    my_igraph_betweenness_estimate(&g, &res, igraph_vss_all(), directed, thread, -1, 0);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    long int no_of_nodes = igraph_vcount(&g);
    auto result = py::array_t<double>(no_of_nodes);
    py::buffer_info buf = result.request();
    double *ptr = (double *)buf.ptr;
    for (size_t idx = 0; idx < no_of_nodes; idx++)
        ptr[idx] = VECTOR(res)[idx];
    return result;
}*/

/*void Graph::betweenness()
{
    igraph_vector_t res;
    igraph_vector_init(&res, 0);
    auto t0 = std::chrono::steady_clock::now();
    //igraph_betweenness_estimate(&g, &res, igraph_vss_all(), IGRAPH_OUT, -1, 0, 0);
    my_igraph_betweenness_estimate(&g, &res, igraph_vss_all(), 0, 10, -1, 1);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
              << '\n';
    for(int i=0; i< 8;i++)
        printf("%lf\n", VECTOR(res)[i]);
}*/

int Graph::my_igraph_betweenness_estimate(const igraph_t *graph, igraph_vector_t *res,
                                          const igraph_vs_t vids, igraph_bool_t directed,
                                          int thread,
                                          igraph_real_t cutoff,
                                          igraph_bool_t nobigint)
{

    long int no_of_nodes = igraph_vcount(graph);

    /*unsigned long long int *nrgeo = 0;
    igraph_biguint_t *big_nrgeo = 0;*/
    igraph_vector_t v_tmpres, *tmpres = &v_tmpres;
    igraph_vit_t vit;

    igraph_adjlist_t adjlist_out, adjlist_in;
    igraph_adjlist_t *adjlist_out_p;

    /* Ensure that 0 is interpreted as infinity in the igraph 0.8 series. TODO: remove for 0.9. */
    if (cutoff == 0)
    {
        cutoff = -1;
    }

    /*if (weights) {
        return igraph_i_betweenness_estimate_weighted(graph, res, vids, directed,
                cutoff, weights, nobigint);
    }*/

    if (!igraph_vs_is_all(&vids))
    {
        /* subset */
        IGRAPH_VECTOR_INIT_FINALLY(tmpres, no_of_nodes);
    }
    else
    {
        /* only  */
        IGRAPH_CHECK(igraph_vector_resize(res, no_of_nodes));
        igraph_vector_null(res);
        tmpres = res;
    }

    directed = directed && igraph_is_directed(graph);
    if (directed)
    {
        IGRAPH_CHECK(igraph_adjlist_init(graph, &adjlist_out, IGRAPH_OUT));
        IGRAPH_FINALLY(igraph_adjlist_destroy, &adjlist_out);
        IGRAPH_CHECK(igraph_adjlist_init(graph, &adjlist_in, IGRAPH_IN));
        IGRAPH_FINALLY(igraph_adjlist_destroy, &adjlist_in);
        adjlist_out_p = &adjlist_out;
    }
    else
    {
        IGRAPH_CHECK(igraph_adjlist_init(graph, &adjlist_out, IGRAPH_ALL));
        IGRAPH_FINALLY(igraph_adjlist_destroy, &adjlist_out);
        IGRAPH_CHECK(igraph_adjlist_init(graph, &adjlist_in, IGRAPH_ALL));
        IGRAPH_FINALLY(igraph_adjlist_destroy, &adjlist_in);
        adjlist_out_p = &adjlist_out;
    }
    /*for (j = 0; j < no_of_nodes; j++)
    {
        igraph_vector_int_clear(igraph_adjlist_get(adjlist_in_p, j));
    }

    if (nobigint) {
        nrgeo = igraph_Calloc(no_of_nodes, unsigned long long int);
        if (nrgeo == 0) {
            IGRAPH_ERROR("betweenness failed", IGRAPH_ENOMEM);
        }
        IGRAPH_FINALLY(igraph_free, nrgeo);
    } else {
        big_nrgeo = igraph_Calloc(no_of_nodes + 1, igraph_biguint_t);
        if (!big_nrgeo) {
            IGRAPH_ERROR("betweenness failed", IGRAPH_ENOMEM);
        }
        IGRAPH_FINALLY(igraph_i_destroy_biguints, big_nrgeo);
        for (j = 0; j < no_of_nodes; j++) {
            IGRAPH_CHECK(igraph_biguint_init(&big_nrgeo[j]));
        }
        IGRAPH_CHECK(igraph_biguint_init(&D));
        IGRAPH_FINALLY(igraph_biguint_destroy, &D);
        IGRAPH_CHECK(igraph_biguint_init(&R));
        IGRAPH_FINALLY(igraph_biguint_destroy, &R);
        IGRAPH_CHECK(igraph_biguint_init(&T));
        IGRAPH_FINALLY(igraph_biguint_destroy, &T);
    }*/

    /* here we go */
    #pragma omp parallel for num_threads(thread)
    for (int source = 0; source < no_of_nodes; source++)
    {
        //printf("%d\n", source);
        double *tmpscore;
        tmpscore = igraph_Calloc(no_of_nodes, double);

        long int *distance;
        distance = igraph_Calloc(no_of_nodes, long int);

        igraph_dqueue_t q = IGRAPH_DQUEUE_NULL;
        igraph_dqueue_init(&q, 100);

        igraph_adjlist_t adjlist_in_p;
        igraph_adjlist_init_empty(&adjlist_in_p, no_of_nodes);
        mpf_t* nrgeo = new mpf_t[no_of_nodes] ;//malloc(no_of_nodes, sizeof(mpf_t));

        for (int i = 0; i < no_of_nodes; i++)
        {
            mpf_init(nrgeo[i]);
            if (i == source)
                mpf_set_ui(nrgeo[i], 1);
            else
                mpf_set_ui(nrgeo[i], 0);
        }
        igraph_stack_t stack = IGRAPH_STACK_NULL;
        igraph_stack_init(&stack, no_of_nodes);
        //IGRAPH_FINALLY(igraph_stack_destroy, &stack);
        igraph_dqueue_push(&q, source);

        distance[source] = 1;

        while (!igraph_dqueue_empty(&q))
        {
            long int actnode = (long int)igraph_dqueue_pop(&q);

            /*if (cutoff >= 0 && distance[actnode] > cutoff + 1)
                {
                    distance[actnode] = 0;

                    mpf_set_ui(nrgeo[actnode], 0);

                    tmpscore[actnode] = 0;
                    igraph_vector_int_clear(igraph_adjlist_get(&adjlist_in_p, actnode));
                    continue;
                }*/

            igraph_stack_push(&stack, actnode);
            igraph_vector_int_t *neis;
            neis = igraph_adjlist_get(adjlist_out_p, actnode);
            int nneis = igraph_vector_int_size(neis);
            for (int j = 0; j < nneis; j++)
            {
                long int neighbor = (long int)VECTOR(*neis)[j];
                if (distance[neighbor] == 0)
                {
                    distance[neighbor] = distance[actnode] + 1;
                    igraph_dqueue_push(&q, neighbor);
                }
                if (distance[neighbor] == distance[actnode] + 1 &&
                    (distance[neighbor] <= cutoff + 1 || cutoff < 0))
                {
                    /* Only add if the node is not more distant than the cutoff */
                    igraph_vector_int_t *v = igraph_adjlist_get(&adjlist_in_p,
                                                                neighbor);
                    igraph_vector_int_push_back(v, actnode);

                    mpf_add(nrgeo[neighbor], nrgeo[neighbor], nrgeo[actnode]);

                    /*if (nobigint) {
                        nrgeo[neighbor] += nrgeo[actnode];
                    } else {
                        IGRAPH_CHECK(igraph_biguint_add(&big_nrgeo[neighbor],
                                                        &big_nrgeo[neighbor],
                                                        &big_nrgeo[actnode]));
                    }*/
                }
            }
        } /* while !igraph_dqueue_empty */
        /* Ok, we've the distance of each node and also the number of
           shortest paths to them. Now we do an inverse search, starting
           with the farthest nodes. */
        while (!igraph_stack_empty(&stack))
        {
            long int actnode = (long int)igraph_stack_pop(&stack);
            igraph_vector_int_t *neis;
            neis = igraph_adjlist_get(&adjlist_in_p, actnode);
            int nneis = igraph_vector_int_size(neis);
            for (int j = 0; j < nneis; j++)
            {
                long int neighbor = (long int)VECTOR(*neis)[j];
                if (mpf_cmp_ui(nrgeo[actnode], 0) != 0)
                {
                    mpf_t present;
                    mpf_init(present);
                    mpf_div(present, nrgeo[neighbor], nrgeo[actnode]);
                    //printf("%lf\n", mpf_get_d(present));
                    tmpscore[neighbor] += (tmpscore[actnode] + 1) *
                                          (double)mpf_get_d(present);
                    mpf_clear(present);
                }
                else
                    //tmpscore[neighbor] = IGRAPH_INFINITY;
                    tmpscore[neighbor] = 0;
            }

            if (actnode != source)
            {
                #pragma omp atomic
                VECTOR(*tmpres)[actnode] = VECTOR(*tmpres)[actnode] + tmpscore[actnode];
            }

            /* Reset variables */
            //igraph_vector_int_clear(igraph_adjlist_get(adjlist_in_p, actnode));
        }
        //printf("2\n");
        for (int i = 0; i < no_of_nodes; i++)
            mpf_clear(nrgeo[i]);
        
        //delete [] nrgeo;

        igraph_adjlist_destroy(&adjlist_in_p);
        igraph_Free(distance);
        igraph_dqueue_destroy(&q);
        igraph_stack_destroy(&stack);
        igraph_Free(tmpscore);

    } /* for source < no_of_nodes */

    /* Keep only the requested vertices */
    /*if (!igraph_vs_is_all(&vids))
    {
        IGRAPH_CHECK(igraph_vit_create(graph, vids, &vit));
        IGRAPH_FINALLY(igraph_vit_destroy, &vit);
        IGRAPH_CHECK(igraph_vector_resize(res, IGRAPH_VIT_SIZE(vit)));
        int k;
        for (k = 0, IGRAPH_VIT_RESET(vit); !IGRAPH_VIT_END(vit);
             IGRAPH_VIT_NEXT(vit), k++)
        {
            long int node = IGRAPH_VIT_GET(vit);
            VECTOR(*res)
            [k] = VECTOR(*tmpres)[node];
        }

        igraph_vit_destroy(&vit);
        igraph_vector_destroy(tmpres);
    }*/

    /* divide by 2 for undirected graph */
    if (!directed)
    {
        int nneis = igraph_vector_size(res);
        for (int j = 0; j < nneis; j++)
        {
            VECTOR(*res)[j] /= 2.0;
        }
    }
    /*for(int i = 0; i < no_of_nodes; i++)
        printf("%lf\n", VECTOR(*tmpres)[i]);*/

    igraph_adjlist_destroy(&adjlist_out);
    igraph_adjlist_destroy(&adjlist_in);

    return 0;
}
