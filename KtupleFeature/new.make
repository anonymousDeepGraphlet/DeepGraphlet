CXXFLAGS = -mcmodel=large
run:run.o graph.o
	g++ -g run.o graph.o -o run $(CXXFLAGS) -fopenmp
run.o:graph.o run.cpp
	g++ -g -c -Wall -std=c++11 $(CXXFLAGS) run.cpp -o run.o -fopenmp
graph.o:graph.cpp
	g++ -g -c -Wall -std=c++11 $(CXXFLAGS)  graph.cpp -o graph.o -fopenmp
clean:
	rm -rf *.o