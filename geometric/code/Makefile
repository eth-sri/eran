GUROBI_FLAGS = -m64 -g -I${GUROBI_HOME}/include -L${GUROBI_HOME}/lib -lgurobi_c++ -lgurobi81 -lm

TRANSFORMS = $(wildcard transforms/*.cpp)
UTILS = $(wildcard utils/*.cpp)
DOMAINS = $(wildcard domains/*.cpp)
ABSTRACTION = $(wildcard abstraction/*.cpp)

shared_object: geometric_constraints.cpp
	g++ -Wl,-soname,libgeometric.so -o build/libgeometric.so -shared -fPIC -O3 -march=native -mavx2 -mfma geometric_constraints.h geometric_constraints.cpp $(UTILS) $(DOMAINS) $(TRANSFORMS) $(ABSTRACTION) -I${CURDIR} $(GUROBI_FLAGS) -std=c++11 -lpthread

deepg_constraints: deepg_constraints.cpp
	g++ -o deepg_constraints.o -O3 -march=native -mavx2 -mfma deepg_constraints.h deepg_constraints.cpp $(UTILS) $(DOMAINS) $(TRANSFORMS) $(ABSTRACTION) -I${CURDIR} $(GUROBI_FLAGS) -std=c++11 -lpthread

clean:
	rm *.so
