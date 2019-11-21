#include "domains/polyhedra.h"
#include "utils/lipschitz.h"
#include "gurobi_c++.h"
#include <vector>

std::pair<std::vector<double>, double> findLower(GRBEnv env, LipschitzFunction, std::default_random_engine, int, double, int, Statistics& counter);
std::pair<std::vector<double>, double> findUpper(GRBEnv env, LipschitzFunction, std::default_random_engine, int, double, int, Statistics& counter);

