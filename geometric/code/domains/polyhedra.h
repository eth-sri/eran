#include "domains/interval.h"
#include "utils/lipschitz.h"
#include <ostream>
#include <vector>

#pragma once

class Polyhedra {

public:
    Interval it;
    std::vector<double> wLower, wUpper;
    double biasLower, biasUpper;
    size_t dim;
    int degree;

    Polyhedra() {}
    
    Polyhedra(Interval it,
	      std::vector<double> wLower, double biasLower,
	      std::vector<double> wUpper, double biasUpper,
	      int degree);

    Interval evaluate(PointD p);
    std::vector<double> to_vector();
};

std::ostream& operator << (std::ostream& os, const Polyhedra& p);

