#include "domains/polyhedra.h"
#include "utils/constants.h"
// #include "utils/math.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>

Polyhedra::Polyhedra(Interval it,
                     std::vector<double> wLower, double biasLower,
                     std::vector<double> wUpper, double biasUpper,
                     int degree) {
  assert(wLower.size() == wUpper.size());
  this->it = it;
  this->wLower = wLower;
  this->biasLower = biasLower;
  this->wUpper = wUpper;
  this->biasUpper = biasUpper;
  this->dim = wLower.size();
  this->degree = degree;
}

Interval Polyhedra::evaluate(PointD p) {
  assert(p.x.size() * degree == wLower.size() && p.x.size() * degree == wUpper.size());
  double lb = biasLower, ub = biasUpper;
  for (size_t i = 0; i < p.x.size(); ++i) {
    for (int j = 0; j < degree; ++j) {
        lb += wLower[i * degree + j] * pow(p.x[i], j + 1);
        ub += wUpper[i * degree + j] * pow(p.x[i], j + 1);
    }
  }
  return {lb, ub};
}


std::vector<double> Polyhedra::to_vector() {
  std::vector<double> result;
  result.push_back(biasLower);
  for (size_t i = 0; i < wLower.size(); ++i) {
    result.push_back(wLower[i]);
  }
  result.push_back(biasUpper);
  for (size_t i = 0; i < wUpper.size(); ++i) {
    result.push_back(wUpper[i]);
  }
  return result;
}


std::ostream& operator << (std::ostream& os, const Polyhedra& p) {
  os << p.biasLower;
  for (size_t i = 0; i < p.wLower.size(); ++i) {
    os << " " << p.wLower[i];
  }
  os << " | ";
  os << p.biasUpper;
  for (size_t i = 0; i < p.wUpper.size(); ++i) {
    os << " " << p.wUpper[i];
  }
  return os;
}
