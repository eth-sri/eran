#include "constants.h"

int Constants::NUM_ATTACKS = 1; // Number of attacks to generate
int Constants::POLY_DEGREE = 1; // Degree of the polynomial of the parameters used to fit the polyhedra
double Constants::POLY_EPS = 0.01; // Precision used for the optimization in the polyhedra case

int Constants::NUM_THREADS = 1;
double Constants::MAX_COEFF = 1;
int Constants::LP_SAMPLES = 1000;
int Constants::NUM_POLY_CHECK = 100;

std::string Constants::SPLIT_MODE = "standard"; // standard, gradient, gradientSign
std::string Constants::UB_ESTIMATE = "Triangle"; // CauchySchwarz = CauchySchwarz, Triangle = Triangle estimate

