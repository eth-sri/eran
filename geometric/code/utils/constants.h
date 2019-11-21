#include <string>
#pragma once

namespace Constants {
    const double EPS = 1e-9; // Precision used to compare floating point numbers
    const int MAX_ZONO_TERMS = 4;

    extern int NUM_ATTACKS; // Number of attacks to generate
    extern int POLY_DEGREE; // Degree of the polynomial of the parameters used to fit the polyhedra
    extern double POLY_EPS; // Precision used for the optimization in the polyhedra case

    extern int NUM_THREADS; // Number of worker threads
    extern double MAX_COEFF; // Maximum magnitude of the coefficients
    extern int LP_SAMPLES; // Number of samples in linear program
    extern int NUM_POLY_CHECK; // Number of samples to check for polyhedra soundness

    extern std::string SPLIT_MODE; // standard, gradient, gradientSign
    extern std::string UB_ESTIMATE; // CauchySchwarz = CauchySchwarz, Triangle = Triangle estimate
};