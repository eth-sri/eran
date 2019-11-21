#include "domains/dp.h"
#include <iostream>

DeepPoly1d::DeepPoly1d(double lw, double lc, double uw, double uc, Interval delta) {
    this->lw = lw;
    this->lc = lc;
    this->uw = uw;
    this->uc = uc;
    this->delta = delta;
//    printf("DeepPoly: %.3lf * delta + %.3lf <= p <= %.3lf * delta + %.3lf\n", lw, lc, uw, uc);
}

Interval DeepPoly1d::it() const {
    double imin = std::min(delta.inf * lw + lc, delta.sup * lw + lc);
    double imax = std::max(delta.inf * uw + uc, delta.sup * uw + uc);
//    printf("it() %.3lf * delta + %.3lf <= p <= %.3lf * delta + %.3lf\n", lw, lc, uw, uc);
    return {imin, imax};
}

DeepPoly1d DeepPoly1d::operator-() const {
    return {-uw, -uc, -lw, -lc, delta};
}

DeepPoly1d DeepPoly1d::operator+(const DeepPoly1d &other) const {
    return {lw + other.lw, lc + other.lc, uw + other.uw, uc + other.uc, delta};
}

DeepPoly1d DeepPoly1d::operator*(const DeepPoly1d &other) const {
        Interval a_it = it(), b_it = other.it();
//        std::cout << "a_it: " << a_it << std::endl;
//        std::cout << "b_it: " << a_it << std::endl;
        assert(a_it.inf >= -1e-9);
        assert(b_it.inf >= -1e-9);
        if (a_it.sup - a_it.inf < b_it.sup - b_it.inf) {
            return {a_it.inf * other.lw, a_it.inf * other.lc, a_it.sup * other.uw, a_it.sup * other.uc, delta};
        } else {
            return {b_it.inf * lw, b_it.inf * lc, b_it.sup * uw, b_it.sup * uc, delta};
        }
}

DeepPoly1d DeepPoly1d::operator+(const double &scalar) {
    return {lw, lc + scalar, uw, uc + scalar, delta};
}

DeepPoly1d DeepPoly1d::operator-(const double &scalar) {
    return (*this) + (-scalar);
}


DeepPoly1d operator * (const double& c, const DeepPoly1d& dp) {
    if (c > 0) {
        return {dp.lw * c, dp.lc * c, dp.uw * c, dp.uc * c, dp.delta};
    } else {
        return {dp.uw * c, dp.uc * c, dp.lw * c, dp.lc * c, dp.delta};
    }
}