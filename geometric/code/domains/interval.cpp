#include "domains/interval.h"
#include "utils/constants.h"
#include <limits>
#include <iostream>
#include <cassert>
#include <cmath>

Interval::Interval(double inf, double sup) {
    assert(inf <= sup + Constants::EPS);
    if (inf > sup) {
        std::swap(inf, sup);
    }
    this->inf = inf;
    this->sup = sup;
}

Interval& Interval::operator += (const Interval &other) {
    this->inf += other.inf;
    this->sup += other.sup;
    return (*this);
}

Interval::Interval() {
    this->inf = std::numeric_limits<double>::infinity();
    this->sup = -std::numeric_limits<double>::infinity();
}

Interval Interval::getR() {
    return {-std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
}

bool Interval::is_empty() const {
    return this->inf == std::numeric_limits<double>::infinity() &&
           this->sup == -std::numeric_limits<double>::infinity();
}

Interval abs(const Interval& it) {
  if (it.sup < 0) {
    return Interval(-it.sup, -it.inf);
  } else if (it.inf > 0) {
    return Interval(it.inf, it.sup);
  }
  return {0, std::max(-it.inf, it.sup)};
}

Interval normalizeAngle(Interval phi) {
    Interval ret = phi;
    while (ret.inf > M_PI) {
        ret = ret + (-2 * M_PI);
    }
    while (ret.inf < -M_PI) {
        ret = ret + (2 * M_PI);
    }
    return ret;
}

bool Interval::contains(double x) const {
  return x >= inf - Constants::EPS && x <= sup + Constants::EPS;
}

Interval Interval::cosine() const {
  if (sup - inf >= 2*M_PI) {
    return {-1, 1};
  }
  auto it = normalizeAngle(*this);
  assert(-M_PI <= it.inf and it.inf <= M_PI);

  double ret_inf = cos(it.inf);
  double ret_sup = cos(it.sup);
  Interval ret = Interval(std::min(ret_inf, ret_sup), std::max(ret_inf, ret_sup));

  if (it.contains(M_PI)) {
    ret.inf = -1;
  }
  if (it.contains(0) || it.contains(2*M_PI)) {
    ret.sup = 1;
  }

  return ret;
}

Interval Interval::sine() const {
    if (sup - inf >= 2*M_PI) {
        return {-1, 1};
    }
    auto it = normalizeAngle(*this);
    assert(-M_PI <= it.inf and it.inf <= M_PI);
    assert(-M_PI <= it.sup and it.sup <= 3*M_PI);

    double ret_inf = sin(it.inf);
    double ret_sup = sin(it.sup);
    Interval ret = Interval(std::min(ret_inf, ret_sup), std::max(ret_inf, ret_sup));

    if (it.contains(-0.5*M_PI) || it.contains(1.5*M_PI)) {
        ret.inf = -1;
    }
    if (it.contains(0.5*M_PI) || it.contains(2.5*M_PI)){
        ret.sup = 1;
    }

    return ret;
}

Interval Interval::operator - () const {
  return Interval(-sup, -inf);
}

Interval Interval::operator + (const Interval& other) const {
    return Interval(inf + other.inf, sup + other.sup);
}

Interval Interval::operator + (double other) const {
    return Interval(inf + other, sup + other);
}

Interval Interval::operator - (double other) const {
    return *this + (-other);
}

Interval Interval::operator - (const Interval& other) const {
  return -other + *this;
}

Interval Interval::operator * (double other) const {
    if (other > 0) {
        return Interval(inf * other, sup * other);
    }
    return {sup * other, inf * other};
}

double Interval::length() const {
    return sup - inf;
}

Interval Interval::operator * (const Interval& other) const {
    Interval tmp1 = (*this) * other.inf;
    Interval tmp2 = (*this) * other.sup;
    return {std::min(tmp1.inf, tmp2.inf), std::max(tmp1.sup, tmp2.sup)};
}

Interval Interval::meet(const Interval& other) const {
  if (this->is_empty() || other.is_empty()) {
    return Interval();
  }
  if (inf > other.sup || other.inf > sup) {
    return Interval();
  }
  return Interval(std::max(inf, other.inf), std::min(sup, other.sup));
}

Interval Interval::join(const Interval& other) const {
  return Interval(std::min(inf, other.inf), std::max(sup, other.sup));
}

// Interval Interval::join(std::vector<Interval> intervals) {
//   Interval ret = intervals[0];
//   for (Interval it : intervals) {
//     ret = ret.join(it);
//   }
//   return ret;
// }

Interval operator - (const double& a, const Interval &it) {
    return -it + a;
}

Interval operator + (const double& a, const Interval &it) {
    return it + a;
}

Interval operator * (const double& a, const Interval &it) {
    return it * a;
}

std::ostream& operator<<(std::ostream& os, const Interval& it) {
  return os << "[" << it.inf << ", " << it.sup << "]";
}

Interval cos(Interval phi) {
    return phi.cosine();
}

Interval sin(Interval phi) {
    return phi.sine();
}

Interval Interval::pow(int k) const {
    if (k == 0) {
        return {1, 1};
    }
    Interval ret = {1, 1};
    for (int j = 0; j < k; ++j) {
        ret = ret * (*this);
    }
    return ret;
}
