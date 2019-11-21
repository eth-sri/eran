#include <ostream>
#include <vector>

#pragma once

class Interval {
 public:
  double inf;
  double sup;

  Interval(double inf, double sup);
  Interval();
  bool is_empty() const;
  bool contains(double x) const;
  Interval cosine() const;
  Interval sine() const;
  double length() const;

  // pass by reference
  Interval operator + (const Interval& other) const;
  Interval operator - (const Interval& other) const;
  Interval operator * (const Interval& other) const;
  Interval operator - () const;
  Interval& operator += (const Interval& other);


  Interval operator - (double other) const;
  Interval operator + (double other) const;
  Interval operator * (double other) const;

  Interval meet(const Interval& other) const;
  Interval join(const Interval& other) const;
  Interval pow(int k) const;

  static Interval join(std::vector<Interval> intervals);
  static Interval getR();
};

Interval operator - (const double& a, const Interval &it);
Interval operator + (const double& a, const Interval &it);
Interval operator * (const double& a, const Interval &it);
std::ostream& operator<<(std::ostream& os, const Interval& it);
Interval normalizeAngle(Interval phi);

Interval cos(Interval phi);
Interval sin(Interval phi);
Interval abs(const Interval& it);
