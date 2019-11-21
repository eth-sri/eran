#include "domains/interval.h"
#include <cassert>
#include <vector>

class DeepPoly1d {

public:

    double lw, lc, uw, uc;
    Interval delta;

    DeepPoly1d(double lw, double lc, double uw, double uc, Interval delta);
    Interval it() const;

    DeepPoly1d operator + (const DeepPoly1d& other) const;
    DeepPoly1d operator * (const DeepPoly1d& other) const;
    DeepPoly1d operator - () const;
    DeepPoly1d operator + (const double &scalar);
    DeepPoly1d operator - (const double &scalar);
};

DeepPoly1d operator * (const double& c, const DeepPoly1d& dp);
