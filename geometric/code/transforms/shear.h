#include "transforms/transformation.h"
#include <cmath>

#pragma once

class ShearTransformation : public SpatialTransformation {
public:

    explicit ShearTransformation (HyperBox domain) : SpatialTransformation(domain) {
        assert(domain.dim == 1);
    }

    Pixel<double> transform(const Pixel<double>&, const std::vector<double>& ) const override;
    
    // Interval:
    Pixel<Interval> transform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<vector<Interval>, vector<Interval>> gradTransform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<Interval, Interval> dx(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<Interval, Interval> dy(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    SpatialTransformation* getInverse() override;
};
