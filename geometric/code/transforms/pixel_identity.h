#include "pixel_transform.h"

#pragma once

class PixelIdentity : public PixelTransformation {
public:

    explicit PixelIdentity (HyperBox domain) : PixelTransformation(domain) {
        assert(domain.dim == 0);
    }

    double transform(double pixelValue, const std::vector<double>&params) const;
    Interval transform(const Interval& pixelValue, const std::vector<Interval>& params) const;

    vector<Interval> gradTransform(const Interval& pixelValue, const std::vector<Interval>& params) const;
    Interval dp(const Interval& pixelValue, const std::vector<Interval>& params) const;
};

