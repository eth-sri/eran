#include "pixel_transform.h"

#pragma once

class BrightnessTransformation : public PixelTransformation {
public:

    explicit BrightnessTransformation (HyperBox domain) : PixelTransformation(domain) {
        assert(domain.dim == 2);
    }


    double transform(double pixelValue, const std::vector<double>&params) const;
    Interval transform(const Interval& pixelValue, const std::vector<Interval>& params) const;

    vector<Interval> gradTransform(const Interval& pixelValue, const std::vector<Interval>& params) const;
    Interval dp(const Interval& pixelValue, const std::vector<Interval>& params) const;
};