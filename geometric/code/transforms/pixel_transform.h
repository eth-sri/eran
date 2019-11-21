#include "utils/lipschitz.h"
#include <vector>

#pragma once

class PixelTransformation {

public:
    HyperBox domain;
    size_t dim;

    explicit PixelTransformation(HyperBox domain) {
        this->domain = domain;
        this->dim = domain.dim;
    }

    vector<PointD> randomParams(int k, std::default_random_engine generator) const {
        return domain.sample(k, generator);
    }

    virtual double transform(double pixelValue, const std::vector<double>&params) const = 0;
    virtual Interval transform(const Interval& pixel, const std::vector<Interval>& params) const = 0;

    virtual vector<Interval> gradTransform(const Interval& pixelValue, const std::vector<Interval>& params) const = 0;
    virtual Interval dp(const Interval& pixelValue, const std::vector<Interval>& params) const = 0;
};

