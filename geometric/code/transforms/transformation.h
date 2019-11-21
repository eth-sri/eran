#include "utils/lipschitz.h"
#include "utils/utilities.h"
#include "utils/constants.h"
#include <functional>
#include <vector>

#pragma once

class SpatialTransformation {

public:
    HyperBox domain;
    size_t dim;

    explicit SpatialTransformation(HyperBox domain) {
        this->domain = domain;
        this->dim = domain.dim;
    }

    virtual Pixel<double> transform(const Pixel<double>& pixel, const std::vector<double>& params) const = 0;
    virtual Pixel<Interval> transform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const = 0;
    virtual pair<vector<Interval>, vector<Interval>> gradTransform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const = 0;
    virtual pair<Interval, Interval> dx(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const = 0;
    virtual pair<Interval, Interval> dy(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const = 0;
    virtual SpatialTransformation* getInverse() = 0;
};