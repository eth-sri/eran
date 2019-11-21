#include "shear.h"
#include <iostream>

Pixel<double> ShearTransformation::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == 1);
    return {pixel.x + pixel.y * params[0], pixel.y, pixel.channel};
}

// Interval
Pixel<Interval> ShearTransformation::transform(const Pixel<Interval> &pixel,
                                               const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {pixel.x + pixel.y * params[0], pixel.y, pixel.channel};
}

pair<vector<Interval>, vector<Interval>> ShearTransformation::gradTransform(const Pixel<Interval> &pixel,
                                                                            const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{pixel.y}, {{0, 0}}};
}

pair<Interval, Interval> ShearTransformation::dx(const Pixel<Interval> &pixel,
                                                 const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{1, 1}, params[0]};
}

pair<Interval, Interval> ShearTransformation::dy(const Pixel<Interval> &pixel,
                                                 const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{0, 0}, {1, 1}};
}

SpatialTransformation* ShearTransformation::getInverse() {
    HyperBox new_domain = HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
    return new ShearTransformation(new_domain);
}