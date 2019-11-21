#include "scale.h"
#include <iostream>

Pixel<double> ScaleTransformation::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == 1);
    return {pixel.x * params[0], pixel.y * params[0], pixel.channel};
}

// Interval
Pixel<Interval> ScaleTransformation::transform(const Pixel<Interval> &pixel,
                                               const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {pixel.x * params[0], pixel.y * params[0], pixel.channel};
}

pair<vector<Interval>, vector<Interval>> ScaleTransformation::gradTransform(const Pixel<Interval> &pixel,
                                                                            const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{pixel.x}, {pixel.y}};
}

pair<Interval, Interval> ScaleTransformation::dx(const Pixel<Interval> &pixel,
                                                 const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {params[0], {0, 0}};
}

pair<Interval, Interval> ScaleTransformation::dy(const Pixel<Interval> &pixel,
                                                 const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{0, 0}, params[0]};
}

SpatialTransformation* ScaleTransformation::getInverse() {
    HyperBox new_domain = HyperBox({{1.0/this->domain[0].sup, 1.0/this->domain[0].inf}});
    return new ScaleTransformation(new_domain);
}