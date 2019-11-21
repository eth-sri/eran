#include "rotation.h"
#include <iostream>

Pixel<double> RotationTransformation::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == 1);
    double phi = params[0];
    return {pixel.x * cos(phi) - pixel.y * sin(phi), pixel.x * sin(phi) + pixel.y * cos(phi), pixel.channel};
}


Pixel<Interval> RotationTransformation::transform(const Pixel<Interval> &pixel,
                                                  const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    Interval phi = params[0];
    return {pixel.x * cos(phi) - pixel.y * sin(phi), pixel.x * sin(phi) + pixel.y * cos(phi), pixel.channel};
}

pair<vector<Interval>, vector<Interval>> RotationTransformation::gradTransform(const Pixel<Interval> &pixel,
                                                                   const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    Interval phi = params[0];
    return {{-pixel.x * sin(phi) - pixel.y * cos(phi)}, {pixel.x * cos(phi) - pixel.y * sin(phi)}};
}

pair<Interval, Interval> RotationTransformation::dx(const Pixel<Interval> &pixel,
                                                    const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    Interval phi = params[0];
    return {cos(phi), -sin(phi)};
}

pair<Interval, Interval> RotationTransformation::dy(const Pixel<Interval> &pixel,
                                                    const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    Interval phi = params[0];
    return {sin(phi), cos(phi)};
}

SpatialTransformation* RotationTransformation::getInverse()  {
    HyperBox new_domain = HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
    return new RotationTransformation(new_domain);
}