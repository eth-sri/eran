#include "translation.h"
#include <iostream>

Pixel<double> TranslationTransformation::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == 2);
    double deltaX = params[0];
    double deltaY = params[1];
    return {pixel.x + deltaX, pixel.y + deltaY, pixel.channel};
}

// Interval:
Pixel<Interval> TranslationTransformation::transform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const {
    assert(params.size() == 2);
    Interval deltaX = params[0];
    Interval deltaY = params[1];
    return {pixel.x + deltaX, pixel.y + deltaY, pixel.channel};
}

pair<vector<Interval>, vector<Interval>> TranslationTransformation::gradTransform(
        const Pixel<Interval> &pixel,
        const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    return {{Interval(1.0, 1.0), Interval(0.0, 0.0)}, {Interval(0.0, 0.0), Interval(1.0, 1.0)}};
}

pair<Interval, Interval> TranslationTransformation::dx(const Pixel<Interval> &pixel,
                                                       const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    return {{1.0, 1.0}, {0.0, 0.0}};
}

pair<Interval, Interval> TranslationTransformation::dy(const Pixel<Interval> &pixel,
                                                       const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    return {{0.0, 0.0}, {1.0, 1.0}};
}

SpatialTransformation* TranslationTransformation::getInverse() {
    HyperBox new_domain = HyperBox({
        {-this->domain[0].sup, -this->domain[0].inf},
        {-this->domain[1].sup, -this->domain[1].inf}});
    return new TranslationTransformation(new_domain);
}