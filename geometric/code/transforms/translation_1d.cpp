#include "translation_1d.h"
#include <iostream>

Pixel<double> TranslationTransformation1d::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == 1);
    double delta = params[0];
    return {pixel.x + delta, pixel.y + delta, pixel.channel};
}

// Interval:
Pixel<Interval> TranslationTransformation1d::transform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const {
    assert(params.size() == 1);
    Interval delta = params[0];
    return {pixel.x + delta, pixel.y + delta, pixel.channel};
}

pair<vector<Interval>, vector<Interval>> TranslationTransformation1d::gradTransform(
        const Pixel<Interval> &pixel,
        const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{Interval(1.0, 1.0)}, {Interval(1.0, 1.0)}};
}

pair<Interval, Interval> TranslationTransformation1d::dx(const Pixel<Interval> &pixel,
                                                       const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{1.0, 1.0}, {0.0, 0.0}};
}

pair<Interval, Interval> TranslationTransformation1d::dy(const Pixel<Interval> &pixel,
                                                       const std::vector<Interval> &params) const {
    assert(params.size() == 1);
    return {{0.0, 0.0}, {1.0, 1.0}};
}

SpatialTransformation* TranslationTransformation1d::getInverse() {
    HyperBox new_domain = HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
    return new TranslationTransformation1d(new_domain);
}
