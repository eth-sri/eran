#include "brightness_transform.h"
#include <algorithm>

double BrightnessTransformation::transform(double pixelValue, const std::vector<double> &params) const {
    assert(params.size() == 2);
    double ret = pixelValue * params[0] + params[1];
    if (ret > 1) {
        return 1;
    } else if (ret < 0) {
        return 0;
    }
    return ret;
}

Interval BrightnessTransformation::transform(const Interval &pixelValue, const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    Interval ret = pixelValue * params[0] + params[1];
    if (ret.inf > 1) {
        return {1, 1};
    } else if (ret.sup < 0) {
        return {0, 0};
    }
    return ret.meet({0, 1});
}

vector<Interval> BrightnessTransformation::gradTransform(const Interval& pixelValue,
                                                         const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    Interval newPixelValue = pixelValue * params[0] + params[1];
    vector<Interval> ret = {pixelValue, Interval(1, 1)};
    if (newPixelValue.inf < 0 || newPixelValue.sup > 1) {
        ret[0] = ret[0].join({0, 0});
        ret[1] = ret[1].join({0, 0});
    }
    return ret;
}

Interval BrightnessTransformation::dp(const Interval &pixelValue, const std::vector<Interval> &params) const {
    assert(params.size() == 2);
    Interval newPixelValue = pixelValue * params[0] + params[1];
    Interval ret = params[0];
    if (newPixelValue.inf < 0 || newPixelValue.sup > 1) {
        ret = ret.join({0, 0});
    }
    return ret;
}
