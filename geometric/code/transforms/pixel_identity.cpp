#include "pixel_identity.h"

double PixelIdentity::transform(double pixelValue, const std::vector<double> &params) const {
    if (pixelValue > 1) {
        return 1;
    } else if (pixelValue < 0) {
        return 0;
    }
    return pixelValue;
}

Interval PixelIdentity::transform(const Interval &pixelValue, const std::vector<Interval> &params) const {
    if (pixelValue.inf > 1) {
        return {1, 1};
    } else if (pixelValue.sup < 0) {
        return {0, 0};
    }
    return pixelValue.meet({0, 1});
}

vector<Interval> PixelIdentity::gradTransform(const Interval &pixelValue, const std::vector<Interval> &params) const {
    return {};
}

Interval PixelIdentity::dp(const Interval &pixelValue, const std::vector<Interval> &params) const {
    if (pixelValue.inf > 1) {
        return {0, 0};
    } else if (pixelValue.sup < 0) {
        return {0, 0};
    }
    if (pixelValue.inf > 0 && pixelValue.sup < 1) {
        return {1, 1};
    }
    return {0, 1};
}

