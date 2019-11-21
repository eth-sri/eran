#include "composition.h"

Pixel<double> CompositionTransform::transform(const Pixel<double>& pixel, const std::vector<double>& params) const {
    assert(params.size() == dim);
    int j = 0;
    Pixel<double> ret = pixel;
    for (SpatialTransformation* t : transforms) {
        vector<double> newParams(params.begin() + j, params.begin() + j + t->dim);
        j += t->dim;
        ret = t->transform(ret, newParams);
    }
    return ret;
}

Pixel<Interval> CompositionTransform::transform(const Pixel<Interval> &pixel,
                                                const std::vector<Interval> &params) const {
    assert(params.size() == dim);
    int j = 0;
    Pixel<Interval> ret = pixel;
    for (SpatialTransformation* t : transforms) {
        vector<Interval> newParams(params.begin() + j, params.begin() + j + t->dim);
        j += t->dim;
        ret = t->transform(ret, newParams);
    }
    return ret;
}

SpatialTransformation* CompositionTransform::getInverse() {
    vector<SpatialTransformation*> invTransforms;
    for (auto it = transforms.rbegin(); it != transforms.rend(); ++it) {
        invTransforms.push_back((*it)->getInverse());
    }
    return new CompositionTransform(invTransforms);
}

pair<vector<Interval>, vector<Interval>> CompositionTransform::computeGrad(const Pixel<Interval> &pixel,
                                                                           const std::vector<Interval> &params,
                                                                           Interval& dxdx,
                                                                           Interval& dxdy,
                                                                           Interval& dydx,
                                                                           Interval& dydy) const {
    vector<Pixel<Interval>> tPixel;
    vector<vector<Interval>> tParams;

    int j = 0;
    Pixel<Interval> currPixel = pixel;
    for (SpatialTransformation* t : transforms) {
        vector<Interval> newParams(params.begin() + j, params.begin() + j + t->dim);
        j += t->dim;
        tPixel.push_back(currPixel);
        tParams.push_back(newParams);
        currPixel = t->transform(currPixel, newParams);
    }

    vector<Interval> retX, retY;
    for (int i = (int)transforms.size() - 1; i >= 0; --i) {
        vector<Interval> gradX, gradY;
        tie(gradX, gradY) = transforms[i]->gradTransform(tPixel[i], tParams[i]);

        for (int idx = (int)gradX.size() - 1; idx >= 0; --idx) {
            retX.insert(retX.begin(), dxdx * gradX[idx] + dxdy * gradY[idx]);
            retY.insert(retY.begin(), dydx * gradX[idx] + dydy * gradY[idx]);
        }

        Interval cdxdx, cdxdy, cdydx, cdydy;
        tie(cdxdx, cdxdy) = transforms[i]->dx(tPixel[i], tParams[i]);
        tie(cdydx, cdydy) = transforms[i]->dy(tPixel[i], tParams[i]);

        auto tmp_dxdx = dxdx * cdxdx + dxdy * cdydx;
        auto tmp_dxdy = dxdx * cdxdy + dxdy * cdydy;
        auto tmp_dydx = dydx * cdxdx + dydy * cdydx;
        auto tmp_dydy = dydy * cdydy + dydx * cdxdy;

        dxdx = tmp_dxdx;
        dxdy = tmp_dxdy;
        dydx = tmp_dydx;
        dydy = tmp_dydy;
    }

    return {retX, retY};
}

// pari< partial derivatives of first return value with respect to parameters, same for second >
pair<vector<Interval>, vector<Interval>> CompositionTransform::gradTransform(const Pixel<Interval> &pixel,
                                                                             const std::vector<Interval> &params) const {
    Interval dxdx(1, 1), dxdy(0, 0), dydx(0, 0), dydy(1, 1);
    return computeGrad(pixel, params, dxdx, dxdy, dydx, dydy);
}

// pari< dx'/dx, dx'/dy >
pair<Interval, Interval> CompositionTransform::dx(const Pixel<Interval> &pixel,
                                                  const std::vector<Interval> &params) const {
    Interval dxdx(1, 1), dxdy(0, 0), dydx(0, 0), dydy(1, 1);
    computeGrad(pixel, params, dxdx, dxdy, dydx, dydy);
    return {dxdx, dxdy};
}

// pari< dy'/dx, dy'/dy >
pair<Interval, Interval> CompositionTransform::dy(const Pixel<Interval> &pixel,
                                                  const std::vector<Interval> &params) const {
    Interval dxdx(1, 1), dxdy(0, 0), dydx(0, 0), dydy(1, 1);
    computeGrad(pixel, params, dxdx, dxdy, dydx, dydy);
    return {dydx, dydy};
}

