#include "domains/dp.h"
#include "utils/constants.h"
#include "transforms/interpolation.h"
#include <cmath>
#include <iostream>
#include <tuple>

std::tuple<int, int, int, int> calculateBoundingBox(Interval x, Interval y, int parity) {
    int lo_x = (int)floor(x.inf - Constants::EPS);
    int hi_x = (int)ceil(x.sup + Constants::EPS);
    int lo_y = (int)floor(y.inf - Constants::EPS);
    int hi_y = (int)ceil(y.sup + Constants::EPS);

    if (abs(lo_x) % 2 != parity) {
        --lo_x;
    }
    if (abs(hi_x) % 2 != parity) {
        ++hi_x;
    }
    if (abs(lo_y) % 2 != parity) {
        --lo_y;
    }
    if (abs(hi_y) % 2 != parity) {
        ++hi_y;
    }

    return make_tuple(lo_x, hi_x, lo_y, hi_y);
}


Interval InterpolationTransformation::transform(Pixel<double> pix, const Image &img, bool lower) const {
    return transform({{pix.x, pix.x}, {pix.y, pix.y}, pix.channel}, img, lower);
}

Polyhedra InterpolationTransformation::transformCustom(
        Pixel<Interval> pix, Pixel<double> oldPix, const Image &img, HyperBox hbox, Interval boxIt) const {
    Interval ret, ret2;
    int parity = (img.nRows - 1) % 2;
    int lo_x, hi_x, lo_y, hi_y;

    assert(hbox.dim == 1);
    Interval it_delta = {hbox.it[0].inf, hbox.it[0].sup};

    std::tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(pix.x, pix.y, parity);
    double cx = oldPix.x, cy = oldPix.y;

    bool first = true;
    double lmin, lmax, rmin, rmax;

    for (int x1 = lo_x; x1 < hi_x; x1 += 2) {
        for (int y1 = lo_y; y1 < hi_y; y1 += 2) {
            Interval x_box = pix.x.meet(Interval(x1, x1 + 2));
            Interval y_box = pix.y.meet(Interval(y1, y1 + 2));

            if (x_box.is_empty() || y_box.is_empty()) {
                continue;
            }

            double alpha = img.find_pixel(x1, y1, pix.channel).inf;
            double beta = img.find_pixel(x1, y1 + 2, pix.channel).inf;
            double gamma = img.find_pixel(x1 + 2, y1, pix.channel).inf;
            double delta = img.find_pixel(x1 + 2, y1 + 2, pix.channel).inf;

            // Translation 1d
             double min_delta = max(x1 - cx, y1 - cy);
             double max_delta = min(x1 + 2 - cx, y1 + 2 - cy);
             Interval sub_delta = Interval(min_delta, max_delta).meet(it_delta);
             DeepPoly1d px = {1, cx, 1, cx, sub_delta};
             DeepPoly1d py = {1, cy, 1, cy, sub_delta};

            // Scale 1d
            //double min_lambda = max(cx > 0 ? x1/cx : (x1+2)/cx, cy > 0 ? y1/cy : (y1+2)/cy);
            //double max_lambda = min(cx < 0 ? x1/cx : (x1+2)/cx, cy < 0 ? y1/cy : (y1+2)/cy);
            //Interval sub_delta = Interval(min_lambda, max_lambda).meet(it_delta);
            //DeepPoly1d px = {cx, 0, cx, 0, sub_delta};
            //DeepPoly1d py = {cy, 0, cy, 0, sub_delta};

            // Shear 1d
            //double min_lambda = cy > 0 ? (x1 - cx) / cy : (x1 + 2 - cx) / cy;
            //double max_lambda = cy < 0 ? (x1 - cx) / cy : (x1 + 2 - cx) / cy;
            //Interval sub_delta = Interval(min_lambda, max_lambda).meet(it_delta);
            //DeepPoly1d px = {cy, cx, cy, cx, sub_delta};
            //DeepPoly1d py = {0, cy, 0, cy, sub_delta};



            Interval tmp = 0.25 * (alpha * (x1 + 2 - pix.x) * (y1 + 2 - pix.y)
                                   + beta * (x1 + 2 - pix.x) * (pix.y - y1)
                                   + gamma * (pix.x - x1) * (y1 + 2 - pix.y)
                                   + delta * (pix.x - x1) * (pix.y - y1));

            DeepPoly1d tmp_dp = 0.25 * (alpha * (-px + (x1 + 2)) * (-py + (y1 + 2))
                                        + beta * (-px + (x1 + 2)) * (py - y1)
                                        + gamma * (px - x1) * (-py + (y1 + 2))
                                        + delta * (px - x1) * (py - y1));

            double tmp_lmin = tmp_dp.lw * sub_delta.inf + tmp_dp.lc;
            double tmp_lmax = tmp_dp.uw * sub_delta.inf + tmp_dp.uc;
            double tmp_rmin = tmp_dp.lw * sub_delta.sup + tmp_dp.lc;
            double tmp_rmax = tmp_dp.uw * sub_delta.sup + tmp_dp.uc;

            if (first) {
                lmin = tmp_lmin;
                lmax = tmp_lmax;
                rmin = tmp_rmin;
                rmax = tmp_rmax;
            } else {
                lmin = min(lmin, tmp_lmin);
                lmax = max(lmax, tmp_lmax);
                rmin = min(rmin, tmp_rmin);
                rmax = max(rmax, tmp_rmax);
            }
            first = false;
        }
    }

    double wLow = (rmin - lmin) / (it_delta.sup - it_delta.inf);
    double wUp = (rmax - lmax) / (it_delta.sup - it_delta.inf);
    double biasLow = lmin - wLow * it_delta.inf;
    double biasUp = lmax - wUp * it_delta.inf;

    Polyhedra poly(ret, {wLow}, biasLow, {wUp}, biasUp, 1);
    return poly;
}


/*
  Performs bilinear interpolation of pixel pix in the original image.
  Arguments:
    pix: pixel which is interpolated (pix.x, pix.y) are coordinates of this pixel (both are intervals)
	img: original image
	lower: only for experiments with noise, whether to use minimum or maximum noise
 */
Interval InterpolationTransformation::transform(Pixel<Interval> pix, const Image& img, bool lower) const {
    Interval ret;
    int parity = (img.nRows - 1) % 2;
    int lo_x, hi_x, lo_y, hi_y;

	// computes bounding box of possible concrete pixel locations
    std::tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(pix.x, pix.y, parity);

	// traverse all interpolation regions
    for (int x1 = lo_x; x1 < hi_x; x1 += 2) {
        for (int y1 = lo_y; y1 < hi_y; y1 += 2) {
     		// intersect pixel coordinates with interpolation region
            Interval x_box = pix.x.meet(Interval(x1, x1 + 2));
            Interval y_box = pix.y.meet(Interval(y1, y1 + 2));

			// skip if there is no intersection with this region
            if (x_box.is_empty() || y_box.is_empty()) {
                continue;
            }

			// compute pixel values in 4 corners of interpolation region
            double alpha, beta, gamma, delta;
            if (lower) {
                alpha = img.find_pixel(x1, y1, pix.channel).inf;
                beta = img.find_pixel(x1, y1 + 2, pix.channel).inf;
                gamma = img.find_pixel(x1 + 2, y1, pix.channel).inf;
                delta = img.find_pixel(x1 + 2, y1 + 2, pix.channel).inf;
            } else {
                alpha = img.find_pixel(x1, y1, pix.channel).sup;
                beta = img.find_pixel(x1, y1 + 2, pix.channel).sup;
                gamma = img.find_pixel(x1 + 2, y1, pix.channel).sup;
                delta = img.find_pixel(x1 + 2, y1 + 2, pix.channel).sup;
            }

			// use formula for bilinear interpolation to obtain all possible pixel values
            Interval tmp = 0.25 * (alpha * (x1 + 2 - pix.x) * (y1 + 2 - pix.y)
                                   + beta * (x1 + 2 - pix.x) * (pix.y - y1)
                                   + gamma * (pix.x - x1) * (y1 + 2 - pix.y)
                                   + delta * (pix.x - x1) * (pix.y - y1));

            ret = ret.join(tmp);
        }
    }

    return ret;
}

pair<bool, vector<Interval>> InterpolationTransformation::gradTransform(Pixel<Interval> pix, const Image &img, bool lower) const {
    int parity = (img.nRows - 1) % 2;
    int lo_x, hi_x, lo_y, hi_y;
    std::tie(lo_x, hi_x, lo_y, hi_y) = calculateBoundingBox(pix.x, pix.y, parity);

    Interval dfdx, dfdy;
    // cnt: number of intersecting boxes
    int cnt = 0;

    for (int x1 = lo_x; x1 < hi_x; x1 += 2) {
        for (int y1 = lo_y; y1 < hi_y; y1 += 2) {
            Interval x_box = pix.x.meet(Interval(x1, x1 + 2));
            Interval y_box = pix.y.meet(Interval(y1, y1 + 2));

            if (x_box.is_empty() || y_box.is_empty()) {
                continue;
            }

            double alpha, beta, gamma, delta;
            if (lower) {
                alpha = img.find_pixel(x1, y1, pix.channel).inf;
                beta = img.find_pixel(x1, y1 + 2, pix.channel).inf;
                gamma = img.find_pixel(x1 + 2, y1, pix.channel).inf;
                delta = img.find_pixel(x1 + 2, y1 + 2, pix.channel).inf;
            } else {
                alpha = img.find_pixel(x1, y1, pix.channel).sup;
                beta = img.find_pixel(x1, y1 + 2, pix.channel).sup;
                gamma = img.find_pixel(x1 + 2, y1, pix.channel).sup;
                delta = img.find_pixel(x1 + 2, y1 + 2, pix.channel).sup;
            }

            ++cnt;
            dfdx = dfdx.join(0.25 * ((gamma - alpha) * (y1 + 2 - pix.y) + (delta - beta) * (pix.y - y1)));
            dfdy = dfdy.join(0.25 * ((beta - alpha) * (x1 + 2 - pix.x) + (delta - gamma) * (pix.x - x1)));
        }
    }

    return {cnt == 1, {dfdx, dfdy}};
}
