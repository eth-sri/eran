#include "domains/polyhedra.h"
#include "utils/utilities.h"

#pragma once

class InterpolationTransformation {

public:

    Interval transform (Pixel<double> pix, const Image& img, bool lower) const;
    Interval transform (Pixel<Interval> pix, const Image& img, bool lower) const;
    Polyhedra transformCustom(Pixel<Interval> pix, Pixel<double> oldPix, const Image& img, HyperBox hbox, Interval it) const;

    pair<bool, vector<Interval>> gradTransform (Pixel<Interval> pix, const Image& img, bool lower) const;
};

// Calculates bounding box of squares in which pixel can be interpolated
std::tuple<int, int, int, int> calculateBoundingBox(Interval x, Interval y, int parity);

