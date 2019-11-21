#include "domains/interval.h"
#include "utils/constants.h"
#include "utils/utilities.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <functional>
#include <random>

#pragma once

using namespace std;

class PointD {

public:
    vector<double> x;

    PointD() {}
    PointD(vector<double> x) { this->x = x; }
    operator vector<double>() const { return x; }
    PointD operator + (const PointD& other) const;
};

class HyperBox {

public:
    vector<Interval> it;
    size_t dim;
    
    HyperBox() { dim = 0; }
    HyperBox(vector<Interval> intervals) { this->it = intervals; this-> dim = intervals.size(); }

    PointD center() const;
    int getIndexToCut(pair<bool, vector<Interval>> grad) const;
    double diameter() const;
    Interval& operator[](int i);
    vector<PointD> sample(int, std::default_random_engine) const; // sample point from HyperBox uniformly at random
    vector<HyperBox> split(int k, vector<vector<double>>& splitPoints) const; // split HyperBox in smaller HyperBoxes, make k splits per dimension
    bool inside(PointD p) const; // check whether point is inside of hbox
    void split(size_t dim1, HyperBox& hbox1, HyperBox& hbox2) const;
    static HyperBox concatenate(const HyperBox& hbox1, const HyperBox& hbox2);
};

std::ostream& operator << (std::ostream& os, const PointD& pt);
std::ostream& operator << (std::ostream& os, const HyperBox& box);

class LipschitzFunction {

public:

    function<double(vector<double>)> f; // function from vector<double> to double
    HyperBox domain; // domain represented as HyperBox
    function<pair<bool, vector<Interval>>(const HyperBox& hbox)> gradF_interval = [](const HyperBox& hbox) {
        cout << "empty" << endl;
        assert(false);
        return make_pair(false, vector<Interval>());
    };

    LipschitzFunction(function<double(vector<double>)> f, HyperBox domain, // const DLFunction dlfunc,
                      function<pair<bool, vector<Interval>>(const HyperBox& hbox)> gradF_interval = nullptr) {
        this->f = f;
        this->domain = domain;
        if (gradF_interval != nullptr) {
            this->gradF_interval = gradF_interval;
        }
    }

    void setGradientFunction(function<pair<bool, vector<Interval>>(const HyperBox& hbox)> gradF_interval) {
        this->gradF_interval = gradF_interval;
    }

    LipschitzFunction operator + (const LipschitzFunction&);
    LipschitzFunction operator - (const LipschitzFunction&);
    LipschitzFunction operator - () const;

    double getUpperBoundCauchySchwarz(const HyperBox& subdomain, PointD x, pair<bool, vector<Interval>> grad) const;
    double getUpperBoundTriangle(const HyperBox& subdomain, PointD x, pair<bool, vector<Interval>> grad) const;
    double getUpperBound(const HyperBox& subdomain, PointD x) const;

    double maximize(double epsilon, int p, Statistics& counter, int maxIter = 1000000000) const;
    double minimize(double epsilon, int p, Statistics& counter, int maxIter = 1000000000) const;

    static LipschitzFunction getLinear(HyperBox domain, vector<double> weights, double bias, int degree);
};

LipschitzFunction operator * (const double, const LipschitzFunction&);
vector<HyperBox> branching(HyperBox& box, int p, pair<bool, vector<Interval>> grad);

