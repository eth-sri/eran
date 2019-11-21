#include "utils/lipschitz.h"
#include "utils/constants.h"
#include <queue>
#include <random>

std::ostream& operator << (std::ostream& os, const PointD& pt) {
    os << "(";
    for (size_t i = 0; i < pt.x.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << pt.x[i];
    }
    os << ")";
    return os;
}

PointD PointD::operator + (const PointD& other) const {
    assert(x.size() == other.x.size());
    vector<double> retX = this->x;
    for (size_t i = 0; i < other.x.size(); ++i) {
        retX[i] += other.x[i];
    }
    return PointD(retX);
}

std::ostream& operator << (std::ostream& os, const HyperBox& box) {
    os << "(";
    for (size_t i = 0; i < box.it.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << box.it[i];
    }
    os << ")";
    return os;
}

vector<PointD> HyperBox::sample(int k, std::default_random_engine generator) const {
    vector<std::uniform_real_distribution<double>> diss;
    for (auto itv : it) {
         diss.emplace_back(itv.inf, itv.sup);
    }

    vector<PointD> ret;
    for (int i = 0; i < k; ++i) {
        PointD samplePoint;
        for (auto d : diss) {
            samplePoint.x.push_back(d(generator));
        }
        ret.push_back(samplePoint);
    }

    return ret;
}

bool HyperBox::inside(PointD p) const {
    assert(dim == p.x.size());
    for (size_t i = 0; i < dim; ++i) {
        if (p.x[i] < it[i].inf - Constants::EPS) {
            return false;
        }
        if (p.x[i] > it[i].sup + Constants::EPS) {
            return false;
        }
    }
    return true;
}

PointD HyperBox::center() const {
    std::vector<double> ret;
    for (auto itv : it) {
        ret.push_back(0.5 * (itv.inf + itv.sup));
    }
    return PointD(ret);
}

void HyperBox::split(size_t dim1, HyperBox &hbox1, HyperBox &hbox2) const {
    assert(dim1 <= dim);
    hbox1.it.insert(hbox1.it.begin(), it.begin(), it.begin() + dim1);
    hbox2.it.insert(hbox2.it.begin(), it.begin() + dim1, it.end());
    hbox1.dim = hbox1.it.size();
    hbox2.dim = hbox2.it.size();
}

HyperBox HyperBox::concatenate(const HyperBox &hbox1, const HyperBox &hbox2) {
    HyperBox hbox;
    hbox.it.insert(hbox.it.end(), hbox1.it.begin(), hbox1.it.end());
    hbox.it.insert(hbox.it.end(), hbox2.it.begin(), hbox2.it.end());
    hbox.dim = hbox1.dim + hbox2.dim;
    return hbox;
}

vector<HyperBox> HyperBox::split(int k, vector<vector<double>>& splitPoints) const {
    if (!splitPoints.empty()) {
        assert(splitPoints.size() == dim);
        for (int i = 0; i < dim; ++i) {
            for (double& x : splitPoints[i]) {
                x = it[i].inf + x * (it[i].sup - it[i].inf);
            }
        }
    } else {
        splitPoints.resize(dim);
        for (int i = 0; i < dim; ++i) {
            double delta = it[i].length() / k;
            for (int j = 1; j <= k - 1; ++j) {
                splitPoints[i].push_back(it[i].inf + j * delta);
            }
        }
    }

    vector<vector<Interval>> chunks(dim);
    for (int i = 0; i < dim; ++i) {
        double prev = it[i].inf;
        for (double x : splitPoints[i]) {
            chunks[i].emplace_back(prev, x);
            prev = x;
        }
        chunks[i].emplace_back(prev, it[i].sup);
    }

    vector<HyperBox> ret;
    for (size_t i = 0; i < dim; ++i) {
        vector<HyperBox> tmp = ret;
        ret.clear();

        for (const Interval& chunk : chunks[i]) {
            if (i == 0) {
                ret.push_back(HyperBox({chunk}));
            } else {
                for (HyperBox hbox : tmp) {
                    HyperBox newBox = hbox;
                    ++newBox.dim;
                    newBox.it.push_back(chunk);
                    ret.push_back(newBox);
                }
            }
        }
    }
    return ret;
}

int HyperBox::getIndexToCut(pair<bool, vector<Interval>> grad) const {
    vector<double> ret;

    if (Constants::SPLIT_MODE == "standard") {
        for (auto itv : it) {
            ret.push_back(itv.sup - itv.inf);
        }
    } else if (Constants::SPLIT_MODE == "gradient") {
        for (int k = 0; k < it.size(); ++k) {
            ret.push_back( (it.at(k).sup - it.at(k).inf) * (grad.second.at(k).sup - grad.second.at(k).inf));
        }
    } else if (Constants::SPLIT_MODE == "gradientSign") {
        for (int k = 0; k < it.size(); ++k) {
            if (grad.second.at(k).inf < 0 && 0 < grad.second.at(k).sup ) {
                ret.push_back( (it.at(k).sup - it.at(k).inf) * (grad.second.at(k).sup - grad.second.at(k).inf));
            } else {
                ret.push_back( 0 );
            }
        }
    } else {
        assert(false);
    }

    return (int)(max_element(ret.begin(), ret.end()) - ret.begin());
}

double HyperBox::diameter() const {
    double sum = 0;
    for (auto itv : it) {
        sum += pow(itv.sup - itv.inf, 2.0);
    }
    return sqrt(sum);
}

Interval& HyperBox::operator[](int i) {
    return it[i];
}

vector<HyperBox> branching(HyperBox& box, int p, pair<bool, vector<Interval>> grad) {
    int t = box.getIndexToCut(grad);
    double delta = (box[t].sup - box[t].inf) / p;
    vector<HyperBox> ret;

    for (int q = 0; q < p; ++q) {
        HyperBox tmp = box;
        tmp[t] = {tmp[t].inf + delta*q, tmp[t].inf + delta*(q+1)};
        ret.push_back(tmp);
    }

    return ret;
}

// LipschitzFunction operator * (const double x, const LipschitzFunction& f) {
//     auto fgrad = f.gradF;
//     auto ff = f.f;
//     std::function<double(vector<double>)> mulF = [x, ff](vector<double> xp) {
//         return x * ff(xp);
//     };
//     function<pair<bool, vector<Interval>>(const HyperBox& hbox)> mulGrad = [x, fgrad](const HyperBox& hbox) {
//         auto ret = fgrad(hbox);
//         for (size_t i = 0; i < ret.second.size(); ++i) {
//             ret.second[i] = ret.second[i] * x;
//         }
//         return ret;
//     };
//     return LipschitzFunction(mulF, f.domain, f.image * x, mulGrad);
// }

// LipschitzFunction LipschitzFunction::operator * (const LipschitzFunction &other) {
//     assert(domain.it.size() == other.domain.it.size());
//     assert(false);
//     auto tmp1 = f, tmp2 = other.f;
//     std::function<double(vector<double>)> mulF = [tmp1, tmp2](vector<double> x) {
//         return tmp1(x) * tmp2(x);
//     };
//     return LipschitzFunction(mulF, domain, image * other.image);
// }

LipschitzFunction LipschitzFunction::operator + (const LipschitzFunction& other) {
    assert(domain.it.size() == other.domain.it.size());
    auto tmp1 = f, tmp2 = other.f;
    std::function<double(vector<double>)> sumF = [tmp1, tmp2](vector<double> x) {
        return tmp1(x) + tmp2(x);
    };
    auto tmpGrad1_i = gradF_interval, tmpGrad2_i = other.gradF_interval;
    function<pair<bool, vector<Interval>>(const HyperBox& hbox)> addGrad_i = [tmpGrad1_i, tmpGrad2_i](const HyperBox& hbox) {
        auto p1 = tmpGrad1_i(hbox), p2 = tmpGrad2_i(hbox);
        vector<Interval> ret;
        for (size_t i = 0; i < p1.second.size(); ++i) {
            ret.push_back(p1.second[i] + p2.second[i]);
        }
        return make_pair(true, ret);
    };

    return LipschitzFunction(sumF, domain, addGrad_i);
}

LipschitzFunction LipschitzFunction::operator - (const LipschitzFunction& other) {
    return *this + (-other);
}

LipschitzFunction LipschitzFunction::operator - () const {
    auto tmpF = f;
    auto tmpGradF_i = gradF_interval;
    std::function<double(vector<double>)> negativeF = [tmpF](vector<double> x) {
        return -tmpF(x);
    };

    function<pair<bool, vector<Interval>>(const HyperBox& hbox)> negativeGradF_i = [tmpGradF_i](const HyperBox& hbox) {
        auto ret = tmpGradF_i(hbox);
        for (size_t i = 0; i < ret.second.size(); ++i) {
            ret.second[i] = -ret.second[i];
        }
        return ret;
    };

    auto ret = LipschitzFunction(negativeF, domain, negativeGradF_i);
    return ret;
}


double LipschitzFunction::getUpperBoundCauchySchwarz(const HyperBox& subdomain, PointD x, pair<bool, vector<Interval>> grad) const {
    double lipConst = 0;
        for (const Interval& pd : grad.second) {
            lipConst += max(pd.sup * pd.sup, pd.inf * pd.inf);
        }
        lipConst = sqrt(lipConst);
    return f(x) + 0.5 * lipConst * subdomain.diameter();
}

double LipschitzFunction::getUpperBoundTriangle(const HyperBox& subdomain, PointD x, pair<bool, vector<Interval>> grad) const {
    double ret = f(x);
    for (int k = 0; k < subdomain.dim; ++k) {
        ret += 0.5 * max(abs(grad.second[k].inf), abs(grad.second[k].sup)) * (subdomain.it[k].sup - subdomain.it[k].inf);
    }
    return ret;
}

double LipschitzFunction::getUpperBound(const HyperBox& subdomain, PointD x) const {
    pair<bool, vector<Interval>> grad = gradF_interval(subdomain);
    if (Constants::UB_ESTIMATE == "Triangle") {
        return getUpperBoundTriangle(subdomain, x, grad);
    } else if (Constants::UB_ESTIMATE == "CauchySchwarz") {
        return getUpperBoundCauchySchwarz(subdomain, x, grad);
    } else {
        assert(false);
    }
}

double LipschitzFunction::maximize(double epsilon, int p, Statistics& counter, int maxIter) const {
    PointD x_opt = domain.center();
    double f_opt = f(x_opt);

    if ((getUpperBound(domain, x_opt) - f_opt) <= epsilon) {
        return getUpperBound(domain, x_opt);
    }

    auto cmp = [](pair<HyperBox, double> left, pair<HyperBox, double> right) {
        return left.second > right.second;
    };
    priority_queue<pair<HyperBox, double>, std::vector<pair<HyperBox, double>>, decltype(cmp)> pq(cmp);
    pq.push({domain, getUpperBound(domain, x_opt)});

    for (int it = 0; it < maxIter && !pq.empty(); ++it) {
        pair<HyperBox, double> top = pq.top();
        pq.pop();

        HyperBox hbox = top.first;
        double bound = top.second;

        if (bound < f_opt) {
            continue;
        }

        pair<bool, vector<Interval>> grad = gradF_interval(hbox);

        /*
         * If function is differentiable on the entire hyperbox, make use of the gradients.
         * For every dimension j such that partial derivative of the function w.r.t variable x_j is
         * negative/positive on the entire hyperbox, set the value to left or right border of hyperbox immediately.
         */
        bool modifiedBox = false;
        HyperBox newBox = hbox;

        for (size_t i = 0; i < hbox.dim; ++i) {
            if (hbox.it[i].sup > hbox.it[i].inf) {
                if (grad.second[i].inf >= 0) {
                    modifiedBox = true;
                    newBox.it[i] = {hbox.it[i].sup, hbox.it[i].sup};
                } else if (grad.second[i].sup <= 0) {
                    modifiedBox = true;
                    newBox.it[i] = {hbox.it[i].inf, hbox.it[i].inf};
                }
            }
        }

        if (modifiedBox) {
            auto new_x = newBox.center();
            if (f(new_x) > f_opt) {
                f_opt = f(new_x);
            }
            double newBound = getUpperBound(newBox, new_x);
            if (newBound - f_opt > epsilon) {
                pq.push({newBox, newBound});
            }
            continue;
        }

        vector<HyperBox> newBoxes = branching(hbox, p, grad);

        // Evaluation of sub-problems
        for (HyperBox chunkBox : newBoxes) {
            vector<double> x = chunkBox.center();
            double f_x = f(x);

            if (f_x > f_opt) {
                f_opt = f_x;
            }

            double chunkBound = getUpperBound(chunkBox, x);
            if (chunkBound - f_opt > epsilon) {
                pq.push({chunkBox, chunkBound});
            }
        }
    }

    double ret = f_opt + epsilon;
    while (!pq.empty()) {
        pair<HyperBox, double> top = pq.top();
        pq.pop();
        ret = max(ret, top.second);
    }
    return ret;
}

double LipschitzFunction::minimize(double epsilon, int p, Statistics& counter, int maxIter) const  {
    LipschitzFunction selfNegative = -(*this);
    return -selfNegative.maximize(epsilon, p, counter, maxIter);
}

LipschitzFunction LipschitzFunction::getLinear(HyperBox domain, vector<double> weights, double bias, int degree) {
    assert(domain.dim * degree == weights.size());

    std::function<double(vector<double>)> f = [domain, weights, bias, degree](vector<double> params) {
        assert(params.size() == domain.dim);
        double ret = bias;
        for (size_t i = 0; i < domain.dim; ++i) {
            for (int j = 0; j < degree; ++j) {
                ret += weights[i * degree + j] * pow(params[i], j + 1);
            }
        }
        return ret;
    };

    std::function<pair<bool, vector<Interval>>(HyperBox)> gradF_interval = [domain, weights, degree](HyperBox hbox) {
        vector<Interval> ret;
        for (size_t i = 0; i < domain.dim; ++i) {
            Interval d(0, 0);
            for (int j = 0; j < degree; ++j) {
                d = d + (j + 1) * weights[i * degree + j] * hbox.it[i].pow(j);
            }
            ret.push_back(d);
        }
        return make_pair(true, ret);
    };

    vector<double> linWeights;
    for (size_t i = 0; i < domain.dim; ++i) {
        linWeights.push_back(weights[i * degree]);
    }
    return LipschitzFunction(f, domain, gradF_interval);
}
