#include "transforms/transformation.h"

#pragma once

class CompositionTransform : public SpatialTransformation {
public:

    vector<SpatialTransformation*> transforms;

    explicit CompositionTransform(vector<SpatialTransformation*> transforms) :
            SpatialTransformation(HyperBox()) {
        this->transforms = transforms;
        for (SpatialTransformation* t : transforms) {
            for (Interval it : t->domain.it) {
                this->domain.it.push_back(it);
            }
        }
        this->dim = this->domain.dim = this->domain.it.size();
        
    }

    Pixel<double> transform(const Pixel<double>& pixel, const std::vector<double>& params) const override;
    Pixel<Interval> transform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<vector<Interval>, vector<Interval>> computeGrad(
            const Pixel<Interval>& pixel,
            const std::vector<Interval>& params,
            Interval&, Interval&, Interval&, Interval&) const;
    
    pair<vector<Interval>, vector<Interval>> gradTransform(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<Interval, Interval> dx(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    pair<Interval, Interval> dy(const Pixel<Interval>& pixel, const std::vector<Interval>& params) const override;
    SpatialTransformation* getInverse() override;
};