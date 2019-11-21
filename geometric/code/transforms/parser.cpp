#include "parser.h"

SpatialTransformation* getSpatialTransformation(string& ts) {
    vector<string> tokens;
    while (ts.find(';') != std::string::npos) {
        tokens.push_back(ts.substr(0, ts.find(';')));
        ts = ts.substr(ts.find(';') + 1);
    }
    tokens.push_back(ts);

    vector<SpatialTransformation*> transforms;
    for (const string& s : tokens) {
        size_t pos = s.find('(');
        vector<double> d;

        string name = s.substr(0, pos);
        string rest = s.substr(pos + 1, s.size() - pos - 2);
        rest += ",";

        string curr;
        for (const char& chr : rest) {
            if (chr == ',') {
                d.push_back(stod(curr));
                curr = "";
            } else {
                curr += chr;
            }
        }

        if (name == "Rotation") {
            assert(d.size() == 2);
            d[0] = d[0] / 180 * M_PI;
            d[1] = d[1] / 180 * M_PI;
            auto domain = HyperBox({Interval(d[0], d[1])});
            transforms.push_back(new RotationTransformation(domain));
        } else if (name == "Translation1d") {
            assert(d.size() == 2);
            auto domain = HyperBox({Interval(d[0] * 2.0, d[1] * 2.0)});
            transforms.push_back(new TranslationTransformation1d(domain));
        } else if (name == "Translation") {
            assert(d.size() == 4);
            auto domain = HyperBox({Interval(d[0], d[1]), Interval(d[2], d[3])});
            transforms.push_back(new TranslationTransformation(domain));
        } else if (name == "Scale") {
            assert(d.size() == 2);
            auto domain = HyperBox({Interval(d[0], d[1])});
            transforms.push_back(new ScaleTransformation(domain));
        } else if (name == "Shear") {
            assert(d.size() == 2);
            auto domain = HyperBox({Interval(d[0], d[1])});
            transforms.push_back(new ShearTransformation(domain));
        } else {
            throw "Transformation not supported!";
        }
    }

    if (transforms.size() == 1) {
        return transforms[0];
    }

    CompositionTransform inv_transform = CompositionTransform(transforms);
    return inv_transform.getInverse();
}

PixelTransformation* getPixelTransformation(string& ts) {
    if (ts.empty()) {
        return new PixelIdentity({});
    }
    vector<string> tokens;
    while (ts.find(';') != std::string::npos) {
        tokens.push_back(ts.substr(0, ts.find(';')));
        ts = ts.substr(ts.find(';') + 1);
    }
    tokens.push_back(ts);

    vector<PixelTransformation*> transforms;
    for (const string& s : tokens) {
        size_t pos = s.find('(');
        vector<double> d;

        string name = s.substr(0, pos);
        string rest = s.substr(pos + 1, s.size() - pos - 2);
        rest += ",";

        string curr;
        for (const char& chr : rest) {
            if (chr == ',') {
                d.push_back(stod(curr));
                curr = "";
            } else {
                curr += chr;
            }
        }

        if (name == "Brightness") {
            assert(d.size() == 4);
            auto domain = HyperBox({Interval(d[0], d[1]), Interval(d[2], d[3])});
            transforms.push_back(new BrightnessTransformation(domain));
        } else {
            throw "Transformation not supported!";
        }
    }

    if (transforms.empty()) {
        return new PixelIdentity({});
    }
    if (transforms.size() == 1) {
        return transforms[0];
    }
    return new PixelIdentity({});
}
