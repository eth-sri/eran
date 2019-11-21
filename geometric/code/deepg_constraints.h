

#include "domains/polyhedra.h"
#include "utils/lipschitz.h"
#include "utils/constants.h"
#include "transforms/transformation.h"
#include "transforms/interpolation.h"
#include "transforms/parser.h"
#include <vector>
#include <string>

extern "C" {
using namespace std;

class TransformAttackContainer{

    public:
        // ---- Constructors
        TransformAttackContainer(double noise,
                                    int n_splits,
                                    int inside_splits,
                                    int nRows,
                                    int nCols,
                                    int nChannels,
                                    string calc_type,
                                    string images,
                                    string transformName,
                                    string pixelTransformName,
                                    bool debug,
                                    vector<vector<double> > splitPoints);

        // ---- Members set by Constructor
        double noise;
        int n_splits, inside_splits, nRows, nCols, nChannels;
        string calc_type, images;
        bool debug;
        vector<vector<double> > splitPoints;
        SpatialTransformation& spatialTransformation;
        PixelTransformation& pixelTransformation;
        const InterpolationTransformation& interpolationTransformation = InterpolationTransformation();

        // ---- Members set by Methods
        vector<vector<double> > transform_vector, attack_vector;

        // ---- Methods
        void setTransformationsAndAttacksFor(int image_number);
};

TransformAttackContainer* getTransformAttackContainer(string config);
}