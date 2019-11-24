

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
                                    SpatialTransformation& spatialTransformation,
                                    PixelTransformation& pixelTransformation,
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
        vector<double> attack_param_vector;
        vector<vector<double> > transform_vector, attack_image_vector;

        // ---- Methods
        void setTransformationsAndAttacksFor(int image_number);
};
TransformAttackContainer* getTransformAttackContainer(char* config);

void setTransformationsAndAttacksFor(TransformAttackContainer& container, int image_number) {
    container.setTransformationsAndAttacksFor(image_number);
};

double* get_attack_params(TransformAttackContainer& container) {
    return container.attack_param_vector.data();
};

int get_attack_params_dim(TransformAttackContainer& container) {
    return container.attack_param_vector.size();
};

double* get_attack_images(TransformAttackContainer& container) {
    return container.attack_image_vector.data();
};

int get_attack_images_dim(TransformAttackContainer& container) {
    return container.attack_image_vector.size();
};

int* getTransformDimension(TransformAttackContainer& container) {
    int result[2] = {container.transform_vector.size(), container.transform_vector[0].size()};
    return result;
};
}