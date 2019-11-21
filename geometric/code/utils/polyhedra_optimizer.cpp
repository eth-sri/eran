#include "utils/polyhedra_optimizer.h"
#include <random>

pair<vector<double>, double> findLower(
        GRBEnv env,
        LipschitzFunction lf,
        std::default_random_engine generator,
        int k,
        double eps,
        int degree, 
        Statistics& counter) {
    /**
     * @param env Gurobi env
     * @param lf Lipschitz continuous function for which lower bound should be found
     * @param generator random numbers generator
     * @param k number of points to sample initially for constraints
     * @param eps epsilon for Lipschitz optimization
     * @param degree degree of the polyhedra constraint
     * @param counter: object that counts how many steps the optimizer performes
     */
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_OutputFlag, 0);

    vector<GRBVar> w;
    for (size_t i = 0; i < lf.domain.dim; ++i) {
        for (int j = 0; j < degree; ++j) {
            w.push_back(model.addVar(-Constants::MAX_COEFF, Constants::MAX_COEFF, 0.0, GRB_CONTINUOUS));
        }
    }
    GRBVar bias = model.addVar(-Constants::MAX_COEFF, Constants::MAX_COEFF, 0.0, GRB_CONTINUOUS, "bias");

    vector<PointD> samples = lf.domain.sample(k, generator);
    GRBLinExpr obj = 0;

    for (const PointD& sample : samples) {
        double evalF = lf.f(sample);

        // Add constraint w[0]*sample.x[0] + bias <= evalF
        GRBLinExpr evalPoly = bias;
        for (size_t i = 0; i < lf.domain.dim; ++i) {
            for (int j = 0; j < degree; ++j) {
                evalPoly += w[i * degree + j] * pow(sample.x[i], j + 1);
            }
        }

        model.addConstr(evalPoly <= evalF);
        obj += evalF - evalPoly;
    }
    
    model.setObjective(obj, GRB_MINIMIZE);
    model.optimize();

    // Now find the maximum global violation and adjust bias
    double biasOpt = bias.get(GRB_DoubleAttr_X);
    vector<double> wOpt;
    for (size_t i = 0; i < lf.domain.dim; ++i) {
        for (int j = 0; j < degree; ++j) {
            wOpt.push_back(w[i * degree + j].get(GRB_DoubleAttr_X));
        }
    }
    auto lowF = LipschitzFunction::getLinear(lf.domain, wOpt, biasOpt, degree);
    double maxViolation = (lowF - lf).maximize(eps, 3, counter);
    biasOpt -= maxViolation;
    
    return {wOpt, biasOpt};
}

pair<vector<double>, double> findUpper(
        GRBEnv env,
        LipschitzFunction lf,
        std::default_random_engine generator,
        int k,
        double eps,
        int degree,
        Statistics& counter) {
  vector<double> wUpper;
  double biasUpper;
  std::tie(wUpper, biasUpper) = findLower(env, -lf, generator, k, eps, degree, counter);
  biasUpper *= -1;
  for (size_t i = 0; i < wUpper.size(); ++i) {
    wUpper[i] *= -1;
  }
  return {wUpper, biasUpper};
}
