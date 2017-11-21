#ifndef BOUNDING_BOX_REGRESSOR_h
#define BOUNDING_BOX_REGRESSOR_h
#include <iostream>
#include <vector>
#include <Eigen/Dense> // Checkout: if it will be faster if use sparse

#include "Constants.h"
#include "Common.h"
#include "helper.h"
#include "bounding_box.h"

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::MatrixXf;
using Eigen::Matrix;

class BoundingBoxRegressor {

public:
    // given the bbox and its Conv features, regress it to refine
    void refineBoundingBox (BoundingBox &bbox, std::vector<float> &feature);

    // given the frame 0 bboxes and their corresponding features, and gt bbox at frame 0, train the 4 models
    void trainModelUsingInitialFrameBboxes(std::vector<std::vector<float> > &features, 
                                           const std::vector<BoundingBox> & bboxes, const BoundingBox &gt);

    // the actual train model function
    void trainModels(const std::vector<std::vector<float> > &features, 
                     const std::vector<float> &dx_labels, const std::vector<float> &dy_labels, 
                     const std::vector<float> &dw_labels, const std::vector<float> &dh_labels);

private:
    MatrixXd T_;
    MatrixXd T_inv_;
    VectorXd Y_mu_;
    MatrixXd Beta_;

};

#endif