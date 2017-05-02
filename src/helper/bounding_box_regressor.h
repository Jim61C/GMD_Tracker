#ifndef BOUNDING_BOX_REGRESSOR_h
#define BOUNDING_BOX_REGRESSOR_h
#include <iostream>
#include <vector>

#include <dlib/svm.h>

#include "Constants.h"
#include "Common.h"
#include "helper.h"
#include "bounding_box.h"

using namespace dlib;

typedef matrix<float, BBOX_REGRESSION_FEATURE_LENGTH, 1> sample_type;
typedef linear_kernel<sample_type> kernel_type;

class BoundingBoxRegressor {

public:
    // given the bbox and its Conv features, regress it to refine
    void refineBoundingBox (BoundingBox &bbox, std::vector<float> &feature);

    // given the frame 0 bboxes and their corresponding features, and gt bbox at frame 0, train the 4 models
    void trainModelUsingInitialFrameBboxes(std::vector<std::vector<float> > &features, 
                                           const std::vector<BoundingBox> & bboxes, const BoundingBox &gt);

    // the actual train model function
    void trainModels(std::vector<std::vector<float> > &features, 
                     std::vector<float> &dx_labels, std::vector<float> &dy_labels, 
                     std::vector<float> &dw_labels, std::vector<float> &dh_labels);

private:
    krr_trainer<kernel_type> trainer_;
    // 4 model for dx, dy, dw, dh
    decision_function<kernel_type> test_dx_;
    decision_function<kernel_type> test_dy_;
    decision_function<kernel_type> test_dw_;
    decision_function<kernel_type> test_dh_;
};

#endif