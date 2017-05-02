#include "bounding_box_regressor.h"
#include <assert.h>
#include <math.h>

#define DEBUG_ASSERT_DIFF_FEATURES

void printSamples(std::vector<sample_type> & samples) {
    // DEBUG samples
    for (int i = 0; i < samples.size(); i++) {
        cout << "samples["<< i << "]" << endl;
        for (int j = 0; j < samples[i].nr(); j ++) {
            cout << samples[i](j) << " ";
        }
        cout << endl;
    }
}

void BoundingBoxRegressor::refineBoundingBox (BoundingBox &bbox, std::vector<float> &feature) {

     assert (feature.size() == BBOX_REGRESSION_FEATURE_LENGTH);
     sample_type m;

     for (int i = 0; i < feature.size(); i ++) {
         m(i) = feature[i];
     }

     float dx = test_dx_(m);
     float dy = test_dy_(m);
     float dw = test_dw_(m);
     float dh = test_dh_(m);

     double ctr_x = bbox.get_center_x();
     double ctr_y = bbox.get_center_y();
     double w = bbox.get_width();
     double h = bbox.get_height();

     double refined_ctr_x = dx * w + ctr_x;
     double refined_ctr_y = dy * h + ctr_y;
     double refined_w = exp(dw) * w;
     double refined_h = exp(dh) * h;

     bbox.x1_ = refined_ctr_x - refined_w/2.0;
     bbox.x2_ = refined_ctr_x + refined_w/2.0;
     bbox.y1_ = refined_ctr_y - refined_h/2.0;
     bbox.y2_ = refined_ctr_y + refined_h/2.0;
}

void BoundingBoxRegressor::trainModelUsingInitialFrameBboxes(std::vector<std::vector<float> > &features, const std::vector<BoundingBox> & bboxes, 
                                       const BoundingBox &gt) {
    
    assert (features.size() == bboxes.size());

#ifdef DEBUG_ASSERT_DIFF_FEATURES
    for (int i = 0; i < features.size(); i ++) {
        for (int j = i + 1; j < features.size(); j++) {
            // assert(!equalVector(features[i], features[j]));
            if (equalVector(features[i], features[j])) {
                cout << "candidate bbox " << i << " and " << j << " have the exact same conv5 feature" << endl;
                cout << "IOU between these two boxes:" << bboxes[i].compute_IOU(bboxes[j]) << endl;
            }
        }
    }
#endif


    // construct the 4 labels
    std::vector<float> dx_labels;
    std::vector<float> dy_labels;
    std::vector<float> dw_labels;
    std::vector<float> dh_labels;

    double gt_x = gt.get_center_x();
    double gt_y = gt.get_center_y();
    double gt_w = gt.get_width();
    double gt_h = gt.get_height();

    for (int i = 0; i < bboxes.size(); i ++) {
        double this_bbox_x = bboxes[i].get_center_x();
        double this_bbox_y = bboxes[i].get_center_y();
        double this_bbox_w = bboxes[i].get_width();
        double this_bbox_h = bboxes[i].get_height();


        float this_dx_label = (float)((gt_x -this_bbox_x)/this_bbox_w);
        float this_dy_label = (float)((gt_y -this_bbox_y)/this_bbox_h);
        float this_dw_label = (float)(log(gt_w/this_bbox_w));
        float this_dh_label = (float)(log(gt_h/this_bbox_h));

        dx_labels.push_back(this_dx_label);
        dy_labels.push_back(this_dy_label);
        dw_labels.push_back(this_dw_label);
        dh_labels.push_back(this_dh_label);
    }

    trainModels(features, dx_labels, dy_labels, dw_labels, dh_labels);

}

void BoundingBoxRegressor::trainModels(std::vector<std::vector<float> > &features, 
                    std::vector<float> &dx_labels, std::vector<float> &dy_labels, 
                    std::vector<float> &dw_labels, std::vector<float> &dh_labels) {
    
    // assert make sure data dim match
    assert(features.size() == dx_labels.size());
    assert(features.size() == dy_labels.size());
    assert(features.size() == dw_labels.size());
    assert(features.size() == dh_labels.size());

    if (features.size() > 0) {
        assert(features[0].size() == BBOX_REGRESSION_FEATURE_LENGTH);
    }

    // construct sample
    std::vector<sample_type> samples;
    sample_type m;

    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < BBOX_REGRESSION_FEATURE_LENGTH; j++) {
            m(j) = features[i][j];
        }
        samples.push_back(m);
    }
    
    trainer_ = krr_trainer<kernel_type>();
    trainer_.set_kernel(kernel_type());
    trainer_.set_lambda(LAMBDA);
    test_dx_ = trainer_.train(samples, dx_labels);
    test_dy_ = trainer_.train(samples, dy_labels);
    test_dw_ = trainer_.train(samples, dw_labels);
    test_dh_ = trainer_.train(samples, dh_labels);
}
