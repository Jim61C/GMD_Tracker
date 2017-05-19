#include "bounding_box_regressor.h"
#include <assert.h>
#include <math.h>
#include <map>

#define DEBUG_ASSERT_DIFF_FEATURES

void printSamples(MatrixXd X) {
    // DEBUG X
    for (int i = 0; i < X.rows(); i++) {
        cout << "X["<< i << "]" << endl;
        for (int j = 0; j < X.cols(); j ++) {
            cout << X(i, j) << " ";
        }
        cout << endl;
    }
}

// TODO: try different solver and debug
VectorXd Solve(MatrixXd A, VectorXd y, double lambda, string method = "normal") {
    if (method.compare("cholesky") == 0) {
        // Use Cholesky Decomposition
        // (A'A + Lambda*I) x = A'y
        // LL' = (...), LL' x = A'y
        // z = L'x = L.inverse() * A'y
        // x = L'.inverse() * z
        Eigen::LLT<MatrixXd> llt(A.transpose() * A + lambda * MatrixXd::Identity(A.cols(), A.cols()));
        cout << "finished LLT for matrix of size:" << A.cols() << ", " << A.cols() << endl;
        MatrixXd L = llt.matrixL();
        MatrixXd z = L.inverse() * (A.transpose() * y);
        VectorXd x = L.transpose().inverse() * z;
        return x;
    }
    else if (method.compare("normal") == 0) {
        MatrixXd H = lambda * MatrixXd::Identity(A.cols(), A.cols());
        H.noalias() += A.transpose() * A; 
        cout << "finish A^T*A" << endl;
        VectorXd x;
        x.noalias() = H.ldlt().solve(A.transpose() * y);
        return x;
    }
    else if (method.compare("svd") == 0) {
        MatrixXd H = lambda * MatrixXd::Identity(A.cols(), A.cols());
        H.noalias() += A.transpose() * A;
        return H.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(A.transpose() * y);
    }
    else {
        cout << "Unknown solver method" << endl;
        exit(-1);
    }
}

void BoundingBoxRegressor::refineBoundingBox (BoundingBox &bbox, std::vector<float> &feature) {

     assert (feature.size() == BBOX_REGRESSION_FEATURE_LENGTH);
     const int S = BBOX_REGRESSION_FEATURE_LENGTH;

     VectorXd query(S);
     for (int i = 0;i < S; i ++) {
         query(i) = feature[i];
     }

     VectorXd result;
     result.noalias() = query.transpose() * (Beta_.block(0,0,S,4)) + Beta_.row(S);
     result.noalias() = result.transpose() * T_inv_ + Y_mu_.transpose();
     float dx = (float)(result(0));
     float dy = (float)(result(1));
     float dw = (float)(result(2));
     float dh = (float)(result(3));

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

    // Duplicate removal here, if several bboxes have the same feature, use their average/ just one
    std::vector<std::vector<float> > features_unique;
    std::vector<BoundingBox> bboxes_unique;

    std::map<std::vector<float>, std::vector<int> > feature_to_bbox_idxes;
    for (int i = 0; i < features.size(); i++) {
        vector<float> this_feature = features[i];
        if (feature_to_bbox_idxes.find(this_feature) == feature_to_bbox_idxes.end()) {
            vector<int> this_idxes;
            this_idxes.push_back(i);
            feature_to_bbox_idxes[this_feature] = this_idxes;
        }
        else {
            feature_to_bbox_idxes[this_feature].push_back(i);
        }
    }

    for (auto item : feature_to_bbox_idxes) {
        features_unique.push_back(item.first);

        // here use average, TODO: try just use the first one 
        int count = 0;
        double x1_sum = 0;
        double y1_sum = 0;
        double x2_sum = 0;
        double y2_sum = 0;

        for (auto idx : item.second) {
            x1_sum += bboxes[idx].x1_;
            y1_sum += bboxes[idx].y1_;
            x2_sum += bboxes[idx].x2_;
            y2_sum += bboxes[idx].y2_;
            count ++;
        }
        
        BoundingBox averaged_bbox(x1_sum/count, y1_sum/count, x2_sum/count, y2_sum/count);
        bboxes_unique.push_back(averaged_bbox);
    }

    assert (features_unique.size() == bboxes_unique.size());

#ifdef DEBUG_ASSERT_DIFF_FEATURES
    for (int i = 0; i < features_unique.size(); i ++) {
        for (int j = i + 1; j < features_unique.size(); j++) {
            assert(!equalVector(features_unique[i], features_unique[j]));
            assert (features_unique[i].size() == features_unique[j].size());
            // if (equalVector(features_unique[i], features_unique[j])) {
            //     cout << "candidate bbox " << i << " and " << j << " have the exact same conv5 feature" << endl;
            //     cout << "IOU between these two boxes:" << bboxes_unique[i].compute_IOU(bboxes_unique[j]) << endl;
            // }
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

    for (int i = 0; i < bboxes_unique.size(); i ++) {
        double this_bbox_x = bboxes_unique[i].get_center_x();
        double this_bbox_y = bboxes_unique[i].get_center_y();
        double this_bbox_w = bboxes_unique[i].get_width();
        double this_bbox_h = bboxes_unique[i].get_height();


        float this_dx_label = (float)((gt_x -this_bbox_x)/this_bbox_w);
        float this_dy_label = (float)((gt_y -this_bbox_y)/this_bbox_h);
        float this_dw_label = (float)(log(gt_w/this_bbox_w));
        float this_dh_label = (float)(log(gt_h/this_bbox_h));

        dx_labels.push_back(this_dx_label);
        dy_labels.push_back(this_dy_label);
        dw_labels.push_back(this_dw_label);
        dh_labels.push_back(this_dh_label);
    }

    trainModels(features_unique, dx_labels, dy_labels, dw_labels, dh_labels);

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

    // Construct X matrix, shape (D, S + 1), + 1 for bias ; Y matrix, shape (D, 4), dx, dy, dw, dh accordingly
    int D = features.size();
    int S = BBOX_REGRESSION_FEATURE_LENGTH;
    MatrixXd X(D, S + 1);
    MatrixXd Y(D, 4);

    for (int i = 0; i < D; i ++) {
        // set X
        for (int j = 0; j < S; j ++) {
            X(i, j) = (double)(features[i][j]);
        }
        // set Y
        Y(i, 0) = (double)(dx_labels[i]);
        Y(i, 1) = (double)(dy_labels[i]);
        Y(i, 2) = (double)(dw_labels[i]);
        Y(i, 3) = (double)(dh_labels[i]);
    }

    for (int i =0; i < D; i++) {
        for(int j = 0;j < S; j ++) {
            assert(X(i,j) == features[i][j]);
        }
    }

    // set bias
    VectorXd bias = VectorXd::Constant(D, 1);
    X.col(S) = bias;

    for (int i = 0;i < D; i ++) {
        assert(X(i, S) == 1);
    }
    
    // Whitening transform
    Y_mu_ = VectorXd(4);
    Y_mu_ << Y.col(0).mean(), Y.col(1).mean(), Y.col(2).mean(), Y.col(3).mean();

    for (int j = 0; j < 4; j ++) {
        Y.col(j) = Y.col(j) - VectorXd::Constant(Y.col(j).size(), Y_mu_(j));
    }
    // get data covariance matrix
    MatrixXd cov = (Y.transpose() * Y) / D;
    // do eigen value decomposition
    Eigen::EigenSolver<MatrixXd> es(cov);
    MatrixXd diagnal = MatrixXd::Constant(cov.rows(), cov.cols(), 0);
    // take ^1/2, add an epsilon to avoid division by zero later 
    double epsilon = 0.001;
    for (int j = 0; j < 4; j ++) {
        diagnal(j,j) = sqrt(es.eigenvalues()(j).real() + epsilon);
    }
    // let Whitening = P * D^1/2 * P^T
    T_inv_ = es.eigenvectors().real() * diagnal * es.eigenvectors().real().transpose();
    // take diagnal^-1, just take the reciprocal of the diagnal entries
    MatrixXd diagnal_reciprocal = MatrixXd::Zero(diagnal.rows(), diagnal.cols());
    for (int j = 0; j < 4; j++) {
        diagnal_reciprocal(j,j) = 1.0/diagnal(j,j);
    }
    // take Whitening^-1
    T_ = es.eigenvectors().real() * diagnal_reciprocal * es.eigenvectors().real().transpose();
    Y = Y * T_;

    // train 4 regressors for dx, dy, dw, dh
    MatrixXd Beta(S+1, 4); // the regressed parameters
    for (int j = 0; j < 4; j ++) {
        cout << "solve for model " << j << endl;
        Beta.col(j) = Solve(X, Y.col(j), LAMBDA, "normal");
    }

    // save the models
    Beta_ = Beta;
}
