#include "regressor.h"
#include "math.h"

#include "helper/high_res_timer.h"
#include <algorithm>

// Credits:
// This file was mostly taken from:
// https://github.com/BVLC/caffe/tree/master/examples/cpp_classification

using caffe::Blob;
using caffe::Net;
using caffe::Layer;
using caffe::LayerParameter;
using caffe::ParamSpec;
using std::string;
using namespace std;

// #define DEBUG_MDNET_INPUT
// #define DEBUG_CONV3_FEATURE
// #define DEBUG_FREEZE_LAYER
// #define DEBUG_GETPROBOUTPUT
// #define LOG_TIME
// #define DEBUG_PRE_FORWARDFAST
// #define DEBUG_PRE_FORWARDFAST_IMAGE_SCALE
// #define DEBUG_PREPROCESS_SAMPLE

// #define DEBUG_CANDIDATE_IN_PREDICTFAST
// #define INSPECT_TARGET_IN_PREFORWARD
// #define DEBUG_TARGET_PREDICT_FAST
// #define DEBUG_MINIMUM_SCALE_REQUIRED

// We need 2 inputs: one for the current frame and one for the previous frame.
const int kNumInputs = 2;

Regressor::Regressor(const string& deploy_proto,
                     const string& caffe_model,
                     const int gpu_id,
                     const int num_inputs,
                     const bool do_train,
                     const int K)
  : num_inputs_(num_inputs),
    deploy_proto_(deploy_proto), 
    caffe_model_(caffe_model),
    modified_params_(false),
    K_(K),
    hrt_("Regressor")

{
  SetupNetwork(deploy_proto, caffe_model, gpu_id, do_train);
}

Regressor::Regressor(const string& deploy_proto,
                     const string& caffe_model,
                     const int gpu_id,
                     const int num_inputs,
                     const bool do_train)
  : num_inputs_(num_inputs),
    deploy_proto_(deploy_proto), 
    caffe_model_(caffe_model),
    modified_params_(false),
    K_(-1),
    hrt_("Regressor")

{
  SetupNetwork(deploy_proto, caffe_model, gpu_id, do_train);
}

Regressor::Regressor(const string& deploy_proto,
                     const string& caffe_model,
                     const int gpu_id,
                     const bool do_train)
  : num_inputs_(kNumInputs),
    deploy_proto_(deploy_proto), 
    caffe_model_(caffe_model),
    modified_params_(false),
    K_(-1),
    hrt_("Regressor")
{
  SetupNetwork(deploy_proto, caffe_model, gpu_id, do_train);
}

void Regressor::SetupNetwork(const string& deploy_proto,
                             const string& caffe_model,
                             const int gpu_id,
                             const bool do_train) {
#ifdef CPU_ONLY
  printf("Setting up Caffe in CPU mode\n");
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  printf("Setting up Caffe in GPU mode with ID: %d\n", gpu_id);
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

  if (do_train) {
    printf("Setting phase to train\n");
    net_.reset(new Net<float>(deploy_proto, caffe::TRAIN));
  } else {
    printf("Setting phase to test\n");
    net_.reset(new Net<float>(deploy_proto, caffe::TEST));
  }

  if (caffe_model != "NONE") {
    net_->CopyTrainedLayersFrom(caffe_model_);
  } else {
    printf("Not initializing network from pre-trained model\n");
  }

  //CHECK_EQ(net_->num_inputs(), num_inputs_) << "Network should have exactly " << num_inputs_ << " inputs.";
  // CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0]; // assume 0th input is for image input

  printf("Network image size: %d, %d\n", input_layer->width(), input_layer->height());

  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  // Load the binaryproto mean file.
  SetMean();

  // if (do_train && K_ != -1) {
  //   // Only if training and model has K domains, Lock the domain specific layers, will be opened each time during training
  //   LockDomainLayers();
  // }
}

void Regressor::SetMean() {
  // Set the mean image.
  mean_ = cv::Mat(input_geometry_, CV_32FC3, mean_scalar);
}

void Regressor::Init() {
  if (modified_params_ ) {
    printf("Reloading new params\n");
    net_->CopyTrainedLayersFrom(caffe_model_);
    modified_params_ = false;
  }
}

void Regressor::Reset() {
  net_.reset(); // decrease reference count
  net_.reset(new Net<float>(deploy_proto_, caffe::TRAIN));
  printf("In Regressor, Reset net_\n");
  net_->CopyTrainedLayersFrom(caffe_model_);
}

void Regressor::LockDomainLayers() {
  // assert K_ is the same as number of loss layers in net_
  assert (net_->output_blob_indices().size() == K_);

  for (int i = 0; i< K_;i ++) {
    std::string this_layer_name = FREEZE_LAYER_PREFIX + std::to_string(i);
    // unlock this layer
    const boost::shared_ptr<Layer<float> > layer_pt = net_->layer_by_name(this_layer_name);

    cout << "layer_pt->param_propagate_down(0):" << layer_pt->param_propagate_down(0) << endl;
    cout << "layer_pt->param_propagate_down(1):" << layer_pt->param_propagate_down(1) << endl;
    if (layer_pt->param_propagate_down(0)) {
      layer_pt->set_param_propagate_down(0, false);
    }
    if (layer_pt->param_propagate_down(1)) {
      layer_pt->set_param_propagate_down(1, false);
    }
    const LayerParameter & layer_parameter = layer_pt->layer_param();
    cout << "layer_parameter.propagate_down_size():" << layer_parameter.propagate_down_size() << endl;
    cout << "layer_parameter.param_size():" << layer_parameter.param_size() << endl;
  }
}

void Regressor::Regress(const cv::Mat& image_curr,
                        const cv::Mat& image, const cv::Mat& target,
                        BoundingBox* bbox) {
  assert(net_->phase() == caffe::TEST);

  // Estimate the bounding box location of the target object in the current image.
  std::vector<float> estimation;
  Estimate(image, target, &estimation);

  // Wrap the estimation in a bounding box object.
  *bbox = BoundingBox(estimation);
}


void Regressor::PreForwardFast(const cv::Mat image_curr, 
                               const std::vector<BoundingBox> &candidate_bboxes,
                               const cv::Mat & image,
                               const cv::Mat & target) {
  
  const vector<string> & layer_names = net_->layer_names();
  
  Blob<float>* input_target = net_->input_blobs()[TARGET_NETWORK_INPUT_IDX];
  input_target->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  // Process the inputs so we can set them.
  std::vector<cv::Mat> target_channels;
  WrapInputLayerGivenIndex(&target_channels, TARGET_NETWORK_INPUT_IDX);
  // Set the t-1 target
  Preprocess(target, &target_channels);

  int layer_conv1_idx = FindLayerIndexByName(layer_names, "conv1");
  int layer_pool6_idx = FindLayerIndexByName(layer_names, "pool6");

  // Perform a forward-pass in the network.
  net_->ForwardFromTo(layer_conv1_idx, layer_pool6_idx);
  std::vector<cv::Mat> pool6_image;
  WrapOutputBlob("pool6", &pool6_image);

#ifdef INSPECT_TARGET_IN_PREFORWARD
  vector<cv::Mat> target_splitted;
  WrapOutputBlob("target", &target_splitted);
  cv:: Mat target_merged;
  cv::merge(target_splitted, target_merged);
  cv::add(target_merged, cv::Mat(target_merged.size(), CV_32FC3, mean_scalar), target_merged);
  target_merged.convertTo(target_merged, CV_8UC3);
  imshow("t-1 target in PreforwardFast before reshape target input", target_merged);
  waitKey(1);
#endif

  //------------------------- Now Forward the ROIs -------------------
  // Reshape target input
  input_target->Reshape(candidate_bboxes.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  
  // Process the candidate, full image's input, i.e., image_curr, just one! Also record the scales
  int im_min_size = std::min(image_curr.size().width, image_curr.size().height);
  int im_max_size = std::max(image_curr.size().width, image_curr.size().height);

  double scale_curr = TARGET_SIZE / im_min_size;
  if (round(scale_curr * im_max_size) > MAX_SIZE) {
    scale_curr = MAX_SIZE / im_max_size;
  }

  cv::Mat image_scaled;
  cv::resize(image_curr, image_scaled, cv::Size(), scale_curr, scale_curr);

#ifdef DEBUG_PRE_FORWARDFAST_IMAGE_SCALE 
  cout << "scale_curr:" << scale_curr << endl;
  cv::imshow("image_scaled:", image_scaled);
#endif

  // Reshape Candidate input, full image's input, i.e., image_curr
  Blob<float>* input_candidate = net_->input_blobs()[CANDIDATE_NETWORK_INPUT_IDX];
  input_candidate->Reshape(1, num_channels_,
                       image_scaled.size().height, image_scaled.size().width);

  // Reshape the labels
  Blob<float> * input_label_blob = net_->input_blobs()[LABEL_NETWORK_INPUT_IDX];
  const size_t num_labels = candidate_bboxes.size();
  // reshape to (|R|, 1)
  vector<int> shape_label;
  shape_label.push_back(num_labels);
  shape_label.push_back(1);
  input_label_blob->Reshape(shape_label);

  // Reshape the rois
  Blob<float> * input_rois_blob = net_->input_blobs()[ROIS_NETWORK_INPUT_IDX];
  const size_t num_rois = candidate_bboxes.size();
  // reshape to (|R|, 5)
  vector<int> shape_rois;
  shape_rois.push_back(num_rois);
  shape_rois.push_back(5);
  input_rois_blob->Reshape(shape_rois);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Put image_curr
  std::vector<cv::Mat> image_curr_channels;
  WrapInputLayerGivenIndex(&image_curr_channels, CANDIDATE_NETWORK_INPUT_IDX);

  // Set the inputs to the network.
  Preprocess(image_scaled, &image_curr_channels, true); // set retain the original image size

  // Put the ROIs
  set_rois(candidate_bboxes, scale_curr);

  // ROI poolings
  int layer_conv1_c_idx = FindLayerIndexByName(layer_names, "conv1_c");
  int layer_pool6_c_idx = FindLayerIndexByName(layer_names, "pool6_c");
  net_->ForwardFromTo(layer_conv1_c_idx, layer_pool6_c_idx);

#ifdef DEBUG_PRE_FORWARDFAST
  std::vector<std::vector<cv::Mat> > pool6_c_features;
  WrapOutputBlob("pool6_c", &pool6_c_features);

  // check if the 256 maps are all the same across candidates
  for (int m = 0; m < candidate_bboxes.size(); m ++) {
    for (int n = m + 1; n < candidate_bboxes.size(); n++) {
      bool is_different = false;
      for (int j = 0; j < 256; j++) {
        if(!equalMat(pool6_c_features[m][j], pool6_c_features[n][j])) {
          is_different = true;
        }
      }
      if (!is_different) {
        cout << "candidate " << m << " and " << n << "have the same pool5_c feature"<< endl;
      }
    }
  }
#endif

  // ------------------ Duplicate the pool5 features mannualy for candidate_bboxes.size() times -----------------
  std::vector<std::vector<cv::Mat> > pool6_channels;

  WrapBlobByNameBatch("pool6", &pool6_channels);

  PreprocessDuplicateIn(pool6_image, &pool6_channels);

#ifdef INSPECT_TARGET_IN_PREFORWARD
  bool inspect_target_after_reshape = false;
  if (inspect_target_after_reshape) {
    vector<vector<cv::Mat> > target_splitted;
    WrapOutputBlob("target", &target_splitted);
    cv:: Mat target_merged;
    cv::merge(target_splitted[0], target_merged);
    cv::add(target_merged, cv::Mat(target_merged.size(), CV_32FC3, mean_scalar), target_merged);
    target_merged.convertTo(target_merged, CV_8UC3);
    imshow("t-1 target end of PreForwardFast", target_merged);
    waitKey(0);
  }
#endif

}

// Get the BBox Conv Features used for BoundingBox Regression
void Regressor::GetBBoxConvFeatures(const cv::Mat& image_curr,
                       const std::vector<BoundingBox> &candidate_bboxes, std::vector <std::vector<float> > &features) {
    int batch_size = INNER_BATCH_SIZE;
    int num_batches = (int)(ceil(candidate_bboxes.size()/float(batch_size)));
    for (int i = 0; i < num_batches; i++) {
      vector<BoundingBox> this_candidates(candidate_bboxes.begin() + i * batch_size, 
                                          candidate_bboxes.begin() + std::min((i+1) * batch_size, (int)(candidate_bboxes.size())));
      
      cout << "In GetBBoxConvFeatures, this_candidates.size():" << this_candidates.size() << endl;
      // set input
      vector<cv::Mat> this_candidate_images;
      for (auto bbox: this_candidates) {
        cv::Mat out;
        bbox.CropBoundingBoxOutImage(image_curr, &out);
        this_candidate_images.push_back(out);
      }

      SetCandidates(this_candidate_images);
      const vector<string> & layer_names = net_->layer_names();
      int layer_relu3_idx = FindLayerIndexByName(layer_names, "relu3");
      net_->ForwardTo(layer_relu3_idx);

      // get the conv3 data blob features
      vector<vector<float> > output_features;
      WrapOutputBlob("conv3", &output_features);
      features.insert(features.end(), output_features.begin(), output_features.begin() + output_features.size());
    }
}

void Regressor::PredictFast(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, const BoundingBox & bbox_prev, 
                       BoundingBox* bbox,
                       std::vector<float> *return_probabilities, 
                       std::vector<int> *return_sorted_indexes, 
                       double sd_trans,
                       int cur_frame) {
#ifdef LOG_TIME
  // hrt timer
  hrt_.reset();
  hrt_.start();
#endif

  // TODO: load another net with phase TEST and use Net::ShareTrainedLayersWith() to share weights with the train net
  // Or: Just use solver_'s net_ and test_nets_[0] which are shared weights
#ifdef DEBUG_TARGET_PREDICT_FAST
  imshow("target in Predict Fast", target);
  waitKey(0);
#endif


  // feed in the image_curr as batch size 1 and the bboxes as rois
  set_candidate_images(image_curr, candidate_bboxes);
  const vector<string> & layer_names = net_->layer_names();
  // forward to the flatten_fc6 layer, shape (B, 2)
  int layer_flatten_fc6_idx = FindLayerIndexByName(layer_names, "flatten_fc6");
  net_->ForwardTo(layer_flatten_fc6_idx);

  vector<float> probabilities;
  GetProbOutput(&probabilities);

  vector<float> positive_probabilities;
  for(int i = 0; i < candidate_bboxes.size(); i++) {
    positive_probabilities.push_back(probabilities[2*i+1]);
  }

  assert (positive_probabilities.size() == candidate_bboxes.size());


#ifdef ADD_DISTANCE_PENALTY
  vector<float> distance_scores;
  for(int i = 0; i < candidate_bboxes.size(); i++) {
    double d = candidate_bboxes[i].compute_center_distance(bbox_prev);
    double w = bbox_prev.x2_ - bbox_prev.x1_;
    double h = bbox_prev.y2_ - bbox_prev.y1_;
    double r = round((w+h)/2.0);
    double max_shift = sqrt(pow(KEEP_SD * sd_trans * r, 2) + pow(KEEP_SD * sd_trans * r, 2) + DISTANCE_PENALTY_PAD);
    double this_distance_score = cos(d/ (max_shift) * PI/2);
    if (this_distance_score < 0) {
      cout << "out of range, problem, inspect" << endl;
      exit(-1);
    }
    distance_scores.push_back(this_distance_score);
  }
#endif

#ifdef DEBUG_CANDIDATE_IN_PREDICTFAST
  vector<cv::Mat> image_curr_scaled_splitted;
  WrapOutputBlob("candidate", &image_curr_scaled_splitted);
  cv::Mat image_curr_scale;
  cv::merge(image_curr_scaled_splitted, image_curr_scale);

  vector<cv::Mat> target_in;
  WrapOutputBlob("target", &target_in);
  cv::Mat target_origin;
  cv::merge(target_in, target_origin);

  cv::add(target_origin, cv::Mat(target_origin.size(), CV_32FC3, mean_scalar), target_origin);
  target_origin.convertTo(target_origin, CV_8UC3);
 
  cv::Mat image_curr_scale_origin;
  cv::add(image_curr_scale, cv::Mat(image_curr_scale.size(), CV_32FC3, mean_scalar), image_curr_scale_origin);
  image_curr_scale_origin.convertTo(image_curr_scale_origin, CV_8UC3);

  vector<float> rois_in;
  GetFeatures("rois", &rois_in);

  vector <BoundingBox> bboxes_in;
  for (int i = 0; i < rois_in.size(); i+= 5) {
    // each rois in the rois_in memory is [batch_id, x1, y1, x2, y2]
    BoundingBox this_bbox(rois_in[i + 1],
                          rois_in[i + 2],
                          rois_in[i + 3],
                          rois_in[i + 4]);
    bboxes_in.push_back(this_bbox);
  }

  assert(positive_probabilities.size() == bboxes_in.size()); 

  double min_prob = 1.0;
  double max_prob = 0.0;
  for (int i = 0; i < bboxes_in.size(); i ++) {
    if (positive_probabilities[i] > max_prob) {
      max_prob = positive_probabilities[i];
    }
    if (positive_probabilities[i] < min_prob) {
      min_prob = positive_probabilities[i];
    }
  }

  double min_color = 0;
  double max_color = 255;

  if (cur_frame >= 0) {
    for (int i = 0; i < bboxes_in.size(); i++) {
      if (positive_probabilities[i] > 0.2) {
        Mat this_im_show = image_curr_scale_origin;
        float this_color = (int)((positive_probabilities[i] - min_prob)/(max_prob - min_prob) * (max_color - min_color) + min_color);
        bboxes_in[i].Draw(this_color, 0 , 0, &this_im_show);
        cv::imshow("t-1 target", target_origin);
        
        cv::putText(this_im_show, "box" + std::to_string(i) + ":" + std::to_string(positive_probabilities[i]),
                    cv::Point(bboxes_in[i].get_center_x(), bboxes_in[i].get_center_y()), 
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
#ifdef ADD_DISTANCE_PENALTY
        cv::putText(this_im_show, "box" + std::to_string(i) + ":" + std::to_string(distance_scores[i]),
            cv::Point(bboxes_in[i].get_center_x(), bboxes_in[i].get_center_y() + 15.0), 
            FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
#endif
        cv::imshow("candidate rois on scaled image:", this_im_show);
        cv::waitKey(0);
      }
    }
  }

#endif

#ifdef ADD_DISTANCE_PENALTY
  // add distance penalty to the positive_probabilities
  for (int i = 0; i < positive_probabilities.size(); i++) {
    positive_probabilities[i] *= distance_scores[i];
  }
#endif

#ifdef DEBUG_MDNET_INPUT
  vector<vector<cv::Mat> > input_candidate_images_splitted;
  WrapOutputBlob("candidate", &input_candidate_images_splitted);
  
  vector<cv::Mat> input_candidate_images;
  for (int b = 0; b < input_candidate_images_splitted.size(); b++) {
    cv::Mat candidate_image;
    cv::merge(input_candidate_images_splitted[b], candidate_image); 
    cv::add(candidate_image, cv::Mat(candidate_image.size(), CV_32FC3, mean_scalar), candidate_image);
    candidate_image.convertTo(candidate_image, CV_8UC3);
    input_candidate_images.push_back(candidate_image);
  }

  for (int b = 0; b < input_candidate_images.size(); b++) {
    cv::imshow("input candidate" + std::to_string(1), input_candidate_images[b]);
    cv::waitKey(1);
  }

#endif

  // initialize original index locations
  vector<int> idx(positive_probabilities.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&positive_probabilities](int i1, int i2) {return positive_probabilities[i1] > positive_probabilities[i2];});

  double x1_weighted = 0;
  double y1_weighted = 0;
  double x2_weighted = 0;
  double y2_weighted = 0;
  double denominator = 0;

  for (int i = 0 ; i< TOP_ESTIMATES; i ++) {
    double this_prob = positive_probabilities[idx[i]];
    
    x1_weighted += candidate_bboxes[idx[i]].x1_ * this_prob;
    y1_weighted += candidate_bboxes[idx[i]].y1_ * this_prob;
    x2_weighted += candidate_bboxes[idx[i]].x2_ * this_prob;
    y2_weighted += candidate_bboxes[idx[i]].y2_ * this_prob;

    denominator += this_prob;
  }

  x1_weighted /= denominator;
  y1_weighted /= denominator;
  x2_weighted /= denominator;
  y2_weighted /= denominator;

  *bbox = BoundingBox(x1_weighted, y1_weighted, x2_weighted, y2_weighted);
  *return_probabilities = positive_probabilities;
  *return_sorted_indexes = idx;

#ifdef LOG_TIME
  hrt_.stop();
  cout << "time spent for PredictFast: " << hrt_.getMilliseconds() << " ms" << endl;
#endif
}

void Regressor::Predict(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, 
                       BoundingBox* bbox,
                       std::vector<float> *return_probabilities, 
                       std::vector<int> *return_sorted_indexes) {
#ifdef LOG_TIME
  // hrt timer
  hrt_.reset();
  hrt_.start();
#endif

  // Prepare the corresponding vector<cv::Mat> for images, targets, candidates to feed into network
  std::vector<cv::Mat> images_flattened;
  std::vector<cv::Mat> targets_flattened;
  std::vector<cv::Mat> candidates_flattened;

  // flatten
  for (int i = 0; i <candidate_bboxes.size(); i++) {
    // Crop the candidate
    const BoundingBox &this_box = candidate_bboxes[i];
    cv::Mat this_candidate;
    this_box.CropBoundingBoxOutImage(image_curr, &this_candidate);

    candidates_flattened.push_back(this_candidate);
    images_flattened.push_back(image.clone());
    targets_flattened.push_back(target.clone());
  }

  vector<float> probabilities;
  Estimate(images_flattened, targets_flattened, candidates_flattened, &probabilities);
  assert (probabilities.size() == images_flattened.size() * 2); // since binary classification, prob[1] is POSITIVE probability

  int best_idx = -1;
  float best_prob = 0;
  vector<float> positive_probabilities;
  for(int i = 0; i < images_flattened.size(); i++) {
    positive_probabilities.push_back(probabilities[2*i+1]);
    // if (probabilities[2*i+1] > best_prob) {
    //   best_prob = probabilities[2*i+1];
    //   best_idx = i;
    // }
  }

  // initialize original index locations
  vector<int> idx(positive_probabilities.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&positive_probabilities](int i1, int i2) {return positive_probabilities[i1] > positive_probabilities[i2];});

  double x1_weighted = 0;
  double y1_weighted = 0;
  double x2_weighted = 0;
  double y2_weighted = 0;
  double denominator = 0;

  for (int i = 0 ; i< TOP_ESTIMATES; i ++) {
    double this_prob = positive_probabilities[idx[i]];
    
    x1_weighted += candidate_bboxes[idx[i]].x1_ * this_prob;
    y1_weighted += candidate_bboxes[idx[i]].y1_ * this_prob;
    x2_weighted += candidate_bboxes[idx[i]].x2_ * this_prob;
    y2_weighted += candidate_bboxes[idx[i]].y2_ * this_prob;

    denominator += this_prob;
  }

  x1_weighted /= denominator;
  y1_weighted /= denominator;
  x2_weighted /= denominator;
  y2_weighted /= denominator;

  // assert (best_idx != -1);
  // *bbox = BoundingBox(candidate_bboxes[best_idx]);
  *bbox = BoundingBox(x1_weighted, y1_weighted, x2_weighted, y2_weighted);
  *return_probabilities = positive_probabilities;
  *return_sorted_indexes = idx;

#ifdef LOG_TIME
  //report time
  hrt_.stop();
  cout << "time spent for Predict: " << hrt_.getMilliseconds() << " ms" << endl;
#endif
}

int Regressor::FindLayerIndexByName( const vector<string> & layer_names, const string & target) {
  for (int i = 0; i< layer_names.size(); i++) {
    if (layer_names[i].compare(target) == 0) {
      return i;
    }
  }

  return -1;
}

void Regressor::Estimate(std::vector<cv::Mat> &images_flattened,
                           std::vector<cv::Mat> &targets_flattened,
                           std::vector<cv::Mat> &candidates_flattened,
                           std::vector<float>* output) {

  // // DEBUG

  // std::vector<std::vector<cv::Mat> > pool5_cs;
  // std::vector<std::vector<float> > pool5_cs_flattened;

  // for (int i = 0; i < images_flattened.size();i++) {
  //   std::vector<cv::Mat> this_image_flattened;
  //   std::vector<cv::Mat> this_targets_flattened;
  //   std::vector<cv::Mat> this_candidates_flattened;

  //   this_image_flattened.push_back(images_flattened[i]);
  //   this_targets_flattened.push_back(targets_flattened[i]);
  //   this_candidates_flattened.push_back(candidates_flattened[i]);

  //   SetImages(this_image_flattened, this_targets_flattened);

  
  //   SetCandidates(this_candidates_flattened);

  //   if (net_->input_blobs().size() == 4) {
  //     // get the input blob for labels, reshape to include batch number
  //     Blob<float> * input_label_blob = net_->input_blobs()[3];

  //     // reshape to batch size
  //     vector<int> shape;
  //     shape.push_back(1);
  //     shape.push_back(1);
  //     input_label_blob->Reshape(shape);
  //   }

  //   // Forward dimension change to all layers.
  //   net_->Reshape();

  //   // Perform a forward-pass in the network, until fc8 layer
  //   // net_->ForwardPrefilled();
  //   int layers_size = net_->layers().size();
  //   net_->ForwardTo(layers_size-2); // forward until fc8 layer, inclusive

    
  //   vector<float> temp_fc8;
  //   GetFeatures("fc8", &temp_fc8);
  //   cout << "candidate" << i << ", fc8: " <<temp_fc8[0] << ", " << temp_fc8[1] << endl;

  //   std::vector<cv::Mat> temp_pool5_c;
  //   WrapOutputBlob("roi_pool5_c", &temp_pool5_c);
  //   pool5_cs.push_back(temp_pool5_c);

  //   std::vector<float> temp_pool5_c_flattened;
  //   GetFeatures("roi_pool5_c", &temp_pool5_c_flattened);
  //   pool5_cs_flattened.push_back(temp_pool5_c_flattened);
    
  // }

  // for (int j = 0; j < 256; j++) {
  //   // check if the 256 maps are all the same across candidates
  //   for (int m = 0; m < images_flattened.size(); m ++) {
  //     for (int n = m + 1; n < images_flattened.size(); n++) {
  //       if(!equalMat(pool5_cs[m][j], pool5_cs[n][j])) {
  //         cout << "candidate " << m << " and " << n << "have different pool 5 map at channel" << j << endl;
  //       }
  //     }
  //   }
  // }

  // for (int i = 0; i < pool5_cs_flattened.size(); i ++) {
  //   for (int j = 0; j < pool5_cs_flattened.size(); j++) {
  //     if (!equalVector(pool5_cs_flattened[i], pool5_cs_flattened[j])) {
  //       cout << "candidate " << i << " and " << j << "have different pool5 feature map flattened" << endl;
  //     }
  //   }
  // }


  assert(images_flattened.size() == targets_flattened.size());
  assert(images_flattened.size() == candidates_flattened.size());

  // Set the image and target, Input reshape, WrapInputLayer and Preprocess, input[0]->targets and input[1]->images
  SetImages(images_flattened, targets_flattened);

  // Set the candidates, Input reshape, WrapInputLayer and Preprocess, input[2]
  SetCandidates(candidates_flattened);

  // Reshape the labels if it is there
  if (net_->input_blobs().size() == 4) {
    // get the input blob for labels, reshape to include batch number
    Blob<float> * input_label_blob = net_->input_blobs()[3];
    const size_t num_labels = images_flattened.size();

    // reshape to batch size
    vector<int> shape;
    shape.push_back(num_labels);
    shape.push_back(1);
    input_label_blob->Reshape(shape);
  }

  // Forward dimension change to all layers.
  net_->Reshape();

  // Perform a forward-pass in the network, until fc8 layer
  // net_->ForwardPrefilled();
  int layers_size = net_->layers().size();
  net_->ForwardTo(layers_size-2); // forward until fc8 layer, inclusive
  
  // const vector<string> & layer_names = net_->layer_names();
  // const vector<boost::shared_ptr<Layer<float> > > & layer_ptrs = net_->layers();
  // for (int i = 0; i< layer_ptrs.size(); i++) {
  //   const LayerParameter & layer_parameter = layer_ptrs[i]->layer_param();
  //   cout << "layer " << i << "'s name: " << layer_parameter.name()<< endl;
  //   cout << "layer_names["<< i << "]: "<< layer_names[i]<< endl;

  //   int found_idx = FindLayerIndexByName(layer_names, layer_parameter.name());
  //   assert (i == found_idx);
  // }

  // // FOR DEBUG PREDICTFAST
  // std::vector<std::vector<cv::Mat> > pool5_targets;
  // WrapOutputBlob("pool5", &pool5_targets);
  // // cout << "First candidate, pool5[0] :\n" << pool5_targets[0][0] << endl;
  // // cout << "First candidate, pool5[1] :\n" << pool5_targets[0][1] << endl;
  // for (int i = 0; i < pool5_targets[0].size(); i ++) {
  //   cout << pool5_targets[0][i] << endl;
  // }

  // std::vector<std::vector<cv::Mat> > pool5_images;
  // WrapOutputBlob("pool5_p", &pool5_images);
  // // cout << "First candidate, pool5_p[0] :\n" << pool5_images[0][0] << endl;
  // // cout << "First candidate, pool5_p[1] :\n" << pool5_images[0][1] << endl;
  // for (int i = 0; i < pool5_images[0].size(); i ++) {
  //   cout << pool5_images[0][i] << endl;
  // }
  
  // std::vector<std::vector<cv::Mat> > pool5_candidates;
  // WrapOutputBlob("roi_pool5_c", &pool5_candidates);
  // for (int i = 0; i < pool5_candidates[0].size(); i ++) {
  //   cout << pool5_candidates[0][i] << endl;
  // } 
  // // cout << "First candidate, pool5_c[0] :\n" << pool5_candidates[0][0] << endl;
  // // cout << "First candidate, pool5_c[1] :\n" << pool5_candidates[0][1] << endl;

  // std::vector<float> fc8;
  // GetFeatures("fc8", &fc8);
  // cout << "First candidate, fc8: " << fc8[0] << ", " << fc8[1] << endl;


  // vector<float> fc8;
  // GetFeatures("fc8", &fc8);
  // for (int i =0 ; i < candidates_flattened.size(); i++) {
  //   cout << "candidate" << i << ", fc8: " <<fc8[2*i] << ", " << fc8[2*i + 1] << endl;
  // }

  // Get softmax output
  GetProbOutput(output);
}

void Regressor::Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<float>* output) {
  assert(net_->phase() == caffe::TEST);

  // Reshape the input blobs to be the appropriate size.
  Blob<float>* input_target = net_->input_blobs()[0];
  input_target->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  Blob<float>* input_image = net_->input_blobs()[1];
  input_image->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  Blob<float>* input_bbox = net_->input_blobs()[2];
  input_bbox->Reshape(1, 4, 1, 1);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Process the inputs so we can set them.
  std::vector<cv::Mat> target_channels;
  std::vector<cv::Mat> image_channels;
  WrapInputLayer(&target_channels, &image_channels);

  // Set the inputs to the network.
  Preprocess(image, &image_channels);
  Preprocess(target, &target_channels);

  // Perform a forward-pass in the network.
  net_->ForwardPrefilled();

  // Get the network output.
  GetOutput(output);
}

void Regressor::ReshapeImageInputs(const size_t num_images) {
  // Reshape the input blobs to match the given size and geometry.
  Blob<float>* input_target = net_->input_blobs()[0];
  input_target->Reshape(num_images, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  Blob<float>* input_image = net_->input_blobs()[1];
  input_image->Reshape(num_images, num_channels_,
                       input_geometry_.height, input_geometry_.width);
}

void Regressor::ReshapeCandidateInputs(const size_t num_candidates) {
    Blob<float>* input_candidates = net_->input_blobs()[CANDIDATE_NETWORK_INPUT_IDX];
    input_candidates->Reshape(num_candidates, num_channels_,
                       input_geometry_.height, input_geometry_.width);
}

void Regressor::WrapOutputBlob(const std::string & blob_name, std::vector<cv::Mat>* output_channels) {
  const boost::shared_ptr<Blob<float> > layer = net_->blob_by_name(blob_name.c_str());
  if (blob_name.compare("pool5") == 0 || blob_name.compare("pool5_p") == 0 || blob_name.compare("pool5_c") == 0) {
    assert (layer->channels() % 256 == 0); 
  }

  int out_width = layer->width();
  int out_height = layer->height();
  float* out_data = layer->mutable_cpu_data();
  for (int i = 0; i < layer->channels(); ++i) {
    cv::Mat channel(out_height, out_width, CV_32FC1, out_data);
    output_channels->push_back(channel.clone()); // clone is needed, so that data at this call time will be pushed
    out_data += out_width * out_height;
  }
}

void Regressor::WrapOutputBlob(const std::string & blob_name, std::vector<std::vector<cv::Mat> > *output_channels) {
  const boost::shared_ptr<Blob<float> > layer = net_->blob_by_name(blob_name.c_str());

  // for (int i = 0; i < layer->num_axes(); i++) {
  //   cout <<blob_name << " axis " << i << ", dim: " << layer->shape(i) << endl;
  // }

  if (blob_name.compare("pool5") == 0 || blob_name.compare("pool5_p") == 0 || blob_name.compare("pool5_c") == 0) {
    assert (layer->channels() % 256 == 0); 
  }

  output_channels->resize(layer->shape(0)); // make sure output_channels has number of vectors that is the same as batch size

  int out_width = layer->width();
  int out_height = layer->height();
  float* out_data = layer->mutable_cpu_data();
  for (int n = 0; n < layer->shape(0); ++n) {
      for (int i = 0; i < layer->channels(); ++i) {
        cv::Mat channel(out_height, out_width, CV_32FC1, out_data);
        (*output_channels)[n].push_back(channel.clone());
        out_data += out_width * out_height;
      }
    }
}

void Regressor::WrapOutputBlob(const std::string & blob_name, std::vector<std::vector<float> > *output_features) {
  const boost::shared_ptr<Blob<float> > layer = net_->blob_by_name(blob_name.c_str());

  if (blob_name.compare("pool5") == 0 || blob_name.compare("pool5_p") == 0 || blob_name.compare("pool5_c") == 0) {
    assert (layer->channels() % 256 == 0); 
  }

  output_features->reserve(layer->shape(0));

  int out_width = layer->width();
  int out_height = layer->height();
  float* out_data = layer->mutable_cpu_data();
  for (int n = 0; n < layer->shape(0); ++n) {
      int this_count = out_width * out_height * layer->channels();
      vector<float> this_tensor_feature(out_data, out_data + this_count);
      output_features->push_back(vector<float>(this_tensor_feature));
      out_data += this_count;
    }
}

void Regressor::GetFeatures(const string& feature_name, std::vector<float>* output) const {
  //printf("Getting %s features\n", feature_name.c_str());

  // Get a pointer to the requested layer.
  const boost::shared_ptr<Blob<float> > layer = net_->blob_by_name(feature_name.c_str());

#ifdef DEBUG_FREEZE_LAYER
  string layer_name = "fc8-shapes";
  const boost::shared_ptr<Layer<float> > layer_pt = net_->layer_by_name(layer_name);
  cout << "layer_pt->param_propagate_down(0):" << layer_pt->param_propagate_down(0) << endl;
  cout << "layer_pt->param_propagate_down(1):" << layer_pt->param_propagate_down(1) << endl;
  if (layer_pt->param_propagate_down(0)) {
    layer_pt->set_param_propagate_down(0, false);
  }
  if (layer_pt->param_propagate_down(1)) {
    layer_pt->set_param_propagate_down(1, false);
  }
  const LayerParameter & layer_parameter = layer_pt->layer_param();
  cout << "layer_parameter.propagate_down_size():" << layer_parameter.propagate_down_size() << endl;
  cout << "layer_parameter.param_size():" << layer_parameter.param_size() << endl;
  // // set lr_mult and decay_mult to be 0
  // layer_parameter.param(0).set_lr_mult(0);
  // layer_parameter.param(0).set_decay_mult(0);

  // layer_parameter.param(1).set_lr_mult(0);
  // layer_parameter.param(1).set_decay_mult(0);

  cout << "layer_parameter.param(0).lr_mult():" << layer_parameter.param(0).lr_mult() << endl;
  cout << "layer_parameter.param(0).decay_mult():" << layer_parameter.param(0).decay_mult() << endl;
  // layer_parameter.mutable_param(0)->set_lr_mult(0);

  const vector< boost::shared_ptr< Blob< float > > > & parameters = net_->params();
 
  const vector< float > & params_lr_mult = net_->params_lr();

  for (int i = 0;i< params_lr_mult.size(); i++) {
    cout << "params_lr_mult["<< i << "] :" << params_lr_mult[i] << endl;
  }
#endif

  // Compute the number of elements in this layer.
  int num_elements = 1;
  for (int i = 0; i < layer->num_axes(); ++i) {
    const int elements_in_dim = layer->shape(i);
    //printf("Layer %d: %d\n", i, elements_in_dim);
    num_elements *= elements_in_dim;
  }
  //printf("Total num elements: %d\n", num_elements);

  // Copy all elements in this layer to a vector.
  const float* begin = layer->cpu_data();
  const float* end = begin + num_elements;
  *output = std::vector<float>(begin, end);
}

void Regressor::SetImages(const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets) {
  if (images.size() != targets.size()) {
    printf("Error - %zu images but %zu targets\n", images.size(), targets.size());
  }

  const size_t num_images = images.size();

  // Set network inputs to the appropriate size and number.
  ReshapeImageInputs(num_images);

  // Wrap the network inputs with opencv objects.
  std::vector<std::vector<cv::Mat> > target_channels;
  std::vector<std::vector<cv::Mat> > image_channels;
  WrapInputLayer(num_images, &target_channels, &image_channels);

  // Set the network inputs appropriately.
  Preprocess(images, &image_channels);
  Preprocess(targets, &target_channels);
}

void Regressor::SetCandidates(const std::vector<cv::Mat>& candidates) {

  const size_t num_candidates = candidates.size();

  // Set network inputs to the appropriate size and number.
  ReshapeCandidateInputs(num_candidates);

  // Wrap the network inputs with opencv objects.
  std::vector<std::vector<cv::Mat> > candidate_channels;
  WrapInputLayer(num_candidates, &candidate_channels, CANDIDATE_NETWORK_INPUT_IDX);

  // Set the network inputs appropriately.
  Preprocess(candidates, &candidate_channels);
}


void Regressor::set_rois(const std::vector<BoundingBox>& candidate_bboxes, const double scale, const int batch_id) {

  // Reshape the bbox.
  Blob<float>* input_rois = net_->input_blobs()[ROIS_NETWORK_INPUT_IDX];
  const size_t num_candidates = candidate_bboxes.size();
  vector<int> shape;
  shape.push_back(num_candidates);
  shape.push_back(5);
  input_rois->Reshape(shape);

  // Get a pointer to the bbox memory.
  float* input_rois_data = input_rois->mutable_cpu_data();

  int input_rois_data_counter = 0;
  for (size_t i = 0; i < candidate_bboxes.size(); ++i) {
    const BoundingBox& this_rois = candidate_bboxes[i];

    std::vector<float> bbox_vect;
    bbox_vect.push_back(this_rois.x1_ * scale);
    bbox_vect.push_back(this_rois.y1_ * scale);
    bbox_vect.push_back(this_rois.x2_ * scale);
    bbox_vect.push_back(this_rois.y2_ * scale);

    input_rois_data[input_rois_data_counter] = batch_id; // put the batch id as first col
    input_rois_data_counter++;
    for (size_t j = 0; j < 4; ++j) {
      input_rois_data[input_rois_data_counter] = bbox_vect[j];
      input_rois_data_counter++;
    }
  }
}

/* For MDNet, set_candidate_images
*/
void Regressor::set_candidate_images(const cv::Mat & image_curr, const std::vector<BoundingBox> & candidates_bboxes) {
  // Process the candidate, full image's input, i.e., image_curr, just one! Also record the scales
  // TODO: check if better to make square input
  int im_min_size = std::min(image_curr.size().width, image_curr.size().height);
  int im_max_size = std::max(image_curr.size().width, image_curr.size().height);

  double scale_curr = TARGET_SIZE / im_min_size;

#ifdef DEBUG_MINIMUM_SCALE_REQUIRED
  // need to make sure scale >= INPUT_BBOX_SIZE/(min(bbox_w, bbox_h))
  double scale_needed = 0.0;
  for (auto &bbox: candidates_bboxes) {
    double this_scale = INPUT_BBOX_SIZE/std::min(bbox.get_width(), bbox.get_height());
    scale_needed = std::max(scale_needed, this_scale);
  }
  scale_curr = std::max(scale_curr, scale_needed);
#endif

  // make sure don't get too large to fit in memory
  if (round(scale_curr * im_max_size) > MAX_SIZE) {
    scale_curr = MAX_SIZE / im_max_size;
  }

#ifdef DEBUG_MINIMUM_SCALE_REQUIRED
  if (scale_curr < scale_needed) {
    cout << "scale needed is >= " << scale_needed << ", but scale_curr is " << scale_curr << endl;
  }
#endif

  cv::Mat image_scaled;
  cv::resize(image_curr, image_scaled, cv::Size(), scale_curr, scale_curr);

  // Reshape Candidate input, full image's input, i.e., image_curr
  Blob<float>* input_candidate = net_->input_blobs()[CANDIDATE_NETWORK_INPUT_IDX];
  input_candidate->Reshape(1, num_channels_,
                       image_scaled.size().height, image_scaled.size().width);

  // Reshape the labels, TODO: check if this and net_->Reshape() is necessary
  Blob<float> * input_label_blob = net_->input_blobs()[LABEL_NETWORK_INPUT_IDX];
  const size_t num_labels = candidates_bboxes.size();
  // reshape to (|R|, 1)
  vector<int> shape_label;
  shape_label.push_back(num_labels);
  shape_label.push_back(1);
  input_label_blob->Reshape(shape_label);

  // Reshape the rois
  Blob<float> * input_rois_blob = net_->input_blobs()[ROIS_NETWORK_INPUT_IDX];
  const size_t num_rois = candidates_bboxes.size();
  // reshape to (|R|, 5)
  vector<int> shape_rois;
  shape_rois.push_back(num_rois);
  shape_rois.push_back(5);
  input_rois_blob->Reshape(shape_rois);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Put image_curr
  std::vector<cv::Mat> image_curr_channels;
  WrapInputLayerGivenIndex(&image_curr_channels, CANDIDATE_NETWORK_INPUT_IDX);

  // Set the inputs to the network.
  Preprocess(image_scaled, &image_curr_channels, true); // set retain the original image size

  // Put the ROIs
  set_rois(candidates_bboxes, scale_curr);
}

void Regressor::Estimate(const std::vector<cv::Mat>& images,
                        const std::vector<cv::Mat>& targets,
                        std::vector<float>* output) {
  assert(net_->phase() == caffe::TEST);

  // Set the inputs to the network.
  SetImages(images, targets);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Perform a forward-pass in the network.
  net_->ForwardPrefilled();

  // Get the network output.
  GetOutput(output);
}

void Regressor::GetOutput(std::vector<float>* output) {
  // Debugging, peep into pool5_concat layer feature size
  std::vector<float> pool5_concat_feature;
  GetFeatures("pool5_concat", & pool5_concat_feature);

  cout << "pool5_concat_feature.size():" << pool5_concat_feature.size() << endl;  

  // Get the fc8 output features of the network (this contains the estimated bounding box).
  GetFeatures("fc8", output);
}

void Regressor::GetProbOutput(std::vector<float> *output) {
#ifdef DEBUG_CONV3_FEATURE
  std::vector<std::vector<cv::Mat> > conv3_features;
  WrapOutputBlob("conv3", &conv3_features);

  // check if the 256 maps are all the same across candidates
  for (int m = 0; m < conv3_features.size(); m ++) {
    for (int n = m + 1; n < conv3_features.size(); n++) {
      bool is_different = false;
      for (int j = 0; j < conv3_features[0].size(); j++) {
        if(!equalMat(conv3_features[m][j], conv3_features[n][j])) {
          is_different = true;
        }
      }
      if (!is_different) {
        cout << "candidate " << m << " and " << n << "have the same conv3 feature"<< endl;
      }
    }
  }
#endif

  // get fc8 layer and manually compute softmax since SoftMaxWithLoss is used for finetuning
  std::vector<float> feature_fc6;
  GetFeatures("flatten_fc6", &feature_fc6);
  // batch size is feature_fc6.size()/2
  for (int i = 0;i< feature_fc6.size()/2;i++) {
    // change to softmax prob 
    double exp_0 = exp(feature_fc6[2*i]);
    double exp_1 = exp(feature_fc6[2*i + 1]);
    feature_fc6[2*i] = exp_0/(exp_0 + exp_1);
    feature_fc6[2*i + 1] = exp_1/(exp_0 + exp_1);
  }
  
  *output = feature_fc6;
}

// Wrap the input layer of the network in separate cv::Mat objects
// (one per channel). This way we save one memcpy operation and we
// don't need to rely on cudaMemcpy2D. The last preprocessing
// operation will write the separate channels directly to the input
// layer.
void Regressor::WrapInputLayer(std::vector<cv::Mat>* target_channels, std::vector<cv::Mat>* image_channels) {
  Blob<float>* input_layer_target = net_->input_blobs()[0];
  Blob<float>* input_layer_image = net_->input_blobs()[1];

  int target_width = input_layer_target->width();
  int target_height = input_layer_target->height();
  float* target_data = input_layer_target->mutable_cpu_data();
  for (int i = 0; i < input_layer_target->channels(); ++i) {
    cv::Mat channel(target_height, target_width, CV_32FC1, target_data);
    target_channels->push_back(channel);
    target_data += target_width * target_height;
  }

  int image_width = input_layer_image->width();
  int image_height = input_layer_image->height();
  float* image_data = input_layer_image->mutable_cpu_data();
  for (int i = 0; i < input_layer_image->channels(); ++i) {
    cv::Mat channel(image_height, image_width, CV_32FC1, image_data);
    image_channels->push_back(channel);
    image_data += image_width * image_height;
  }
}

// Wrap candidate, just one image
void Regressor::WrapInputLayer(std::vector<cv::Mat>* candidate_channels) {
  Blob<float>* input_layer_candidate = net_->input_blobs()[2];

  int candidate_width = input_layer_candidate->width();
  int candidate_height = input_layer_candidate->height();
  float* candidate_data = input_layer_candidate->mutable_cpu_data();
  for (int i = 0; i < input_layer_candidate->channels(); ++i) {
    cv::Mat channel(candidate_height, candidate_width, CV_32FC1, candidate_data);
    candidate_channels->push_back(channel);
    candidate_data += candidate_width * candidate_height;
  }
}

void Regressor::WrapInputLayerGivenIndex(std::vector<cv::Mat>* channels, int input_idx) {
  Blob<float>* input_layer = net_->input_blobs()[input_idx];

  int width = input_layer->width();
  int height = input_layer->height();
  float* data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, data);
    channels->push_back(channel);
    data += width * height;
  }
}

// Wrap the input layer of the network in separate cv::Mat objects
// (one per channel). This way we save one memcpy operation and we
// don't need to rely on cudaMemcpy2D. The last preprocessing
// operation will write the separate channels directly to the input
// layer.
void Regressor::WrapInputLayer(const size_t num_images,
                               std::vector<std::vector<cv::Mat> >* target_channels,
                               std::vector<std::vector<cv::Mat> >* image_channels) {
  Blob<float>* input_layer_target = net_->input_blobs()[0];
  Blob<float>* input_layer_image = net_->input_blobs()[1];

  image_channels->resize(num_images);
  target_channels->resize(num_images);

  int target_width = input_layer_target->width();
  int target_height = input_layer_target->height();
  float* target_data = input_layer_target->mutable_cpu_data();
  for (int n = 0; n < num_images; ++n) {
    for (int i = 0; i < input_layer_target->channels(); ++i) {
      cv::Mat channel(target_height, target_width, CV_32FC1, target_data);
      (*target_channels)[n].push_back(channel);
      target_data += target_width * target_height;
    }
  }

  int image_width = input_layer_image->width();
  int image_height = input_layer_image->height();
  float* image_data = input_layer_image->mutable_cpu_data();
  for (int n = 0; n < num_images; ++n) {
    for (int i = 0; i < input_layer_image->channels(); ++i) {
      cv::Mat channel(image_height, image_width, CV_32FC1, image_data);
      (*image_channels)[n].push_back(channel);
      image_data += image_width * image_height;
    }
  }
}

void Regressor::WrapInputLayer(const size_t num_candidates, 
                               std::vector<std::vector<cv::Mat> >* candidate_channels, 
                               int input_idx) {
    Blob<float>* input_layer_candidate = net_->input_blobs()[input_idx];

    candidate_channels->resize(num_candidates);

    int candidate_width = input_layer_candidate->width();
    int candidate_height = input_layer_candidate->height();
    float* candidate_data = input_layer_candidate->mutable_cpu_data();
    for (int n = 0; n < num_candidates; ++n) {
      for (int i = 0; i < input_layer_candidate->channels(); ++i) {
        cv::Mat channel(candidate_height, candidate_width, CV_32FC1, candidate_data);
        (*candidate_channels)[n].push_back(channel);
        candidate_data += candidate_width * candidate_height;
      }
    }
}

void Regressor::WrapBlobByNameBatch(const string & blob_name, std::vector<std::vector<cv::Mat> >* blob_channels) {
    const boost::shared_ptr<Blob<float> > this_blob = net_->blob_by_name(blob_name.c_str());
    
    if (blob_name.compare("pool5") == 0 || blob_name.compare("pool5_p") == 0 || blob_name.compare("pool5_c") == 0) {
      assert (this_blob->channels() % 256 == 0); 
    }
    blob_channels->resize(this_blob->shape(0));

    int this_width = this_blob->width();
    int this_height = this_blob->height();
    float* blob_data = this_blob->mutable_cpu_data();
    for (int n = 0; n < this_blob->shape(0); ++n) {
      for (int i = 0; i < this_blob->channels(); ++i) {
        cv::Mat channel(this_height, this_width, CV_32FC1, blob_data);
        (*blob_channels)[n].push_back(channel);
        blob_data += this_width * this_height;
      }
    }
}

void Regressor::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels,
                            bool keep_original_size) {
  // Convert the input image to the input image format of the network.
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  // Convert the input image to the expected size.
  cv::Mat sample_resized;
  if (!keep_original_size && sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  // Convert the input image to the expected number of channels.
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

#ifdef DEBUG_PREPROCESS_SAMPLE
  std::cout << "after conversion to CV_32FC3: " << sample_float << endl;
#endif
  // Subtract the image mean to try to make the input 0-mean.
  cv::Mat sample_normalized;
  cv::subtract(sample_float, cv::Mat(sample_float.size(), CV_32FC3, mean_scalar), sample_normalized);

#ifdef DEBUG_PREPROCESS_SAMPLE
  std::cout << "after subtract mean: " << sample_normalized << endl;
#endif
  // This operation will write the separate BGR planes directly to the
  // input layer of the network because it is wrapped by the cv::Mat
  // objects in input_channels.
  cv::split(sample_normalized, *input_channels);

  /*CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";*/
}

void Regressor::Preprocess(const std::vector<cv::Mat>& images,
                           std::vector<std::vector<cv::Mat> >* input_channels) {
  for (size_t i = 0; i < images.size(); ++i) {
    const cv::Mat& img = images[i];

    // Convert the input image to the input image format of the network.
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    // Convert the input image to the expected size.
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;

    // Convert the input image to the expected number of channels.
    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    // Subtract the image mean to try to make the input 0-mean.
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    // This operation will write the separate BGR planes directly to the
    // input layer of the network because it is wrapped by the cv::Mat
    // objects in input_channels.
    cv::split(sample_normalized, (*input_channels)[i]);

    /*CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";*/
  }
}

void Regressor::PreprocessDuplicateIn(std::vector<cv::Mat> &data_to_duplicate, std::vector<std::vector<cv::Mat> >* blob_channels) {
  for (int batch_id = 0; batch_id < blob_channels->size(); batch_id ++) {
    // copy for each channel
    for (int channel_id = 0; channel_id < data_to_duplicate.size(); channel_id ++) {
      data_to_duplicate[channel_id].copyTo((*blob_channels)[batch_id][channel_id]);
    }
  }
}
