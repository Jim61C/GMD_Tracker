#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "helper/bounding_box.h"
#include "helper/high_res_timer.h"
#include "network/regressor_base.h"
#include "helper/Constants.h"
#include "helper/helper.h"

using namespace std;

const string FREEZE_LAYER_PREFIX = "fc6_k";
// const string FREEZE_LAYER_PREFIX = "flatten_fc6_k";
// const string LOSS_LAYER_PREFIX = "loss_k";

class Regressor : public RegressorBase {
 public:
  // Set up a network with the architecture specified in deploy_proto,
  // with the model weights saved in caffe_model.
  // If we are using a model with a

  Regressor(const string& deploy_proto,
            const string& caffe_model,
            const int gpu_id,
            const int num_inputs,
            const bool do_train,
            const int K);

  Regressor(const std::string& train_deploy_proto,
            const std::string& caffe_model,
            const int gpu_id,
            const int num_inputs,
            const bool do_train);

  Regressor(const std::string& train_deploy_proto,
            const std::string& caffe_model,
            const int gpu_id,
            const bool do_train);

  // Estimate the location of the target object in the current image.
  // image_curr is the entire current image.
  // image is the best guess as to a crop of the current image that likely contains the target object.
  // target is an image of the target object from the previous frame.
  // Returns: bbox, an estimated location of the target object in the current image.
  virtual void Regress(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, BoundingBox* bbox);

  // Implement the Interface of ML Prediction out of many candidate_bboxes
  virtual void Predict(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, 
                       BoundingBox* bbox,
                       std::vector<float> * return_probabilities,
                       std::vector<int> *return_sorted_indexes);

  virtual void GetBBoxConvFeatures(const cv::Mat& image_curr,
                       const std::vector<BoundingBox> &candidate_bboxes, std::vector <std::vector<float> > &features);

  virtual void PredictFast(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, const BoundingBox & bbox_prev, 
                       BoundingBox* bbox,
                       std::vector<float> *return_probabilities, 
                       std::vector<int> *return_sorted_indexes,
                       double sd_trans,
                       int cur_frame); // TODO: remove cur_frame after debugging

protected:
  // Set the network inputs.
  void SetImages(const std::vector<cv::Mat>& images,
                 const std::vector<cv::Mat>& targets);
  
  // Set the candidates inputs at input[2]
  void SetCandidates(const std::vector<cv::Mat>& candidates);

  // Set the rois
  void set_rois(const std::vector<BoundingBox>& candidate_bboxes, const double scale, const int batch_id = 0);

  /*
  MDNet, Set the candidate image
  */
  void set_candidate_images(const cv::Mat & image_curr, const std::vector<BoundingBox> & candidates_bboxes);

  // Get the features corresponding to the output of the network.
  virtual void GetOutput(std::vector<float>* output);

  // Get the softmax layer output
  virtual void GetProbOutput(std::vector<float> *output);

  // Reshape the image inputs to the network to match the expected size and number of images.
  virtual void ReshapeImageInputs(const size_t num_images);

  virtual void ReshapeCandidateInputs(const size_t num_candidates);

  // Does all the preparations needed, i.e., forward until concat layer to finish the complete forwarding
  void PreForwardFast(const cv::Mat image_curr, 
                      const std::vector<BoundingBox> &candidate_bboxes,
                      const cv::Mat & image,
                      const cv::Mat & target);

  // TODO: current wrap WrapOutputBlob is BUGGY!!! check how to copy out memory to cv Mat
  void WrapOutputBlob(const std::string & blob_name, std::vector<cv::Mat>* output_channels);
  
  void WrapOutputBlob(const std::string & blob_name, std::vector<std::vector<cv::Mat> > *output_channels);

  void WrapOutputBlob(const std::string & blob_name, std::vector<std::vector<float> > *output_features);

  // Get the features in the network with the given name, and copy their values to the output.
  void GetFeatures(const std::string& feature_name, std::vector<float>* output) const;

  // Pass the image and the target to the network; estimate the location of the target in the current image.
  void Estimate(const cv::Mat& image, const cv::Mat& target, std::vector<float>* output);

  // Batch estimation, for tracking multiple targets.
  void Estimate(const std::vector<cv::Mat>& images,
                             const std::vector<cv::Mat>& targets,
                             std::vector<float>* output);

  // Find layer index by name, so that we could do forward from to
  int FindLayerIndexByName( const vector<string> & layer_names, const string & target);

  // Batch ML candidate binary softmax estimation
  void Estimate(std::vector<cv::Mat> &images_flattened,
                           std::vector<cv::Mat> &targets_flattened,
                           std::vector<cv::Mat> &candidates_flattened,
                           std::vector<float>* output);

  // Wrap the input layer of the network in separate cv::Mat objects
  // (one per channel).
  void WrapInputLayer(std::vector<cv::Mat>* target_channels, std::vector<cv::Mat>* image_channels);

  // Wrap candidate, just one imge
  void WrapInputLayer(std::vector<cv::Mat>* candidate_channels);

  // Wrap input according to the given input index, just one image
  void WrapInputLayerGivenIndex(std::vector<cv::Mat>* channels, int input_idx);

  // Wrap the input layer of the network in separate cv::Mat objects
  // (one per channel per image, for num_images images).
  void WrapInputLayer(const size_t num_images,
                      std::vector<std::vector<cv::Mat> >* target_channels,
                      std::vector<std::vector<cv::Mat> >* image_channels);
  
  // Wrap the input of network (candidate) in separate cv::Mat objects
  void WrapInputLayer(const size_t num_candidates, std::vector<std::vector<cv::Mat> >* candidate_channels, int input_idx);

  // Wrap a specific blob in cv::Mat objects
  void WrapBlobByNameBatch(const string & blob_name, std::vector<std::vector<cv::Mat> >* blob_channels);

  // Set the inputs to the network.
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, bool keep_original_size = false);
  void Preprocess(const std::vector<cv::Mat>& images,
                  std::vector<std::vector<cv::Mat> >* input_channels);
  
  // Create batch number of copies and Set blob value
  void PreprocessDuplicateIn(std::vector<cv::Mat> &data_to_duplicate, std::vector<std::vector<cv::Mat> >* blob_channels);

  // If the parameters of the network have been modified, reinitialize the parameters to their original values.
  virtual void Init();

  // If need to reset the net_ after tracking one video
  virtual void Reset();

  // lock the domain layers
  virtual void LockDomainLayers();

 private:
  // Set up a network with the architecture specified in deploy_proto,
  // with the model weights saved in caffe_model.
  void SetupNetwork(const std::string& deploy_proto,
                    const std::string& caffe_model,
                    const int gpu_id,
                    const bool do_train);

  // Set the mean input (used to normalize the inputs to be 0-mean).
  void SetMean();

 private:
  // Number of inputs expected by the network.
  int num_inputs_;

  // Size of the input images.
  cv::Size input_geometry_;

  // Number of image channels: normally either 1 (black and white) or 3 (color).
  int num_channels_;

  // Mean image, used to make the input 0-mean.
  cv::Mat mean_;

  // Folder containing the model parameters.
  std::string caffe_model_;

  // model definition file
  std::string deploy_proto_;

  // Whether the model weights has been modified.
  bool modified_params_;

  // Number of Domains
  int K_;

  // Timer.
  HighResTimer hrt_;
};

#endif // REGRESSOR_H
