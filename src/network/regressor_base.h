#ifndef REGRESSOR_BASE_H
#define REGRESSOR_BASE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include <caffe/caffe.hpp>

class BoundingBox;

// A neural network for the tracker must inherit from this class.
class RegressorBase
{
public:
  RegressorBase();

  // Predict the bounding box.
  // image_curr is the entire current image.
  // image is the best guess as to a crop of the current image that likely contains the target object.
  // target is an image of the target object from the previous frame.
  // Returns: bbox, an estimated location of the target object in the current image.
  virtual void Regress(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, BoundingBox* bbox) = 0;

  // Predict current target as ML estimate, out of given 250 moved boxes
  virtual void Predict(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, 
                       BoundingBox* bbox,
                       std::vector<float> *return_probabilities, 
                       std::vector<int> *return_sorted_indexes) = 0;

  virtual void GetBBoxConvFeatures(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes, std::vector <std::vector<float> > features) = 0;
  
  // Predict faster, with ROI pooling
  virtual void PredictFast(const cv::Mat& image_curr, const cv::Mat& image, const cv::Mat& target, 
                       const std::vector<BoundingBox> &candidate_bboxes,  const BoundingBox & bbox_prev,
                       BoundingBox* bbox,
                       std::vector<float> *return_probabilities, 
                       std::vector<int> *return_sorted_indexes,
                       double sd_trans,
                       int cur_frame) = 0;

  // Called at the beginning of tracking a new object to initialize the network.
  virtual void Init() { }

  // Called at the end of tracking a video, reset
  virtual void Reset() { }

  //virtual boost::shared_ptr<caffe::Net<float> > get_net() { return net_; }

protected:
  boost::shared_ptr<caffe::Net<float> > net_;

};

#endif // REGRESSOR_BASE_H
