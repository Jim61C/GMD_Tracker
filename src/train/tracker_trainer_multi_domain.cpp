#include "tracker_trainer_multi_domain.h"

#include "caffe/caffe.hpp"

#include "network/regressor.h"

// Number of images in each batch, favourately multiplication of 11
const int kBatchSize = 8;

// Number of examples that we generate (by applying synthetic transformations)
// to each image.
const int kGeneratedExamplesPerImage = 0;

TrackerTrainerMultiDomain::TrackerTrainerMultiDomain(ExampleGenerator* example_generator)
  : example_generator_(example_generator),
    num_batches_(0),
    current_k_(-1),
    batch_filled_(false)
{
}

TrackerTrainerMultiDomain::TrackerTrainerMultiDomain(ExampleGenerator* example_generator,
                               RegressorTrainBase* regressor_train)
  : example_generator_(example_generator),
    regressor_train_(regressor_train),
    num_batches_(0),
    current_k_(-1),
    batch_filled_(false)
{
}

void TrackerTrainerMultiDomain::MakeTrainingExamples(std::vector<cv::Mat>* images,
                                          std::vector<cv::Mat>* targets,
                                          std::vector<BoundingBox>* bboxes_gt_scaled,
                                          std::vector<std::vector<BoundingBox> > *candidates, 
                                          std::vector<std::vector<double> >  *labels) {
  // use the previously Reset()'s frame to generate examples and store output in the given containers

  // Generate true example.
  cv::Mat image;
  cv::Mat target;
  BoundingBox bbox_gt_scaled;
  example_generator_->MakeTrueExampleTight(&image, &target, &bbox_gt_scaled);
  images->push_back(image);
  targets->push_back(target);
  bboxes_gt_scaled->push_back(bbox_gt_scaled);
                                           
  std::vector<BoundingBox> this_frame_candidates;
  std::vector<double> this_frame_labels;

  // generate examples and push to this_frame_candidates and this_frame_labels
  example_generator_->MakeCandidatesAndLabelsBBox(&this_frame_candidates, &this_frame_labels);
  cout << "after MakeCandidatesAndLabelsBBox, k == " << current_k_ << endl;
  // TODO: avoid the copying and just pass a vector of one frame's +/- candidates to train
  for(int i = 0; i< images->size(); i ++ ) {
    candidates->push_back(std::vector<BoundingBox>(this_frame_candidates)); // copy
    labels->push_back(std::vector<double>(this_frame_labels)); // copy
  }
}

void TrackerTrainerMultiDomain::ProcessBatch() {
  // cout << "about to invoke actual traning with k: " << current_k_ << endl;
  //// Train the neural network tracker with these examples.
  regressor_train_->TrainBatchFast(image_currs_batch_, 
                               images_batch_,
                               targets_batch_,
                               bboxes_gt_scaled_batch_,
                               candidates_batch_,
                               labels_batch_,
                               current_k_);
}

void TrackerTrainerMultiDomain::SaveLossHistoryToFile(const std::string &save_path) {
  regressor_train_->SaveLossHistoryToFile(save_path);
}

void TrackerTrainerMultiDomain::Train(const cv::Mat& image_prev, const cv::Mat& image_curr,
                           const BoundingBox& bbox_prev, const BoundingBox& bbox_curr) {
  // Check that the saved batches are of appropriate dimensions.
  CHECK_EQ(images_batch_.size(), targets_batch_.size())
      << " images_batch_: " << images_batch_.size() <<
         " targets_batch_: " << targets_batch_.size();

  CHECK_EQ(images_batch_.size(), bboxes_gt_scaled_batch_.size())
      << " images_batch: " << images_batch_.size() <<
         " bboxes_gt_scaled_batch_: " << bboxes_gt_scaled_batch_.size();
  
  CHECK_EQ(images_batch_.size(), image_currs_batch_.size())
    << " images_batch: " << images_batch_.size() <<
        " image_currs_batch_: " << image_currs_batch_.size();

  CHECK_EQ(images_batch_.size(), candidates_batch_.size())
    << " images_batch: " << images_batch_.size() <<
        " candidates_batch_: " << candidates_batch_.size();
  
  CHECK_EQ(images_batch_.size(), labels_batch_.size())
    << " images_batch: " << images_batch_.size() <<
        " labels_batch_: " << labels_batch_.size();
  

  // Set up example generator.
  example_generator_->Reset(bbox_prev,
                           bbox_curr,
                           image_prev,
                           image_curr);

  // Make training examples.
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> targets;
  std::vector<BoundingBox> bboxes_gt_scaled;

  // Make +/- candidates and labels
  std::vector<std::vector<BoundingBox> > candidates; // sampled candiates using IOU
  std::vector<std::vector<double> >  labels; // +/-'s

  // Here, only one will be enqueued
  MakeTrainingExamples(&images, &targets, &bboxes_gt_scaled, &candidates, &labels);

  std::vector<cv::Mat> image_currs;
  image_currs.push_back(image_curr);
  for (int augment_id = 0; augment_id < kGeneratedExamplesPerImage; augment_id ++) {
    image_currs.push_back(image_curr);
  }

  while (images.size() > 0) {
    // Compute the number of images left to complete the batch.
    const int num_in_batch = images_batch_.size();
    const int num_left_in_batch = kBatchSize - num_in_batch;

    // The number of images to use is upper-bounded by the number left in the batch.
    // The rest go into the next batch.
    const int num_use = std::min(static_cast<int>(images.size()), num_left_in_batch);

    if (num_use < 0) {
      printf("Error: num_use: %d\n", num_use);
    }

    // Add the approrpriate number of images to the batch.
    image_currs_batch_.insert(image_currs_batch_.end(), 
                              image_currs.begin(), image_currs.begin() + num_use);
    images_batch_.insert(images_batch_.end(),
                         images.begin(), images.begin() + num_use);
    targets_batch_.insert(targets_batch_.end(),
                          targets.begin(), targets.begin() + num_use);
    bboxes_gt_scaled_batch_.insert(bboxes_gt_scaled_batch_.end(),
                                   bboxes_gt_scaled.begin(),
                                   bboxes_gt_scaled.begin() + num_use);
    candidates_batch_.insert(candidates_batch_.end(),
                             candidates.begin(),
                             candidates.begin() + num_use);
    labels_batch_.insert(labels_batch_.end(), 
                         labels.begin(), labels.begin() + num_use);

    // If we have a full batch, then train!  Otherwise, save this batch for later.
    if (images_batch_.size() == kBatchSize) {
      // Increment the batch count.
      num_batches_++;

      batch_filled_ = true;

      // We have filled up a complete batch, so we should train.
      ProcessBatch();

      // After training, clear the batch.
      image_currs_batch_.clear();
      images_batch_.clear();
      targets_batch_.clear();
      bboxes_gt_scaled_batch_.clear();
      candidates_batch_.clear();
      labels_batch_.clear();

      // Reserve the appropriate amount of space for the next batch.
      image_currs_batch_.reserve(kBatchSize);
      images_batch_.reserve(kBatchSize);
      targets_batch_.reserve(kBatchSize);
      bboxes_gt_scaled_batch_.reserve(kBatchSize);
      candidates_batch_.reserve(kBatchSize);
      labels_batch_.reserve(kBatchSize);

    }

    // Remove the images that were used.
    image_currs.erase(image_currs.begin(), image_currs.begin() + num_use);
    images.erase(images.begin(), images.begin() + num_use);
    targets.erase(targets.begin(), targets.begin() + num_use);
    bboxes_gt_scaled.erase(bboxes_gt_scaled.begin(), bboxes_gt_scaled.begin() + num_use);
    candidates.erase(candidates.begin(), candidates.begin() + num_use);
    labels.erase(labels.begin(), labels.begin() + num_use);
  }
}

// get if full batch
bool TrackerTrainerMultiDomain::get_if_full_batch() {
  return batch_filled_;
}

// set batch_filled
void TrackerTrainerMultiDomain::set_batch_filled(bool val) {
  batch_filled_ = val;
}

// clear remaining data in the batch
void TrackerTrainerMultiDomain::clear_batch_remaining() {
    // After training, clear the batch.
    image_currs_batch_.clear();
    images_batch_.clear();
    targets_batch_.clear();
    bboxes_gt_scaled_batch_.clear();
    candidates_batch_.clear();
    labels_batch_.clear();

    // Reserve the appropriate amount of space for the next batch.
    image_currs_batch_.reserve(kBatchSize);
    images_batch_.reserve(kBatchSize);
    targets_batch_.reserve(kBatchSize);
    bboxes_gt_scaled_batch_.reserve(kBatchSize);
    candidates_batch_.reserve(kBatchSize);
    labels_batch_.reserve(kBatchSize);
}