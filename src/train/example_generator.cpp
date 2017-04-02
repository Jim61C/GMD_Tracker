#include "example_generator.h"

#include <string>

#include "helper/bounding_box.h"

#include "loader/loader_imagenet_det.h"
#include "helper/high_res_timer.h"
#include "helper/helper.h"
#include "helper/image_proc.h"
#include <assert.h>
#include <algorithm> // for shuffling
#include <random>

using std::string;

// #define DEBUG_TRAINING_SAMPLES

// Choose whether to shift boxes using the motion model or using a uniform distribution.
const bool shift_motion_model = true;

ExampleGenerator::ExampleGenerator(const double lambda_shift,
                                   const double lambda_scale,
                                   const double min_scale,
                                   const double max_scale)
  : lambda_shift_(lambda_shift),
    lambda_scale_(lambda_scale),
    min_scale_(min_scale),
    max_scale_(max_scale)
{

    gsl_rng_env_setup();
    rng_ = gsl_rng_alloc(gsl_rng_mt19937);
    // gsl_rng_set(rng_, time(NULL));
    gsl_rng_set(rng_, 800); // to reproduce
}

void ExampleGenerator::Reset(const BoundingBox& bbox_prev,
                             const BoundingBox& bbox_curr,
                             const cv::Mat& image_prev,
                             const cv::Mat& image_curr) {
  // Get padded target from previous image to feed the network.
  CropPadImage(bbox_prev, image_prev, &target_pad_);

  // Save the current image.
  image_curr_ = image_curr;

  // Save the current ground-truth bounding box.
  bbox_curr_gt_ = bbox_curr;

  // Save the previous ground-truth bounding box.
  bbox_prev_gt_ = bbox_prev;
}

void ExampleGenerator::MakeTrainingExamples(const int num_examples,
                                            std::vector<cv::Mat>* images,
                                            std::vector<cv::Mat>* targets,
                                            std::vector<BoundingBox>* bboxes_gt_scaled) {
  for (int i = 0; i < num_examples; ++i) {
    cv::Mat image_rand_focus;
    cv::Mat target_pad;
    BoundingBox bbox_gt_scaled;

    // Make training example by synthetically shifting and scaling the image,
    // creating an apparent translation and scale change of the target object.
    MakeTrainingExampleBBShift(&image_rand_focus, &target_pad, &bbox_gt_scaled);

    images->push_back(image_rand_focus);
    targets->push_back(target_pad);
    bboxes_gt_scaled->push_back(bbox_gt_scaled);
  }
}

// Randomly generates a new moved BoundingBox from bbox as candidate
BoundingBox ExampleGenerator::GenerateOneRandomCandidate(BoundingBox &bbox, gsl_rng* rng, int W, int H,
                                                         const double trans_range, const double scale_range, const string method) {
  double w = bbox.x2_ - bbox.x1_;
  double h = bbox.y2_ - bbox.y1_;
  
  double centre_x = bbox.x1_ + w/2.0;
  double centre_y = bbox.y1_ + h/2.0;
  BoundingBox moved_bbox;
  if (method.compare("uniform") == 0) {

    double dx = (gsl_rng_uniform(rng) * 2.0 - 1.0) * w * trans_range;
    double dy = (gsl_rng_uniform(rng) * 2.0 - 1.0) * h * trans_range;

    double moved_centre_x = centre_x + dx;
    double moved_centre_y = centre_y + dy;

    double ds = pow(SCALE_FACTOR, (gsl_rng_uniform(rng) * 2.0 - 1.0) * scale_range);
    double moved_w = w * ds;
    double moved_h = h * ds;

    moved_bbox.x1_ = moved_centre_x - moved_w /2.0;
    moved_bbox.y1_ = moved_centre_y - moved_h /2.0;
    moved_bbox.x2_ = moved_centre_x + moved_w/2.0;
    moved_bbox.y2_ = moved_centre_y + moved_h/2.0;
  }
  else if (method.compare("gaussian") == 0) {

    double r = round((w+h)/2.0);

    double moved_centre_x = centre_x + SD_X * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng, 1.0))); // keep the range in [-KEEP_SD* SD, KEEP_SD*SD]
    double moved_centre_y = centre_y + SD_Y * r * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng, 1.0))); 

    double ds = pow(MOTION_SCALE_FACTOR, SD_SCALE * std::max(-KEEP_SD, std::min(KEEP_SD, gsl_ran_gaussian(rng, 1.0))) );
    double moved_w = w * ds;
    double moved_h = h * ds;

    moved_bbox.x1_ = moved_centre_x - moved_w /2.0;
    moved_bbox.y1_ = moved_centre_y - moved_h /2.0;
    moved_bbox.x2_ = moved_centre_x + moved_w/2.0;
    moved_bbox.y2_ = moved_centre_y + moved_h/2.0;
  }
  else if (method.compare("whole") == 0) {
    // randomly choose from the entire frame

    double min_x = w/2;
    double min_y = h/2;
    double max_x = W - 1 - w/2; // image_w - 1 to be safe, in case of ceiling later
    double max_y = H - 1 - h/2;

    double new_centre_x = gsl_rng_uniform(rng) * (max_x - min_x) + min_x;
    double new_centre_y = gsl_rng_uniform(rng) * (max_y - min_y) + min_y;

    double ds = pow(SCALE_FACTOR, (gsl_rng_uniform(rng) * 2.0 - 1.0) * scale_range);
    double new_w = w * ds;
    double new_h = h * ds;

    moved_bbox.x1_ = new_centre_x - new_w /2.0;
    moved_bbox.y1_ = new_centre_y - new_h /2.0;
    moved_bbox.x2_ = new_centre_x + new_w /2.0;
    moved_bbox.y2_ = new_centre_y + new_h /2.0;

  }
  else {
    // exit on Unknown sampling method
    cout << "Unknown sampling method! :" << method << endl;
    exit(-1);
  }
  
  return moved_bbox;
}

void ExampleGenerator::MakeCandidatesAndLabels(vector<Mat> *candidates, vector<double> *labels, 
                                               const int num_pos,
                                               const int num_neg) {
  vector<BoundingBox> candidate_bboxes;
  MakeCandidatesAndLabelsBBox(&candidate_bboxes, labels, num_pos, num_neg);

#ifdef DEBUG_TRAINING_SAMPLES
  Mat im_show = image_curr_.clone();
  for (int i = 0; i < candidate_bboxes.size(); i++) {
    if((*labels)[i] == POS_LABEL) {
      candidate_bboxes[i].Draw(255,0,0,&im_show);
    }
    else {
      candidate_bboxes[i].Draw(0,0,255,&im_show);
    }
  }
  bbox_curr_gt_.Draw(255, 255, 255, &im_show, 2);
  imshow("random generated bboxes", im_show);
  waitKey(5);
#endif

  for (int i = 0; i < candidate_bboxes.size(); i++) {
    Mat this_candidate;
    candidate_bboxes[i].CropBoundingBoxOutImage(image_curr_, this_candidate);
    candidates->push_back(this_candidate);
  }
  assert (candidates->size() == labels->size());

  // assert the right number of positive and negative samples generated
  int count_pos = 0;
  int count_neg = 0;
  for (int i = 0; i < labels->size(); i++) {
    if ((*labels)[i] == POS_LABEL) {
      count_pos ++;
    }
    else {
      count_neg ++;
    }
  }
  assert (count_pos == num_pos);
  assert (count_neg == num_neg);

}

void ExampleGenerator::MakeCandidatesAndLabelsBBox(vector<BoundingBox> *candidate_bboxes, vector<double> *labels,
                                   const int num_pos,
                                   const int num_neg) {
  std::vector<pair<double, BoundingBox> > label_candidates;
  
  // generate positive examples
  while (label_candidates.size() < num_pos) {
    BoundingBox this_box = ExampleGenerator::GenerateOneRandomCandidate(bbox_curr_gt_, rng_, image_curr_.size().width, image_curr_.size().height);
    if (bbox_curr_gt_.compute_IOU(this_box) >= POS_IOU_TH) {
      // enqueue this bbox and label
      this_box.crop_against_image(image_curr_); // make sure within image, note here, only crop and check boundary after checking IOU as sometimes the gt bbox could be out of boundary
      if (this_box.valid_bbox_against_width_height(image_curr_.size().width, image_curr_.size().height)) { // make sure valid
        label_candidates.push_back(std::make_pair(POS_LABEL, this_box));
      }
    }
  }


  // generate negative examples
  while (label_candidates.size() < num_pos + num_neg) {
    BoundingBox this_box = ExampleGenerator::GenerateOneRandomCandidate(bbox_curr_gt_, rng_, image_curr_.size().width, image_curr_.size().height, NEG_TRANS_RANGE, NEG_SCALE_RANGE);
    if (bbox_curr_gt_.compute_IOU(this_box) <= NEG_IOU_TH) {
      // enqueue this bbox and label
      this_box.crop_against_image(image_curr_); // make sure within image
      if (this_box.valid_bbox_against_width_height(image_curr_.size().width, image_curr_.size().height)) { // make sure valid
        label_candidates.push_back(std::make_pair(NEG_LABEL, this_box));
      }
    }
  }
  
  // random shuffle
  auto engine = std::default_random_engine{};
  std::shuffle(std::begin(label_candidates), std::end(label_candidates), engine);

  for (int i = 0; i< label_candidates.size(); i++) {
    candidate_bboxes->push_back(label_candidates[i].second);
    labels->push_back(label_candidates[i].first);
  }

  assert (candidate_bboxes->size() == labels->size());
}

void ExampleGenerator::MakeTrueExample(cv::Mat* curr_search_region,
                                       cv::Mat* target_pad,
                                       BoundingBox* bbox_gt_scaled) const {
  *target_pad = target_pad_;

  // Get a tight prior prediction of the target's location in the current image.
  // For simplicity, we use the object's previous location as our guess.
  // TODO - use a motion model?
  const BoundingBox& curr_prior_tight = bbox_prev_gt_;

  // Crop the current image based on the prior estimate, with some padding
  // to define a search region within the current image.
  BoundingBox curr_search_location;
  double edge_spacing_x, edge_spacing_y;
  CropPadImage(curr_prior_tight, image_curr_, curr_search_region, &curr_search_location, &edge_spacing_x, &edge_spacing_y);

  // Recenter the ground-truth bbox relative to the search location.
  BoundingBox bbox_gt_recentered;
  bbox_curr_gt_.Recenter(curr_search_location, edge_spacing_x, edge_spacing_y, &bbox_gt_recentered);

  // Scale the bounding box relative to current crop.
  bbox_gt_recentered.Scale(*curr_search_region, bbox_gt_scaled);
}

void ExampleGenerator::get_default_bb_params(BBParams* default_params) const {
  default_params->lambda_scale = lambda_scale_;
  default_params->lambda_shift = lambda_shift_;
  default_params->min_scale = min_scale_;
  default_params->max_scale = max_scale_;
}

void ExampleGenerator::MakeTrainingExampleBBShift(cv::Mat* image_rand_focus,
                                                  cv::Mat* target_pad,
                                                  BoundingBox* bbox_gt_scaled) const {

  // Get default parameters for how much translation and scale change to apply to the
  // training example.
  BBParams default_bb_params;
  get_default_bb_params(&default_bb_params);

  // Generate training examples.
  const bool visualize_example = false;
  MakeTrainingExampleBBShift(visualize_example, default_bb_params,
                             image_rand_focus, target_pad, bbox_gt_scaled);

}

void ExampleGenerator::MakeTrainingExampleBBShift(
    const bool visualize_example, cv::Mat* image_rand_focus,
    cv::Mat* target_pad, BoundingBox* bbox_gt_scaled) const {
  // Get default parameters for how much translation and scale change to apply to the
  // training example.
  BBParams default_bb_params;
  get_default_bb_params(&default_bb_params);

  // Generate training examples.
  MakeTrainingExampleBBShift(visualize_example, default_bb_params,
                             image_rand_focus, target_pad, bbox_gt_scaled);

}

void ExampleGenerator::MakeTrainingExampleBBShift(const bool visualize_example,
                                                  const BBParams& bbparams,
                                                  cv::Mat* rand_search_region,
                                                  cv::Mat* target_pad,
                                                  BoundingBox* bbox_gt_scaled) const {
  *target_pad = target_pad_;

  // Randomly transform the current image (translation and scale changes).
  BoundingBox bbox_curr_shift;
  bbox_curr_gt_.Shift(image_curr_, bbparams.lambda_scale, bbparams.lambda_shift,
                      bbparams.min_scale, bbparams.max_scale,
                      shift_motion_model,
                      &bbox_curr_shift);

  // Crop the image based at the new location (after applying translation and scale changes).
  double edge_spacing_x, edge_spacing_y;
  BoundingBox rand_search_location;
  CropPadImage(bbox_curr_shift, image_curr_, rand_search_region, &rand_search_location,
               &edge_spacing_x, &edge_spacing_y);

  // Find the shifted ground-truth bounding box location relative to the image crop.
  BoundingBox bbox_gt_recentered;
  bbox_curr_gt_.Recenter(rand_search_location, edge_spacing_x, edge_spacing_y, &bbox_gt_recentered);

  // Scale the ground-truth bounding box relative to the random transformation.
  bbox_gt_recentered.Scale(*rand_search_region, bbox_gt_scaled);

  if (visualize_example) {
    VisualizeExample(*target_pad, *rand_search_region, *bbox_gt_scaled);
  }
}

void ExampleGenerator::VisualizeExample(const cv::Mat& target_pad,
                                        const cv::Mat& image_rand_focus,
                                        const BoundingBox& bbox_gt_scaled) const {
  const bool save_images = false;

  // Show resized target.
  cv::Mat target_resize;
  cv::resize(target_pad, target_resize, cv::Size(227,227));
  cv::namedWindow("Target object", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow("Target object", target_resize);                   // Show our image inside it.
  if (save_images) {
    const string target_name = "Image" + num2str(video_index_) + "_" + num2str(frame_index_) + "target.jpg";
    cv::imwrite(target_name, target_resize);
  }

  // Resize the image.
  cv::Mat image_resize;
  cv::resize(image_rand_focus, image_resize, cv::Size(227, 227));

  // Draw gt bbox.
  BoundingBox bbox_gt_unscaled;
  bbox_gt_scaled.Unscale(image_resize, &bbox_gt_unscaled);
  bbox_gt_unscaled.Draw(0, 255, 0, &image_resize);

  // Show image with bbox.
  cv::namedWindow("Search_region+gt", cv::WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow("Search_region+gt", image_resize );                   // Show our image inside it.
  if (save_images) {
    const string image_name = "Image" + num2str(video_index_) + "_" + num2str(frame_index_) + "image.jpg";
    cv::imwrite(image_name, image_resize);
  }

  cv::waitKey(0);
}
