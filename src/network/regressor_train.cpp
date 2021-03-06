#include "regressor_train.h"
#include <iostream>
#include <fstream>
#include <unordered_set>

const int kNumInputs = 4;
const bool kDoTrain = true;
const int LOSS_SAVE_ITER = 500;

using std::string;
using std::vector;
using caffe::Blob;
using caffe::Layer;
using caffe::LayerParameter;

// #define DEBUG_ROI_POOL_INPUT

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const int num_input,
                               const bool do_train)
  : Regressor(deploy_proto, caffe_model, gpu_id, num_input, do_train),
    RegressorTrainBase(solver_file),
    loss_save_path_("")
{
  solver_.set_net(net_);
}


RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const bool do_train)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, do_train),
    RegressorTrainBase(solver_file),
    loss_save_path_("")
{
  solver_.set_net(net_);
}

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, kDoTrain),
    RegressorTrainBase(solver_file),
    loss_save_path_("")
{
  solver_.set_net(net_);
}

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const string& loss_save_path,
                               const int K)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, kDoTrain, K),
    RegressorTrainBase(solver_file),
    loss_save_path_(loss_save_path)
{
  solver_.set_net(net_);
}

void RegressorTrain::ResetSolverNet() {
  // TODO: if reload a new solver and assign to solver_ makes memory go down, but need to change solver_ to be a shared_ptr

  // free the memory pointed in solver_.net_ 
  solver_.reset_net();
  solver_.set_net(net_);

}

void RegressorTrain::set_test_net(const std::string& test_proto) {
  printf("Setting test net to: %s\n", test_proto.c_str());
  test_net_.reset(new caffe::Net<float>(test_proto, caffe::TEST));
  solver_.set_test_net(test_net_);
}

void RegressorTrain::set_bboxes_gt(const std::vector<BoundingBox>& bboxes_gt) {
  assert(net_->phase() == caffe::TRAIN);

  // Reshape the bbox.
  Blob<float>* input_bbox = net_->input_blobs()[2];
  const size_t num_images = bboxes_gt.size();
  const int bbox_dims = 4;
  std::vector<int> shape;
  shape.push_back(num_images);
  shape.push_back(bbox_dims);
  input_bbox->Reshape(shape);

  // Get a pointer to the bbox memory.
  float* input_bbox_data = input_bbox->mutable_cpu_data();

  int input_bbox_data_counter = 0;
  for (size_t i = 0; i < bboxes_gt.size(); ++i) {
    const BoundingBox& bbox_gt = bboxes_gt[i];

    // Set the bbox data to the ground-truth bbox.
    std::vector<float> bbox_vect;
    bbox_gt.GetVector(&bbox_vect);
    for (size_t j = 0; j < 4; ++j) {
      input_bbox_data[input_bbox_data_counter] = bbox_vect[j];
      input_bbox_data_counter++;
    }
  }
}

void RegressorTrain::set_labels(const std::vector<double>  &labels_flattened) {
  assert(net_->phase() == caffe::TRAIN);

  Blob<float> * input_label_blob = net_->input_blobs()[LABEL_NETWORK_INPUT_IDX];
  const size_t num_labels = labels_flattened.size();

  // reshape to (|R|, 1)
  std::vector<int> shape;
  shape.push_back(num_labels);
  shape.push_back(1);
  input_label_blob->Reshape(shape);
  
  // get a pointer to the label input blob memory.
  float* input_label_data = input_label_blob->mutable_cpu_data();

  // set data
  for (int i = 0; i < num_labels; i++) {
    input_label_data[i] = labels_flattened[i];
  }
}

void RegressorTrain::Train(const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt) {
  assert(net_->phase() == caffe::TRAIN);

  if (images.size() != targets.size()) {
    printf("Error - %zu images but %zu targets\n", images.size(), targets.size());
  }

  if (images.size() != bboxes_gt.size()) {
    printf("Error - %zu images but %zu bboxes_gt", images.size(), bboxes_gt.size());
  }

  // Normally to track we just estimate the bbox location; if we need to backprop,
  // we also need to input the ground-truth bounding boxes.
  set_bboxes_gt(bboxes_gt);

  // Set the image and target.
  SetImages(images, targets);

  // Train the network.
  Step();
}


void RegressorTrain::TrainBatchFast(const std::vector<cv::Mat>& image_currs,
                           const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt,
                           const std::vector<std::vector<BoundingBox> > &candidate_bboxes,
                           const std::vector<std::vector<double> > &labels,
                           int k,
                           int inner_batch_size,
                           int num_nohem) {
    // make sure same number of inputs, images.size() is kBatchSize
    assert (images.size() == image_currs.size());
    assert (images.size() == targets.size());
    assert (images.size() == candidate_bboxes.size());
    assert (images.size() == labels.size());

    for (int i = 0; i< candidate_bboxes.size(); i++) {
      assert (candidate_bboxes[i].size() == labels[i].size());
      const std::vector<BoundingBox> &this_image_candidates = candidate_bboxes[i];
      const std::vector<double> &this_image_labels = labels[i];
      const cv::Mat &this_image = images[i];
      const cv::Mat &this_target = targets[i];
      const cv::Mat &this_image_curr = image_currs[i];

      int total_size = this_image_candidates.size();
      int num_inner_batches =  total_size / inner_batch_size;
      // flatten and get corresponding image/target index
      for (int j = 0; j < num_inner_batches; j ++) {
        std::vector<BoundingBox> this_candidates_flattened(this_image_candidates.begin() + j*inner_batch_size, this_image_candidates.begin() + std::min((j+1)*inner_batch_size, total_size));
        std::vector<double> this_labels_flattened(this_image_labels.begin() + j*inner_batch_size, this_image_labels.begin() + std::min((j+1)*inner_batch_size, total_size));

        TrainForwardBackward(this_image_curr,
                          this_candidates_flattened,
                          this_labels_flattened,
                          this_image,
                          this_target,
                          k, 
                          num_nohem);
        }
    }
}

void RegressorTrain::TrainForwardBackwardWorker(const cv::Mat & image_curr,
                          const std::vector<BoundingBox> &candidates_bboxes, 
                          const std::vector<double> &labels,
                          const cv::Mat & image,
                          const cv::Mat & target,
                          int k, 
                          int num_nohem) {

  // Here: only batch size 1 is implemented, TODO: incorporate batch size > 1, i.e., have a vector<cv::Mat> image_currs coming in
  
  assert(candidates_bboxes.size() == labels.size());
  // actual worker to forward and backward for this pair of image and target with the given candidates
  
  net_->ClearParamDiffs(); // clear the previous param diff

  // forward until concat and prepare duplicated images and targets for keep forwarding
  PreForwardFast(image_curr, candidates_bboxes, image, target);

  // now put the labels ready
  set_labels(labels);

#ifdef DEBUG_ROI_POOL_INPUT
  std::vector<cv::Mat> image_curr_scaled_splitted;
  WrapOutputBlob("candidate", &image_curr_scaled_splitted);
  cv::Mat image_curr_scale;
  cv::merge(image_curr_scaled_splitted, image_curr_scale);

  std::vector<cv::Mat> target_splitted;
  WrapOutputBlob("target", &target_splitted);
  cv:: Mat target_merged;
  cv::merge(target_splitted, target_merged);
  cv::add(target_merged, cv::Mat(target_merged.size(), CV_32FC3, mean_scalar), target_merged);
  target_merged.convertTo(target_merged, CV_8UC3);
  imshow("t-1 target", target_merged);

  // cout << "image_curr_scale.channels(): " << image_curr_scale.channels() << endl;
  // cout << "image_curr_scale.size().width: " << image_curr_scale.size().width << endl;
  // cout << "image_curr_scale.size().height: " << image_curr_scale.size().height << endl;
  // // cout << image_curr_scale << endl;
 
  cv::Mat image_curr_scale_origin;
  cv::add(image_curr_scale, cv::Mat(image_curr_scale.size(), CV_32FC3, mean_scalar), image_curr_scale_origin);
  image_curr_scale_origin.convertTo(image_curr_scale_origin, CV_8UC3);

  std::vector<float> labels_in;
  GetFeatures("label", &labels_in);

  std::vector<float> rois_in;
  GetFeatures("rois", &rois_in);

  std::vector <BoundingBox> bboxes_in;
  for (int i = 0; i < rois_in.size(); i+= 5) {
    // each rois in the rois_in memory is [batch_id, x1, y1, x2, y2]
    BoundingBox this_bbox(rois_in[i + 1],
                          rois_in[i + 2],
                          rois_in[i + 3],
                          rois_in[i + 4]);
    bboxes_in.push_back(this_bbox);
  }

  assert (labels_in.size() == bboxes_in.size());
  for (int i = 0; i < labels_in.size(); i++) {
    if(labels_in[i] == 1) {
      bboxes_in[i].Draw(255, 0, 0, &image_curr_scale_origin);
    }
    else {
      bboxes_in[i].Draw(0, 0, 255, &image_curr_scale_origin);
    }
  }

  cv::imshow("rois on scaled image:", image_curr_scale_origin);
  cv::waitKey(0);
#endif
  
  // // TODO: check if just put in will be faster as the following instead of calling set_labels
  // Blob<float> * input_label_blob = net_->input_blobs()[3];
  // const size_t num_labels = labels.size();
  // float* input_label_data = input_label_blob->mutable_cpu_data();
  // for (int i = 0; i < num_labels; i++) {
  //   input_label_data[i] = labels[i];
  // }

  const std::vector<string> & layer_names = net_->layer_names();
  int layer_pool5_concat_idx = FindLayerIndexByName(layer_names, "concat");
  int layer_loss_idx = FindLayerIndexByName(layer_names, "loss");
  int layer_fc8_idx = FindLayerIndexByName(layer_names, "fc8");
  
  if (num_nohem != -1) {
    net_->ForwardFrom(layer_pool5_concat_idx);
    
    // record probs
    std::vector<float> probs;
    GetProbOutput(&probs);
    std::vector<float> positive_probs;
    for (int i = 0; i < candidates_bboxes.size(); i ++) {
      positive_probs.push_back(probs[2*i + 1]);
    }
    net_->BackwardFromTo(layer_loss_idx, layer_loss_idx);

    // conduct online hard example mining
    string fc8_blob_name = "fc8";
    const boost::shared_ptr<Blob<float> > blob_to_set_diff = net_->blob_by_name(fc8_blob_name.c_str());
    float * diff_begin = blob_to_set_diff->mutable_cpu_diff();
    float * diff_end = diff_begin + blob_to_set_diff->count();
    std::vector<float> loss_diff_val(diff_begin, diff_end);

    std::vector<int> neg_bag;
    std::vector<float> neg_probs;
    unordered_set<int> backprop_idxes;
    for (int i = 0; i < candidates_bboxes.size(); i ++) {
      if (labels[i] == POS_LABEL) {
        // back prop all the positive ones
        backprop_idxes.insert(i);
      }
      else {
        neg_bag.push_back(i);
        neg_probs.push_back(positive_probs[i]);
      }
    }

    std::vector<int> idx(neg_probs.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in neg_prob, from large to small
    sort(idx.begin(), idx.end(),
        [&neg_probs](int i1, int i2) {return neg_probs[i1] > neg_probs[i2];});
    
    for (int i = 0; i < num_nohem; i ++) {
      backprop_idxes.insert(neg_bag[idx[i]]);
    }
    
    // set the mutable diff blob data
    for (int i = 0; i < candidates_bboxes.size(); i ++) {
      // back prop only for hard examples
      if (backprop_idxes.find(i) == backprop_idxes.end()) {
        // set the gradients to be zero for non hard examples
        diff_begin[2*i] = 0;
        diff_begin[2*i + 1] = 0;
      }
    }
    net_->BackwardFromTo(layer_fc8_idx, layer_pool5_concat_idx);
  }
  else {
    net_->ForwardFrom(layer_pool5_concat_idx);
    net_->BackwardTo(layer_pool5_concat_idx);
  }

  
  // update weights
  // no need: UpdateSmoothedLoss(loss, start_iter, average_loss); as here only 1 iter
  solver_.apply_update();
  solver_.increment_iter_save_snapshot();
}

void RegressorTrain::TrainForwardBackward( const cv::Mat & image_curr,
                          const std::vector<BoundingBox> &candidates_bboxes, 
                          const std::vector<double> &labels_flattened,
                          const cv::Mat & image,
                          const cv::Mat & target,
                          int k,
                          int num_nohem) {
    assert(candidates_bboxes.size() == labels_flattened.size());
    
    if (k != -1) {
      // Usual Training, need to freeze layers
      string this_layer_name = FREEZE_LAYER_PREFIX + std::to_string(k);

      // unlock this layer
      const boost::shared_ptr<Layer<float> > layer_pt = net_->layer_by_name(this_layer_name);

      if (!layer_pt->param_propagate_down(0)) {
        layer_pt->set_param_propagate_down(0, true);
      }
      if (!layer_pt->param_propagate_down(1)) {
        layer_pt->set_param_propagate_down(1, true);
      }
      
      TrainForwardBackwardWorker(image_curr, candidates_bboxes, labels_flattened, image, target, k, num_nohem);

      //lock this layer back
      // cout << this_layer_name << " freeze back" << endl;
      if (layer_pt->param_propagate_down(0)) {
        layer_pt->set_param_propagate_down(0, false);
      }
      if (layer_pt->param_propagate_down(1)) {
        layer_pt->set_param_propagate_down(1, false);
      }
    }
    else {
      // // make sure that weights are actually updated
      // const std::vector<boost::shared_ptr<Blob<float> > > & net_params = net_->params();
      // cout << "net_params.size():" << net_params.size() << endl;
      // std::vector<float> fc6_gmd_weights_before;
      // Blob<float> *ptr = net_params[36].get();
      // const float* begin = ptr->cpu_data();
      // const float* end = begin + ptr->count();
      // fc6_gmd_weights_before = std::vector<float>(begin, end);
      
      TrainForwardBackwardWorker(image_curr, candidates_bboxes, labels_flattened, image, target, k, num_nohem);

      if (loss_save_path_.length() != 0) {
        std::vector<float> this_loss_output;
        GetFeatures("loss", &this_loss_output);
        loss_history_.push_back(this_loss_output[0]);
      }

      InvokeSaveLossIfNeeded();
    }
}

void RegressorTrain::TrainBatch(const std::vector<cv::Mat>& images,
                           const std::vector<cv::Mat>& targets,
                           const std::vector<BoundingBox>& bboxes_gt,
                           const std::vector<std::vector<cv::Mat> > &candidates,
                           const std::vector<std::vector<double> > &labels,
                           int k) {
    // make sure same number of inputs, images.size() is kBatchSize
    assert (images.size() == targets.size());
    assert (images.size() == candidates.size());
    assert (images.size() == labels.size());

    // tile (flatten) the images and targets
    std::vector<cv::Mat> images_flattened;
    std::vector<cv::Mat> targets_flattened;
    std::vector<cv::Mat> candidates_flattened;
    std::vector<double> labels_flattened;
    
    for (int i = 0;i<images.size();i++) {
      assert (candidates[i].size() == labels[i].size());
      int num_candidates = candidates[i].size();
      cv::Mat this_image = images[i];
      cv::Mat this_target = targets[i];

      for (int j = 0;j< num_candidates;j++) {
        candidates_flattened.push_back(candidates[i][j]);
        labels_flattened.push_back(labels[i][j]);
        
        // rep images and targets
        images_flattened.push_back(this_image.clone());
        targets_flattened.push_back(this_target.clone());
      }
    }

    //TODO: if random shuffling here helps

    // images_flattened.size() should be 11 * 250, if kBatchSize outside is 11 and POS/NEG sample sizes are 250
    int total_size = images_flattened.size();
    int num_inner_batches =  total_size / INNER_BATCH_SIZE;
    for (int i = 0; i< num_inner_batches; i++) {
      std::vector<cv::Mat> this_images_flattened(images_flattened.begin() + i*INNER_BATCH_SIZE, images_flattened.begin() + std::min((i+1)*INNER_BATCH_SIZE, total_size));
      std::vector<cv::Mat> this_targets_flattened(targets_flattened.begin() + i*INNER_BATCH_SIZE, targets_flattened.begin() + std::min((i+1)*INNER_BATCH_SIZE, total_size));
      std::vector<cv::Mat> this_candidates_flattened(candidates_flattened.begin() + i*INNER_BATCH_SIZE, candidates_flattened.begin() + std::min((i+1)*INNER_BATCH_SIZE, total_size));
      std::vector<double> this_labels_flattened(labels_flattened.begin() + i*INNER_BATCH_SIZE, labels_flattened.begin() + std::min((i+1)*INNER_BATCH_SIZE, total_size));

      Train(this_images_flattened, this_targets_flattened, this_candidates_flattened, this_labels_flattened, k);
    }

    // Train(images_flattened, targets_flattened, candidates_flattened, labels_flattened, k);
}

// TODO: change this API to be pointer based with a number parameter, so that we can avoid the copying performed above!!!
void RegressorTrain::Train(std::vector<cv::Mat> &images_flattened,
                           std::vector<cv::Mat> &targets_flattened,
                           std::vector<cv::Mat> &candidates_flattened,
                           std::vector<double> &labels_flattened,
                           int k) {
    assert(images_flattened.size() == targets_flattened.size());
    assert(images_flattened.size() == candidates_flattened.size());
    assert(images_flattened.size() == labels_flattened.size());
    
  //   for (int i = 0; i< net_->output_blob_indices().size(); i ++) {
  //   std::string temp_layer_name = FREEZE_LAYER_PREFIX + std::to_string(i);
  //   // unlock this layer
  //   const boost::shared_ptr<Layer<float> > layer_pt = net_->layer_by_name(temp_layer_name);

  //   cout << temp_layer_name << ", layer_pt->param_propagate_down(0):" << layer_pt->param_propagate_down(0) << endl;
  //   cout << temp_layer_name << ", layer_pt->param_propagate_down(1):" << layer_pt->param_propagate_down(1) << endl;

  //   const LayerParameter & layer_parameter = layer_pt->layer_param();
  //   cout << temp_layer_name << ", layer_parameter.propagate_down_size():" << layer_parameter.propagate_down_size() << endl;
  //   cout << temp_layer_name << ", layer_parameter.param_size():" << layer_parameter.param_size() << endl;
  // }

    if (k != -1) {
      // Usual Training, need to freeze layers
      string this_layer_name = FREEZE_LAYER_PREFIX + std::to_string(k);

      // unlock this layer
      const boost::shared_ptr<Layer<float> > layer_pt = net_->layer_by_name(this_layer_name);

      if (!layer_pt->param_propagate_down(0)) {
        layer_pt->set_param_propagate_down(0, true);
      }
      if (!layer_pt->param_propagate_down(1)) {
        layer_pt->set_param_propagate_down(1, true);
      }

      // cout << "after turning on:" << endl;
      // cout << this_layer_name << ", layer_pt->param_propagate_down(0):" <<layer_pt->param_propagate_down(0) << endl;
      // cout << this_layer_name << ", layer_pt->param_propagate_down(1):" <<layer_pt->param_propagate_down(1) << endl;
      

      // Set the image and target.
      SetImages(images_flattened, targets_flattened);

      // Set the candidates
      SetCandidates(candidates_flattened);

      // Set the labels
      set_labels(labels_flattened);

      // Train the network.
      Step();
        
      //lock this layer back
      // cout << this_layer_name << " freeze back" << endl;
      if (layer_pt->param_propagate_down(0)) {
        layer_pt->set_param_propagate_down(0, false);
      }
      if (layer_pt->param_propagate_down(1)) {
        layer_pt->set_param_propagate_down(1, false);
      }
    }
    else {
      // Fine Tuning, only one domain, just normally step
      
      // Set the image and target.
      SetImages(images_flattened, targets_flattened);

      // Set the candidates
      SetCandidates(candidates_flattened);

      // Set the labels
      set_labels(labels_flattened);

      // Train the network.
      Step();


      if (loss_save_path_.length() != 0) {
        std::vector<float> this_loss_output;
        GetFeatures("loss", &this_loss_output);
        loss_history_.push_back(this_loss_output[0]);
      }

      InvokeSaveLossIfNeeded();
    }
    
}

void RegressorTrain::Step() {
  assert(net_->phase() == caffe::TRAIN);

  // Train the network.
  solver_.Step(1);
}


void RegressorTrain::SaveLossHistoryToFile(const std::string &save_path) {
  ofstream out_loss_file;
  out_loss_file.open(save_path.c_str(), std::ios_base::app);
  for (int i = 0;i< loss_history_.size();i++) {
      out_loss_file << loss_history_[i] << " ";
  }
  out_loss_file.close();
  loss_history_.clear(); // clear after write
}

void RegressorTrain::InvokeSaveLossIfNeeded() {
  if (loss_history_.size() > LOSS_SAVE_ITER) {
    SaveLossHistoryToFile(loss_save_path_);
  }
}