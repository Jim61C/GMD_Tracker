#include "regressor_train.h"
#include <iostream>
#include <fstream>

const int kNumInputs = 4;
const bool kDoTrain = true;
const int INNER_BATCH_SIZE = 50;

using std::string;
using std::vector;
using caffe::Blob;
using caffe::Layer;
using caffe::LayerParameter;

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const int num_input,
                               const bool do_train)
  : Regressor(deploy_proto, caffe_model, gpu_id, num_input, do_train),
    RegressorTrainBase(solver_file)
{
  solver_.set_net(net_);
}


RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const bool do_train)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, do_train),
    RegressorTrainBase(solver_file)
{
  solver_.set_net(net_);
}

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, kDoTrain),
    RegressorTrainBase(solver_file)
{
  solver_.set_net(net_);
}

RegressorTrain::RegressorTrain(const std::string& deploy_proto,
                               const std::string& caffe_model,
                               const int gpu_id,
                               const string& solver_file,
                               const int K)
  : Regressor(deploy_proto, caffe_model, gpu_id, kNumInputs, kDoTrain, K),
    RegressorTrainBase(solver_file)
{
  solver_.set_net(net_);

  // set up the loss_history_k_domain_
  for (int i = 0; i < K;i++) {
    loss_history_k_domain_.push_back(std::vector<double>());
  }
}

void RegressorTrain::ResetSolverNet() {
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
  vector<int> shape;
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

void RegressorTrain::set_labels(vector<double>  &labels_flattened) {
  assert(net_->phase() == caffe::TRAIN);

  // get the input blob for labels, reshape to include batch number
  Blob<float> * input_label_blob = net_->input_blobs()[3];
  const size_t num_labels = labels_flattened.size();

  // reshape to batch size
  vector<int> shape;
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
    int num_inner_batches = images_flattened.size() / INNER_BATCH_SIZE;
    for (int i = 0; i< num_inner_batches; i++) {
      std::vector<cv::Mat> this_images_flattened(images_flattened.begin() + i*INNER_BATCH_SIZE, images_flattened.begin() + (i+1)*INNER_BATCH_SIZE);
      std::vector<cv::Mat> this_targets_flattened(targets_flattened.begin() + i*INNER_BATCH_SIZE, targets_flattened.begin() + (i+1)*INNER_BATCH_SIZE);
      std::vector<cv::Mat> this_candidates_flattened(candidates_flattened.begin() + i*INNER_BATCH_SIZE, candidates_flattened.begin() + (i+1)*INNER_BATCH_SIZE);
      std::vector<double> this_labels_flattened(labels_flattened.begin() + i*INNER_BATCH_SIZE, labels_flattened.begin() + (i+1)*INNER_BATCH_SIZE);

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

      vector<float> this_loss_output;
      string this_loss_name = LOSS_LAYER_PREFIX + std::to_string(k);
      GetFeatures(this_loss_name, &this_loss_output);
      //record the loss for domain k
      loss_history_k_domain_[k].push_back(this_loss_output[0]);
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
    }
    
}

void RegressorTrain::Step() {
  assert(net_->phase() == caffe::TRAIN);

  // Train the network.
  solver_.Step(1);
}

void RegressorTrain::SaveLossHistoryToFile(const std::string &save_path) {
  ofstream out_loss_file;
  out_loss_file.open(save_path.c_str());
  for (int i = 0;i< loss_history_k_domain_.size();i++) {
    for (int j = 0;j< loss_history_k_domain_[i].size();j++) {
      out_loss_file << loss_history_k_domain_[i][j] << " ";
    }
    out_loss_file << endl;
  }
  out_loss_file.close();
}

