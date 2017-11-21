#include "regressor_train.h"
#include <iostream>
#include <fstream>
#include <unordered_set>

const int kNumInputs = 2;
const bool kDoTrain = true;
const int LOSS_SAVE_ITER = 50;

using std::string;
using std::vector;
using caffe::Blob;
using caffe::Layer;
using caffe::LayerParameter;

// #define DEBUG_ROI_POOL_INPUT
// #define DEBUG_MINIBATCH_OHEM

// #define DEBUG_MDNET_FINETUNE
// #define DEBUG_OBSERVE_PREDICTION
// #define DEBUG_CROSS_FRAME_OHEM
// #define DEBUG_MDNET_TRAIN
#define DEBUG_TIME

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

  loss_history_.resize((K == -1 ? 1: K));
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

void RegressorTrain::FinetuneOHEMTrain(const std::vector<cv::Mat> & candidate_images, 
                                       const std::vector<double> & labels) {
  // Set the candidates
  SetCandidates(candidate_images);
  
  // Set the labels
  set_labels(labels);

#ifdef DEBUG_MDNET_FINETUNE 
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

  std::vector<float> labels_in;
  GetFeatures("label", &labels_in);

  assert (labels_in.size() == input_candidate_images.size());

  for (int b = 0; b < input_candidate_images.size(); b++) {
    if (labels_in[b] == 1) {
      cv::imshow("pos candidate" + std::to_string(1), input_candidate_images[b]);
    }
    else {
      cv::imshow("neg candidate" + std::to_string(1), input_candidate_images[b]);
    }
    cv::waitKey(1);
  }
#endif

  // Train the network.
  Step();

#ifdef DEBUG_OBSERVE_PREDICTION
  vector<float> predictions;
  GetFeatures("flatten_fc6", &predictions);

  vector<float> positive_probs;
  for (int i = 0; i < predictions.size() / 2; i ++) {
    float this_positive_prob = exp(predictions[2*i + 1]) / (exp(predictions[2*i]) + exp(predictions[2*i + 1]));
    positive_probs.push_back(this_positive_prob);
  }

  std::vector<float> this_loss_output;
  GetFeatures("loss", &this_loss_output);

#endif

}

vector<int> RegressorTrain::GetOHEMIndices(const std::vector<cv::Mat>& image_currs,
  const vector<BoundingBox> & neg_bboxes,
  const vector<int> & corres_frame_ids,
  const int & num) {

    // collect the scores in a mini batch manner using MINI_BATCH_SIZE_OHEM
    const vector<string> & layer_names = net_->layer_names();
    int layer_flatten_fc6_idx = FindLayerIndexByName(layer_names, "flatten_fc6");
    vector<double> neg_bboxes_probs;

    int total_size = neg_bboxes.size();
    int num_inner_batches =  total_size / MINI_BATCH_SIZE_OHEM;      
    for (int j = 0; j < num_inner_batches; j ++) {
      std::vector<BoundingBox> this_neg_bboxes(neg_bboxes.begin() + j*MINI_BATCH_SIZE_OHEM, 
      neg_bboxes.begin() + std::min((j+1)*MINI_BATCH_SIZE_OHEM, total_size));
      std::vector<int> this_corres_frame_ids(corres_frame_ids.begin() + j*MINI_BATCH_SIZE_OHEM, 
      corres_frame_ids.begin() + std::min((j+1)*MINI_BATCH_SIZE_OHEM, total_size));
      
      // gather the actual roi images of these negative bboxes
      vector<cv::Mat> neg_candidate_images;
      for (int i = 0; i < this_neg_bboxes.size(); i ++) {
        cv::Mat out;
        this_neg_bboxes[i].CropBoundingBoxOutImage(image_currs[this_corres_frame_ids[i]], &out);
        neg_candidate_images.push_back(out);
      }
      SetCandidates(neg_candidate_images);
      
      net_->ForwardTo(layer_flatten_fc6_idx);
  
      vector<float> probabilities;
      GetProbOutput(&probabilities);
    
      vector<float> positive_probabilities;
      for(int i = 0; i < this_neg_bboxes.size(); i++) {
        positive_probabilities.push_back(probabilities[2*i+1]);
      }

      neg_bboxes_probs.insert(neg_bboxes_probs.end(), positive_probabilities.begin(), positive_probabilities.end());
    }

    assert (neg_bboxes_probs.size() == neg_bboxes.size());

    
    // take the bboxes with highest num pos scores
    vector<int> sorted_indices(neg_bboxes.size());
    iota(sorted_indices.begin(), sorted_indices.end(), 0); // 0, 1, ... neg_bboxes.size() - 1
    
    // sort indices
    sort(sorted_indices.begin(), sorted_indices.end(),
        [&neg_bboxes_probs](int i, int j){
          return neg_bboxes_probs[i] > neg_bboxes_probs[j];
        });


    return vector<int>(sorted_indices.begin(), sorted_indices.begin() + num);
}

void RegressorTrain::FineTuneOHEM(const std::vector<cv::Mat>& image_currs,
  const std::vector<BoundingBox>& pos_candidate_bboxes,
  const std::vector<int>& pos_corres_frame_ids,
  const std::vector<BoundingBox>& neg_candidate_bboxes,
  const std::vector<int>& neg_corres_frame_ids,  
  int max_iter,
  int num_nohem) {
    assert(pos_candidate_bboxes.size() == pos_corres_frame_ids.size());
    assert(neg_candidate_bboxes.size() == neg_corres_frame_ids.size());

#ifdef DEBUG_TIME
    hrt_.reset();
    hrt_.start();
#endif

    // first, sample max_iter * 32 positive samples, max_iter * 4 * 256 negative samples
    std::mt19937 engine;
    engine.seed(time(NULL));

    vector<BoundingBox> pos_samples;
    vector<int> pos_sample_frame_ids;
    while (pos_samples.size() < max_iter * POS_SAMPLE_FINETUNE) {
      int this_sample_num = min(max_iter * POS_SAMPLE_FINETUNE - pos_samples.size(), pos_candidate_bboxes.size());
      std::vector<int> shuffle_indices(pos_candidate_bboxes.size());
      iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
      std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), engine);
      for(int i = 0; i < this_sample_num; i++) {
        pos_samples.push_back(pos_candidate_bboxes[shuffle_indices[i]]);
        pos_sample_frame_ids.push_back(pos_corres_frame_ids[shuffle_indices[i]]);
      }
    }

    vector<BoundingBox> neg_samples;
    vector<int> neg_sample_frame_ids;
    while (neg_samples.size() < max_iter * NEG_SAMPLE_FINETUNE) {
      int this_sample_num = min(max_iter * NEG_SAMPLE_FINETUNE - neg_samples.size(), neg_candidate_bboxes.size());
      std::vector<int> shuffle_indices(neg_candidate_bboxes.size());
      iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
      std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), engine);
      for(int i = 0; i < this_sample_num; i++) {
        neg_samples.push_back(neg_candidate_bboxes[shuffle_indices[i]]);
        neg_sample_frame_ids.push_back(neg_corres_frame_ids[shuffle_indices[i]]);
      }
    }
#ifdef DEBUG_TIME
    hrt_.stop();
    cout << "time spent for sample candidates in FineTuneOHEM: " << hrt_.getMilliseconds() << " ms" << endl;   
    cout << "number of positive samples: " << pos_samples.size() << ", number of negative samples:" << neg_samples.size() << endl; 
#endif

#ifdef DEBUG_CROSS_FRAME_OHEM
    vector<vector<BoundingBox> > hard_examples;
    hard_examples.resize(image_currs.size());
#endif

    // for each iter, get Choose(4*256, 96) nohem, then backprop
    for (int i = 0; i < max_iter; i++) {
      vector<BoundingBox> this_neg_samples(neg_samples.begin() + i*NEG_SAMPLE_FINETUNE, 
      neg_samples.begin() + (i+1)*NEG_SAMPLE_FINETUNE);
      vector<int> this_neg_samples_frame_ids(neg_sample_frame_ids.begin() + i*NEG_SAMPLE_FINETUNE,
      neg_sample_frame_ids.begin() + (i+1)*NEG_SAMPLE_FINETUNE);
      vector<int> ohem_indices = GetOHEMIndices(image_currs, this_neg_samples, this_neg_samples_frame_ids, num_nohem); // ohem_indices are into this_neg_samples

#ifdef DEBUG_CROSS_FRAME_OHEM
      hard_examples.clear();
      hard_examples.resize(image_currs.size());
      for (auto idx : ohem_indices) {
        hard_examples[this_neg_samples_frame_ids[idx]].push_back(this_neg_samples[idx]);
      }
      // visualise
      for (int j = 0; j < image_currs.size(); j ++) {
        cv::Mat this_image = image_currs[j].clone();
        for (auto bbox: hard_examples[j]) {
          bbox.Draw(0, 0, 255, &this_image);
        }
        cv::imshow("image_curr " + std::to_string(j), this_image);
        cv::waitKey(0);
      }
#endif

      // prepare 96 + 32 finetuning crops, shuffle, then backprop
      vector<cv::Mat> candidate_images;
      vector<double> candidate_labels;

      for (auto idx: ohem_indices) {
        cv::Mat out;
        this_neg_samples[idx].CropBoundingBoxOutImage(image_currs[this_neg_samples_frame_ids[idx]], &out);
        candidate_images.push_back(out);
        candidate_labels.push_back(NEG_LABEL);
      }

      for (int idx = i*POS_SAMPLE_FINETUNE; idx < (i+1)*POS_SAMPLE_FINETUNE; idx++) {
        cv::Mat out;
        pos_samples[idx].CropBoundingBoxOutImage(image_currs[pos_sample_frame_ids[idx]], &out);
        candidate_images.push_back(out);
        candidate_labels.push_back(POS_LABEL);
      }

      // shuffle
      assert(candidate_images.size() == candidate_labels.size());
      std::vector<int> shuffle_indices(candidate_images.size());
      iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
      std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), engine);

      vector<cv::Mat> candidate_images_backprop;
      vector<double> candidate_labels_backprop;
      for (auto idx : shuffle_indices) {
        candidate_images_backprop.push_back(candidate_images[idx]);
        candidate_labels_backprop.push_back(candidate_labels[idx]);
      }

      //backprop
      FinetuneOHEMTrain(candidate_images_backprop, candidate_labels_backprop);

    }
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

        Train(this_image_curr,
              this_candidates_flattened,
              this_labels_flattened,
              k);
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
  
#ifdef DEBUG_MINIBATCH_OHEM
    cout << "check neg probs are sorted: " << neg_probs[idx[0]] << ", " << neg_probs[idx[1]] << ", " << neg_probs[idx[2]] << endl;
    // visualise the hard examples in this minibatch
    std::vector<cv::Mat> image_curr_scaled_splitted;
    WrapOutputBlob("candidate", &image_curr_scaled_splitted);
    cv::Mat image_curr_scale;
    cv::merge(image_curr_scaled_splitted, image_curr_scale);
   
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
    assert (bboxes_in.size() == candidates_bboxes.size());
    for (int i = 0; i < labels_in.size(); i++) {
      int j;
      for (j = 0; j < neg_bag.size(); j++) {
        if (neg_bag[j] == i) {
          break;
        }
      }

      if(j != neg_bag.size() && backprop_idxes.find(i) != backprop_idxes.end()) {
        if (labels_in[i] == 1) {
          cout << "not possible to have negative example with label 1!" << endl;
          exit(1);
        }
        bboxes_in[i].Draw(0, 0, 255, &image_curr_scale_origin);
      }
      else if (labels_in[i] == 1) {
        bboxes_in[i].Draw(255, 0, 0, &image_curr_scale_origin);
      }
    }
  
    cv::imshow("OHEM rois on scaled image:", image_curr_scale_origin);
    cv::waitKey(0);

#endif
    
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
        loss_history_[0].push_back(this_loss_output[0]);
        InvokeSaveLossIfNeeded();
      }
    }
}

void RegressorTrain::Train(const cv::Mat &image_curr,
                           const std::vector<BoundingBox> candidates_bboxes,
                           const std::vector<double> &labels_flattened,
                           int k) {
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

      // Set the candidates
      set_candidate_images(image_curr, candidates_bboxes);

      // Set the labels
      set_labels(labels_flattened);

#ifdef DEBUG_MDNET_TRAIN
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
    
      std::vector<float> labels_in;
      GetFeatures("label", &labels_in);
    
      assert (labels_in.size() == input_candidate_images.size());
    
      for (int b = 0; b < input_candidate_images.size(); b++) {
        if (labels_in[b] == 1) {
          cv::imshow("pos candidate" + std::to_string(1), input_candidate_images[b]);
        }
        else {
          cv::imshow("neg candidate" + std::to_string(1), input_candidate_images[b]);
        }
        cv::waitKey(500);
      }
#endif

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

      // save loss
      if (loss_save_path_.length() != 0 && loss_history_.size() > k ) {
        std::vector<float> this_loss_output;
        GetFeatures(LOSS_LAYER_PREFIX + std::to_string(k), &this_loss_output);
        loss_history_[k].push_back(this_loss_output[0]);
        InvokeSaveLossIfNeeded();
      }

    }
    else {
      // Fine Tuning, only one domain, just normally step

      // Set the candidates
      set_candidate_images(image_curr, candidates_bboxes);

      // Set the labels
      set_labels(labels_flattened);

#ifdef DEBUG_MDNET_FINETUNE 
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

  std::vector<float> labels_in;
  GetFeatures("label", &labels_in);

  assert (labels_in.size() == input_candidate_images.size());

  for (int b = 0; b < input_candidate_images.size(); b++) {
    if (labels_in[b] == 1) {
      cv::imshow("pos candidate" + std::to_string(1), input_candidate_images[b]);
    }
    else {
      cv::imshow("neg candidate" + std::to_string(1), input_candidate_images[b]);
    }
    cv::waitKey(1);
  }
#endif

      // Train the network.
      Step();

#ifdef DEBUG_OBSERVE_PREDICTION
  vector<float> predictions;
  GetFeatures("flatten_fc6", &predictions);

  vector<float> positive_probs;
  for (int i = 0; i < predictions.size() / 2; i ++) {
    float this_positive_prob = exp(predictions[2*i + 1]) / (exp(predictions[2*i]) + exp(predictions[2*i + 1]));
    positive_probs.push_back(this_positive_prob);
  }

  std::vector<float> this_loss_output;
  GetFeatures("loss", &this_loss_output);

#endif

      if (loss_save_path_.length() != 0) {
        std::vector<float> this_loss_output;
        GetFeatures("loss", &this_loss_output);
        loss_history_[0].push_back(this_loss_output[0]);
        InvokeSaveLossIfNeeded();
      }

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
  int num_iter_to_save = loss_history_[loss_history_.size() - 1].size();
  for (int i = 0; i < num_iter_to_save; i ++) {
    for (int k = 0; k < loss_history_.size(); k++) {
        out_loss_file << loss_history_[k][i] << " ";
    }
    out_loss_file << endl;
  }
  out_loss_file.close();
  for (int k = 0; k < loss_history_.size(); k ++) {
    loss_history_[k].clear(); // clear after write
  }
}

void RegressorTrain::InvokeSaveLossIfNeeded() {
  if (loss_history_[loss_history_.size() - 1].size() >= LOSS_SAVE_ITER) {
    SaveLossHistoryToFile(loss_save_path_);
  }
}