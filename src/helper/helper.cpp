/*
 * helper.cpp
 *
 *  Created on: Jul 11, 2011
 *      Author: davheld
 */

#include "helper.h"

#include <string>
#include <cstdio>
#include <vector>
#include <cmath>

namespace bfs = boost::filesystem;

using std::string;
using std::sprintf;
using std::vector;

// Maximum length of a string created from a number.
const int kMaxNum2StringSize = 50;

// Default number of decimal places when converting a double to a string.
const int kDefaultDecimalPlaces = 6;


string int2str(const int num) {
 char num_buffer[kMaxNum2StringSize];
 sprintf(num_buffer, "%d", num);
 return num_buffer;
}

string num2str(const int num) {
  return int2str(num);
}

string double2str(const double num, const int decimal_places) {
  const string format = "%." + num2str(decimal_places) + "lf";
  char num_buffer[kMaxNum2StringSize];
  sprintf(num_buffer, format.c_str(), num);
  return num_buffer;
}

string float2str(const float num) {
 char num_buffer[kMaxNum2StringSize];
 sprintf(num_buffer, "%f", num);
 return num_buffer;
}

string unsignedint2str(const unsigned int num) {
  char num_buffer[kMaxNum2StringSize];
  sprintf(num_buffer, "%u", num);
  return num_buffer;
}

string num2str(const double num) {
  return double2str(num, kDefaultDecimalPlaces);
}

string num2str(const double num, int decimal_places) {
  return double2str(num, decimal_places);
}

string num2str(const float num) {
  return float2str(num);
}

string num2str(const unsigned int num) {
  return unsignedint2str(num);
}

string num2str(const size_t num) {
  char num_buffer[50];
  sprintf(num_buffer, "%zu", num);
  return num_buffer;
}

// *******File IO *************

void find_subfolders(const bfs::path& folder, vector<string>* sub_folders) {
  if (!bfs::is_directory(folder)) {
    printf("Error - %s is not a valid directory!\n", folder.c_str());
    return;
  }

  bfs::directory_iterator end_itr; // default construction yields past-the-end
  for (bfs::directory_iterator itr(folder); itr != end_itr; ++itr) {
    if (bfs::is_directory(itr->status())) {
      string filename = itr->path().filename().string();
      sub_folders->push_back(filename);
    }
  }

  // Sort the files by name.
  std::sort(sub_folders->begin(), sub_folders->end());
}

void find_matching_files(const bfs::path& folder, const boost::regex filter,
                         vector<string>* files) {
  if (!bfs::is_directory(folder)) {
    printf("Error - %s is not a valid directory!\n", folder.c_str());
    return;
  }

  bfs::directory_iterator end_itr; // default construction yields past-the-end
  for (bfs::directory_iterator itr(folder); itr != end_itr; ++itr) {
    if (bfs::is_regular_file(itr->status())) {
      string filename = itr->path().filename().string();

      boost::smatch what;
      if(boost::regex_match(filename, what, filter) ) {
        files->push_back(filename);
      }
    }
  }

  // Sort the files by name.
  std::sort(files->begin(), files->end());
}

double sample_rand_uniform() {
  // Generate a random number in (0,1)
  // http://www.cplusplus.com/forum/beginner/7445/
  return (rand() + 1) / (static_cast<double>(RAND_MAX) + 2);
}

double sample_exp(const double lambda) {
  // Sample from an exponential - http://stackoverflow.com/questions/11491458/how-to-generate-random-numbers-with-exponential-distribution-with-mean
  const double rand_uniform = sample_rand_uniform();
  return -log(rand_uniform) / lambda;
}

double sample_exp_two_sided(const double lambda) {
  // Determine which side of the two-sided exponential we are sampling from.
  const double pos_or_neg = (rand() % 2 == 0) ? 1 : -1;

  // Sample from an exponential - http://stackoverflow.com/questions/11491458/how-to-generate-random-numbers-with-exponential-distribution-with-mean
  const double rand_uniform = sample_rand_uniform();
  return log(rand_uniform) / lambda * pos_or_neg;
}



bool equalMat(cv::Mat &mat1, cv::Mat &mat2) {
  cv::Mat dst;
  cv::bitwise_xor(mat1, mat2, dst);        
  if(cv::countNonZero(dst) > 0) //check non-0 pixels
    //do stuff in case cv::Mat are not the same
    return false;
  else
    return true;
}

bool equalVector(std::vector<float> &a, std::vector<float> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i =0;i < a.size(); i++) {
    if(a[i] != b[i]) {
      return false;
    }
  }

  return true; 
}

float sigmoid(float x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

// load from file, assume space delimited
void loadMatrix(const std::string & file_path, MatrixXd & m, int rows, int cols) {
  m = MatrixXd::Constant(rows, cols, 0);

  ifstream in(file_path.c_str());
  string line;
  int i = 0;
  while (!in.eof()) {
    getline(in, line);
    if (line.length() != 0) {
      // parse the line
      istringstream iss(line);
      for (int j = 0; j < cols; j ++) {
        iss >> m(i, j);
      }
      i ++;
    }
  }

  assert(i == rows);
}

void saveMatrix(const std::string & file_path, const MatrixXd & m) {
  ofstream out(file_path.c_str());
  for (int i = 0; i < m.rows(); i ++) {
    for (int j = 0; j < m.cols(); j ++) {
      out << m(i, j) << " ";
    }
    out << endl;
  }
  out.close();
}

void saveFeatures(const std::vector<std::vector<float> > &features, const std::string file_name) {
  ofstream out(file_name.c_str());
  for (auto feature: features) {
    for (auto num: feature) {
      out << num << " ";
    }
    out << endl;
  }
  out.close();
}

void saveBboxesOTBFormat(const std::vector<BoundingBox> &bboxes, const std::string file_name) {
  ofstream out(file_name.c_str());
  for (auto box: bboxes) {
    out << box.x1_ << " " << box.y1_ <<" " <<box.get_width() << " " <<box.get_height() << endl;;
  }
  out.close();
}

void convertEigenToCVMat(const MatrixXd & src, cv::Mat & dst) {
  dst = cv::Mat(src.rows(), src.cols(), CV_64F);
  for(int i = 0; i < src.rows(); i ++) {
    for (int j = 0; j < src.cols(); j ++) {
      dst.at<double>(i, j) = src(i, j);
    }
  }
}

void convertCVToEigenMat(const cv::Mat & src, MatrixXd & dst) {
  dst = MatrixXd(src.rows, src.cols);
  for(int i = 0; i < src.rows; i ++) {
    for (int j = 0; j < src.cols; j ++) {
      dst(i, j) = src.at<double>(i, j);
    }
  } 
}