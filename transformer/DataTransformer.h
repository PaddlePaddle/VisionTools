/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef DATATRANSFORMER_H_
#define DATATRANSFORMER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define DISABLE_COPY(T) \
  T(T&&) = delete;      \
  T(T const&) = delete; \
  void operator=(T const& t) = delete

typedef enum { CHANNEL_MEAN = 0, ELEMENT_MEAN = 1, NULL_MEAN = 2 } MeanType;

struct DataTransformerConfig {
  bool isTest_;
  bool isColor_;
  int cropHeight_;
  int cropWidth_;
  int imgSize_;  // short side
  MeanType meanType_;
  float scale_;
  int imgPixels_;  // the total pixels of transformed image
  float* meanValues_;
};

/**
 * This is an image processing module with OpenCV, such as
 * resizing, scaling, mirroring, substracting the image mean...
 */
class DataTransformer {
public:
  DISABLE_COPY(DataTransformer);

  DataTransformer(std::unique_ptr<DataTransformerConfig>&& config);
  virtual ~DataTransformer() {}

  /**
   * @brief Applies the transformation on one image Mat.
   *
   * @param img    The input image Mat to be transformed.
   * @param target target is used to save the transformed data.
   */
  void transform(cv::Mat& img, float* target) const;

  /**
   * @brief Save image Mat as file.
   *
   * @param filename The file name.
   * @param im       The image to be saved.
   */
  void imsave(std::string filename, cv::Mat& im) const {
    cv::imwrite(filename, im);
  }

  /**
   * @brief Decode the image buffer, then calls transform() function.
   *
   * @param src  The input image buffer.
   * @param size The length of string buffer.
   * @param trg  trg is used to save the transformed data.
   */
  void transfromString(const char* src, int size, float* trg) const;

  /**
   * @brief Load image form image file, then calls transform() function.
   *
   * @param src  The input image file.
   * @param trg  trg is used to save the transformed data.
   */
  void transfromFile(const char* imgFile, float* trg) const;

private:
  std::unique_ptr<DataTransformerConfig> config_;

  /**
   * @brief Generates a random integer from Uniform({min, min + 1, ..., max}).
   * @param min The lower bound (inclusive) value of the random number.
   * @param max The upper bound (inclusive) value of the random number.
   *
   * @return
   * A uniformly random integer value from ({min, min + 1, ..., max}).
   */
  int Rand(int min, int max) const;

};  // class DataTransformer

#endif  // DATATRANSFORMER_H_
