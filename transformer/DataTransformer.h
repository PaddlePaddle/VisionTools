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

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>

#include "ThreadPool.h"
#include "Queue.h"

/**
 * This is an image processing module with OpenCV, such as
 * resizing, scaling, mirroring, substracting the image mean...
 *
 * This class has a double BlockQueue and they shared the same memory.
 * It is used to avoid create memory each time. And it also can
 * return the data even if the data are processing in multi-threads.
 */
class DataTransformer {
public:
  DataTransformer(int threadNum,
                  int capacity,
                  bool isTest,
                  bool isColor,
                  int cropHeight,
                  int cropWidth,
                  int imgSize,
                  bool isEltMean,
                  bool isChannelMean,
                  float* meanValues);
  virtual ~DataTransformer() {
    if (meanValues_) {
      free(meanValues_);
    }
  }

  /**
   * @brief Start multi-threads to transform a list of input image string.
   *        This function reads an image from the specified buffer in the
   *        memory.
   * @param data   Data is the specified image buffer in the memory.
   * @param label  The label of input image.
   */
  void processImgString(std::vector<std::string>& data, int* labels);

  /**
   * @brief Start multi-threads to transform a list of input image file.
   *        This function loads image from the the specified file.
   * @param data   Data is an list of image file.
   * @param label  The label of input image.
   */
  void processImgFile(std::vector<std::string>& data, int* labels);

  /**
   * @brief Applies the transformation on one image Mat.
   *
   * @param img    The input image Mat to be transformed.
   * @param target target is used to save the transformed data.
   */
  void transform(cv::Mat& img, float* target);

  /**
   * @brief Save image Mat as file.
   *
   * @param filename The file name.
   * @param im       The image to be saved.
   */
  void imsave(std::string filename, cv::Mat& im) { cv::imwrite(filename, im); }

  /**
   * @brief Decode the image buffer, then calls transform() function.
   *
   * @param src  The input image buffer.
   * @param size The length of string buffer.
   * @param trg  trg is used to save the transformed data.
   */
  void transfromString(const char* src, const int size, float* trg);

  /**
   * @brief Load image form image file, then calls transform() function.
   *
   * @param src  The input image file.
   * @param trg  trg is used to save the transformed data.
   */
  void transfromFile(std::string imgFile, float* trg);

  /**
   * @brief Return the transformed data and its label.
   */
  void obtain(float* data, int* label);

private:
  int isTest_;
  int isColor_;
  int cropHeight_;
  int cropWidth_;
  int imgSize_;
  int capacity_;
  int fetchCount_;
  int fetchId_;
  bool isEltMean_;
  bool isChannelMean_;
  int numThreads_;
  float scale_;
  int imgPixels_;
  float* meanValues_;

  /**
   * Initialize the mean values.
   */
  void loadMean(float* values);

  /**
   * @brief Generates a random integer from Uniform({min, min + 1, ..., max}).
   * @param min The lower bound (inclusive) value of the random number.
   * @param max The upper bound (inclusive) value of the random number.
   *
   * @return
   * A uniformly random integer value from ({min, min + 1, ..., max}).
   */
  int Rand(int min, int max);

  typedef std::pair<float*, int> DataType;
  typedef std::shared_ptr<DataType> DataTypePtr;
  std::vector<DataTypePtr> prefetch_;
  ThreadPool threadPool_;
  std::vector<std::future<DataTypePtr>> results_;
  BlockingQueue<DataTypePtr> prefetchQueue_;
};  // class DataTransformer

#endif  // DATATRANSFORMER_H_
