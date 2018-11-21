/**
 * Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

#ifndef DATAREADER_CPP_INCLUDE_IMAGE_UTIL_H_
#define DATAREADER_CPP_INCLUDE_IMAGE_UTIL_H_

#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

namespace vistool {

// error code for processed image
enum IMPROC_ERR_CODE_TYPE {
  IMPROC_OK = 0,
  IMPROC_INVALID_IMAGE = 2000,
  IMPROC_INVALID_PARAM = 2001,
  IMPROC_UNKOWN = 9999,
};

int readImage(const std::string &fname, std::vector<char> *buf);

int saveImage(const cv::Mat &img, const std::string &fname);

IMPROC_ERR_CODE_TYPE decode(const char *buf,
                            size_t bufsize,
                            cv::Mat *result,
                            int mode = cv::IMREAD_UNCHANGED);

IMPROC_ERR_CODE_TYPE resize(const cv::Mat &img,
                            const cv::Size &size,
                            cv::Mat *result,
                            int interpolation = cv::INTER_NEAREST,
                            double fx = 0,
                            double fy = 0);

IMPROC_ERR_CODE_TYPE crop(const cv::Mat &img,
                          const cv::Rect &rect,
                          cv::Mat *result);

IMPROC_ERR_CODE_TYPE rotate(const cv::Mat &img,
                            float angle,
                            cv::Mat *result,
                            int resample = cv::INTER_NEAREST);

IMPROC_ERR_CODE_TYPE flip(const cv::Mat &img, int flip_code, cv::Mat *result);

};  // namespace vistool

#endif  // DATAREADER_CPP_INCLUDE_IMAGE_UTIL_H_
