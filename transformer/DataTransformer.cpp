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

#include <glog/logging.h>
#include <time.h>
#include <limits>

#include "DataTransformer.h"

DataTransformer::DataTransformer(
    std::unique_ptr<DataTransformerConfig>&& config)
    : config_(std::move(config)) {}

void DataTransformer::transfromFile(const char* imgFile, float* trg) const {
  int cvFlag =
      config_->isColor_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
  try {
    cv::Mat im = cv::imread(imgFile, cvFlag);
    if (!im.data) {
      LOG(ERROR) << "Could not decode image";
      LOG(ERROR) << im.channels() << " " << im.rows << " " << im.cols;
    }
    this->transform(im, trg);
  } catch (cv::Exception& e) {
    LOG(ERROR) << "Caught exception in cv::imdecode " << e.msg;
  }
}

void DataTransformer::transfromString(const char* src,
                                      int size,
                                      float* trg) const {
  try {
    cv::_InputArray imbuf(src, size);
    int cvFlag =
        config_->isColor_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::Mat im = cv::imdecode(imbuf, cvFlag);
    if (!im.data) {
      LOG(ERROR) << "Could not decode image";
      LOG(ERROR) << im.channels() << " " << im.rows << " " << im.cols;
    }
    this->transform(im, trg);
  } catch (cv::Exception& e) {
    LOG(ERROR) << "Caught exception in cv::imdecode " << e.msg;
  }
}

int DataTransformer::Rand(int min, int max) const {
  std::default_random_engine eng;
  std::uniform_int_distribution<int> dist(min, max);
  return dist(eng);
}

// TODO(qingqing): add more data argumentation operation
// and split this function.
void DataTransformer::transform(cv::Mat& cvImgOri, float* target) const {
  const int imgChannels = cvImgOri.channels();
  const int imgHeight = cvImgOri.rows;
  const int imgWidth = cvImgOri.cols;
  const bool doMirror = (!config_->isTest_) && Rand(0, 1);
  int hoff = 0;
  int woff = 0;
  int th = imgHeight;
  int tw = imgWidth;
  cv::Mat img;
  int imsz = config_->imgSize_;
  if (imsz > 0) {
    double ratio = imgHeight < imgWidth ? double(imsz) / double(imgHeight)
                                        : double(imsz) / double(imgWidth);
    th = int(double(imgHeight) * ratio);
    tw = int(double(imgWidth) * ratio);
    cv::resize(cvImgOri, img, cv::Size(tw, th));
  } else {
    img = cvImgOri;
  }

  cv::Mat cv_cropped_img = img;
  int cropH = config_->cropHeight_;
  int cropW = config_->cropWidth_;
  if (cropH && cropW) {
    if (!config_->isTest_) {
      hoff = Rand(0, th - cropH);
      woff = Rand(0, tw - cropW);
    } else {
      hoff = (th - cropH) / 2;
      woff = (tw - cropW) / 2;
    }
    cv::Rect roi(woff, hoff, cropW, cropH);
    cv_cropped_img = img(roi);
  } else {
    CHECK_EQ(cropH, imgHeight);
    CHECK_EQ(cropW, imgWidth);
  }
  int height = cropH;
  int width = cropW;
  int top_index;
  float scale = config_->scale_;
  float* meanVal = config_->meanValues_;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < imgChannels; ++c) {
        if (doMirror) {
          top_index = (c * height + h) * width + width - 1 - w;
        } else {
          top_index = (c * height + h) * width + w;
        }
        float pixel = static_cast<float>(ptr[img_index++]);
        switch (config_->meanType_) {
          case CHANNEL_MEAN: {
            target[top_index] = (pixel - meanVal[c]) * scale;
            break;
          }
          case ELEMENT_MEAN: {
            int mean_index = (c * height + h) * width + w;
            target[top_index] = (pixel - meanVal[mean_index]) * scale;
            break;
          }
          case NULL_MEAN: {
            target[top_index] = pixel * scale;
            break;
          }
          default:
            LOG(FATAL) << "Unsupport type";
        }
      }
    }
  }  // target: BGR
}
