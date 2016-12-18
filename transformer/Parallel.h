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

#include <Python.h>
#include <glog/logging.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>

#include "DataTransformer.h"
#include "ThreadPool.h"

namespace bn = boost::python::numpy;

class Parallel {
public:
  Parallel(int threadNum,
           bool isTest,
           bool isColor,
           int resizeMinSize,
           int cropSizeH,
           int cropSizeW,
           PyObject* meanValues)
      : threadPool_(threadNum) {
    int channel = isColor ? 3 : 1;
    MeanType meanType;
    float* mean = NULL;
    if (meanValues || meanValues != Py_None) {
      if (!PyArray_Check(meanValues)) {
        LOG(FATAL) << "Object is not a numpy array";
      }
      pyTypeCheck(meanValues);
      int size = PyArray_SIZE(reinterpret_cast<PyArrayObject*>(meanValues));
      mean = (float*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(meanValues));
      meanType = (size == channel) ? CHANNEL_MEAN : NULL_MEAN;
      meanType =
          (size == channel * cropSizeH * cropSizeW) ? ELEMENT_MEAN : meanType;
    }

    imgPixels_ = channel * cropSizeH * cropSizeW;

    DataTransformerConfig* conf = new DataTransformerConfig;
    conf->isTest_ = isTest;
    conf->isColor_ = isColor;
    conf->cropHeight_ = cropSizeH;
    conf->cropWidth_ = cropSizeW;
    conf->imgSize_ = resizeMinSize;
    conf->meanType_ = meanType;
    conf->scale_ = 1.0;
    conf->imgPixels_ = imgPixels_;
    conf->meanValues_ = mean;

    transformerPtr_ = std::unique_ptr<DataTransformer>(
        new DataTransformer(std::unique_ptr<DataTransformerConfig>(conf)));
  }

  ~Parallel() {}

  int start(boost::python::list& pysrc, PyObject* pylabel, int mode) {
    int num = len(pysrc);
    int* labels = (int*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(pylabel));
    for (int i = 0; i < num; ++i) {
      const char* buf = boost::python::extract<const char*>(pysrc[i]);
      int buflen = len(pysrc[i]);
      int label = labels[i];
      Py_intptr_t shape[1] = {this->imgPixels_};
      DataTypePtr trg = std::make_shared<DataType>(
          boost::python::numpy::zeros(
              1, shape, boost::python::numpy::dtype::get_builtin<float>()),
          0);
      results_.emplace_back(
          threadPool_.enqueue([this, buf, buflen, label, trg, mode]() {
            trg->second = label;
            float* data = (float*)((trg->first).get_data());
            if (mode == 0) {
              this->transformerPtr_->transfromString(buf, buflen, data);
            } else if (mode == 1) {
              this->transformerPtr_->transfromFile(buf, data);
            } else {
              LOG(FATAL) << "Unsupport mode " << mode;
            }
            return trg;
          }));
    }
    return 0;
  }

  boost::python::tuple get() {
    DataTypePtr ret = results_.front().get();
    results_.pop_front();
    return boost::python::make_tuple(ret->first, ret->second);
  }

private:
  /**
   * @brief Check whether the type of PyObject is valid or not.
   */
  void pyTypeCheck(PyObject* o) {
    int typenum = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(o));

    // clang-format off
    int type =
        typenum == NPY_UBYTE ? CV_8U :
        typenum == NPY_BYTE ? CV_8S :
        typenum == NPY_USHORT ? CV_16U :
        typenum == NPY_SHORT ? CV_16S :
        typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
        typenum == NPY_FLOAT ? CV_32F :
        typenum == NPY_DOUBLE ? CV_64F : -1;
    // clang-format on

    if (type < 0) {
      LOG(FATAL) << "toMat: Data type = " << type << " is not supported";
    }
  }

  /**
   * @brief Check whether the PyObject is writable or not.
   */
  void pyWritableCheck(PyObject* o) {
    CHECK(PyArray_ISWRITEABLE(reinterpret_cast<PyArrayObject*>(o)));
  }

  /**
   * @brief Check whether the PyObject is c-contiguous or not.
   */
  void pyContinuousCheck(PyObject* o) {
    CHECK(PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject*>(o)));
  }

  int imgPixels_;

  /**
   * @brief An object of DataTransformer, which is used to call
   *        the image processing funtions.
   */
  std::unique_ptr<DataTransformer> transformerPtr_;

  ThreadPool threadPool_;

  typedef std::pair<boost::python::numpy::ndarray, int> DataType;
  typedef std::shared_ptr<DataType> DataTypePtr;
  std::deque<std::future<DataTypePtr>> results_;

};  // Parallel
