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

#include "Parallel.h"

/**
 * DecodeJpeg is an image processing API for interfacing Python and
 * C++ code. The Boost Python Library is used to wrap C++ interfaces.
 * This class is only an interface and there is no specific calculation.
 */
class DecodeJpeg {
public:
  DecodeJpeg(int threadNum,
             bool isTest,
             bool isColor,
             int resize_min_size,
             int cropSizeH,
             int cropSizeW,
             PyObject* meanValues) {
    tfhandlerPtr_ = std::make_shared<Parallel>(threadNum,
                                               isTest,
                                               isColor,
                                               resize_min_size,
                                               cropSizeH,
                                               cropSizeW,
                                               meanValues);
  }

  ~DecodeJpeg() {}

  /**
   * @brief This function calls the start function of Parallel
   *        to process image with multi-threads.
   * @param pysrc    The input image list with string type.
   * @param pylabel  The input label of image.
   *                 Its type is numpy.array with int32.
   * @param mode     Two mode:
   *                 0: the input is image buffer
   *                 1: the input is image file path.
   */
  int start(boost::python::list& pysrc, PyObject* pylabel, int mode) {
    int ret = tfhandlerPtr_->start(pysrc, pylabel, mode);
    return 0;
  }

  /**
   * @brief Return a tuple: (image, label).
   *        The image is transformed image.
   */
  boost::python::tuple get() { return tfhandlerPtr_->get(); }

private:
  std::shared_ptr<Parallel> tfhandlerPtr_;
};  // DecodeJpeg

/**
 * @brief Initialize the Python interpreter and numpy.
 */
static void initPython() {
  Py_Initialize();
  PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
  import_array();
  PyOS_setsig(SIGINT, sighandler);
}

/**
 * Use Boost.Python to expose C++ interface to Python.
 */
BOOST_PYTHON_MODULE(DeJpeg) {
  initPython();
  boost::python::numpy::initialize();
  boost::python::class_<DecodeJpeg>(
      "DecodeJpeg",
      boost::python::init<int, bool, bool, int, int, int, PyObject*>())
      .def("start", &DecodeJpeg::start)
      .def("get", &DecodeJpeg::get);
};
