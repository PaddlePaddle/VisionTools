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

/**
 * function:
 *  implements a producer-consumer pattern:
 *  the raw image data comes from python space,
 *  after a pipeline of image transformation using multi-threading,
 *  the data results will be put back to python space
 **/

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include "imageprocess.h"
#include "include/concurrent.h"
#include "include/transformer.h"
#include "logger.h"
#include "opencv2/opencv.hpp"
#include "util.h"
namespace vistool {

class ImageTransformer : public Transformer {
public:
  ImageTransformer();
  virtual ~ImageTransformer();

  virtual int init(const kv_conf_t &conf);

  virtual int set_processor(IProcessor *p);

  /**
   * @brief launch this transformer to work
   */
  virtual int start();

  /**
   * @brief stop this transformer, note that:
   *        after this, no more data will be feeded in, but remain results can
   * be fetched out
   */
  virtual int stop();

  /**
   * @brief test whether this transformer has already stopped
   */
  virtual bool is_stopped();

  /**
   * @brief put a new request to this transformer
   */
  virtual int put(const transformer_input_data_t &input);

  /**
   * @brief put a new request to this transformer
   */
  virtual int put(int id,
                  const char *image,
                  int img_len,
                  const char *label = "",
                  int label_len = 0);

  /**
   * @brief get a transformed result
   */
  virtual int get(transformer_output_data_t *output);

  /**
   * @brief apply image ops defined in 'this->_ops' to this 'input'
   */
  void process(const transformer_input_data_t &input);

  /**
   * @brief return the number of unconsumed data in this transformer,
   *        call this function in 'stopped' state please in case of
   * race-condition
   */
  int _unconsumed_num() { return (_in_num - _out_num); }

private:
  int _put_task(ITask *t);

private:
  IProcessor *_imgprocess;
  int _id;
  std::atomic<std::uint64_t> _in_num;
  std::atomic<std::uint64_t> _out_num;
  std::string _state;
  ThreadPool _workers;
  BlockingQueue<transformer_output_data_t *> _output_queue;
};

};  // namespace vistool
