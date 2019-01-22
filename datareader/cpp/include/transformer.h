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

#pragma once
#include <map>
#include <string>
#include <vector>
#include "baseprocess.h"
namespace vistool {

/**
 * @brief base class to abstract the transformation on multiple type of data
 * sample,
 *        eg: image/video/audio
 *        processing steps:
 *        1, configure worker number, queue length
 *        2, configure operations needed to be applied on data samples
 *        3, launch this transformer
 *        4, feed data sample and fetch results
 */
class Transformer {
public:
  Transformer();
  virtual ~Transformer();

  /*
   * class method to create a transformer
   */
  static Transformer *create(const std::string &type);

  /*
   * class method to destroy a transformer
   */
  static void destroy(Transformer *t);

  /*
   * init this transformer
   */
  virtual int init(const kv_conf_t &conf) = 0;

  /*
   * set image processor for this transformer
   */
  virtual int set_processor(IProcessor *p) = 0;

  /*
   * launch this transformer to work
   */
  virtual int start() = 0;

  /*
   * stop this transformer
   */
  virtual int stop() = 0;

  /*
   * test whether this transformer has already stopped
   */
  virtual bool is_stopped() = 0;

  /*
   * put a new image processing task to this transformer
   */
  virtual int put(const transformer_input_data_t &input) = 0;

  /*
   * get a transformed image from this transformer
   */
  virtual int get(transformer_output_data_t *output) = 0;

  /*
   * put a new image processing task to this transformer
   */
  virtual int put(int id,
                  const char *image,
                  int image_len,
                  const char *label = "",
                  int label_len = 0) = 0;
};

};  // namespace vistool
