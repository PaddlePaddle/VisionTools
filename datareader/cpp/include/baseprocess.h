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

namespace vistool {

typedef std::map<std::string, std::string> kv_conf_t;
typedef std::vector<vistool::kv_conf_t> ops_conf_t;
typedef std::map<std::string, std::string>::iterator kv_conf_iter_t;
typedef std::map<std::string, std::string>::const_iterator kv_conf_const_iter_t;

class KVConfHelper {
public:
  explicit KVConfHelper(const kv_conf_t &conf) : _conf(conf) {}

  virtual ~KVConfHelper() {}

  std::string get(const std::string &k, const std::string def = "") const;
  bool get(const std::string &k, int *v, int def = -1) const;
  bool get(const std::string &k, float *v, float def = -1.0) const;
  bool get(const std::string &k,
           std::vector<float> *v,
           std::string sep = ",") const;
  bool get(const std::string &k,
           std::vector<int> *v,
           std::string sep = ",") const;

private:
  const kv_conf_t _conf;
};

struct transformer_input_data_t {
  transformer_input_data_t() : id(0) {}
  ~transformer_input_data_t() {}

  transformer_input_data_t &operator=(const transformer_input_data_t &from) {
    this->id = from.id;
    this->data = from.data;
    this->label = from.label;
    return *this;
  }

  unsigned int id;
  std::string data;
  std::string label;
};

struct transformer_output_data_t {
  transformer_output_data_t() : id(0), err_no(0) {}
  ~transformer_output_data_t() {}

  transformer_output_data_t &operator=(const transformer_output_data_t &from) {
    this->id = from.id;
    this->err_no = from.err_no;
    this->err_msg = from.err_msg;
    this->shape = from.shape;
    this->label = from.label;
    this->data = from.data;
    return *this;
  }

  unsigned int id;
  int err_no;
  std::string err_msg;
  std::vector<int> shape;
  std::string label;
  std::string data;
};

enum TRANSFORMER_ERR_CODE_TYPE {
  TRANS_ERR_OK = 0,
  TRANS_ERR_NO_OUTPUT = 1,
  TRANS_ERR_LOGICERROR_EXCEPTION = 1,
  TRANS_ERR_LUA_EXCEPTION = 2,
  TRANS_ERR_LUA_KAGUYA_EXCEPTION = 3,
  TRANS_ERR_LUA_INVALID_OUTPUT = 4,
  TRANS_ERR_STOPPED = 1000,
  TRANS_ERR_INVALID_OP_NAME = 1001,
  TRANS_ERR_RESIZE_NO_INPUT = 1002,
  TRANS_ERR_RESIZE_INVALID_PARAM = 1003,
  TRANS_ERR_CROP_NO_INPUT = 1004,
  TRANS_ERR_CROP_INVALID_PARAM = 1005,
  TRANS_ERR_TRANSPOSE_NO_INPUT = 1006,
  TRANS_ERR_ROTATE_NO_INPUT = 1007,
  TRANS_ERR_ROTATE_INVALID_PARAM = 1008,
  TRANS_ERR_RAND_CROP_INVALID_PARAM = 1009,
  TRANS_ERR_FLIP_INVALID_PARAM = 1010,
};

class IProcessor {
public:
  static IProcessor *create(const std::string &classname,
                            const ops_conf_t &ops);

  static void destroy(IProcessor *p) {
    if (p) {
      delete p;
    }
  }

  IProcessor() {}
  virtual ~IProcessor() {}

  virtual int process(const transformer_input_data_t &input,
                      transformer_output_data_t &output) = 0;
};

};  // namespace vistool
