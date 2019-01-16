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
 */

#include "include/image_transformer.h"
#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "include/baseprocess.h"
#include "include/logger.h"
#include "include/util.h"
namespace vistool {

class MyTask : public ITask {
public:
  static MyTask *create(const transformer_input_data_t &input,
                        ImageTransformer *t) {
    return new MyTask(input, t);
  }

  static MyTask *create(ImageTransformer *t) { return new MyTask(t); }

  static void destroy(MyTask *task) {
    if (task) {
      delete task;
    }
  }

  void execute() {
    _transformer->process(_input);
    MyTask::destroy(this);
  }

  transformer_input_data_t *get_input() { return &this->_input; }

private:
  MyTask(const transformer_input_data_t &input, ImageTransformer *t)
      : _input(input), _transformer(t) {}

  explicit MyTask(ImageTransformer *t) : _transformer(t) {}

  virtual ~MyTask() {}

private:
  transformer_input_data_t _input;
  ImageTransformer *_transformer;
};

ImageTransformer::ImageTransformer()
    : _imgprocess(nullptr),
      _id(0),
      _in_num(0),
      _out_num(0),
      _state(""),
      _workers(),
      _output_queue(1000) {
  static std::atomic_int s_id_generator(0);
  _id = ++s_id_generator;
  LOG(INFO) << "ImageTransformer::ImageTransformer(id:" << _id << ")";
}

ImageTransformer::~ImageTransformer() {
  LOG(INFO) << "ImageTransformer::~ImageTransformer(id:" << _id << ")";
  if (!this->is_stopped()) {
    this->stop();
  }

  this->_workers.notify_exit();
  this->_workers.join();

  while (!this->_output_queue.is_empty()) {
    transformer_output_data_t *d = this->_output_queue.get(100);
    if (d) {
      LOG(INFO) << "delete unconsumed data[" << d->id << "] in output queue";
      delete d;
    } else {
      LOG(INFO) << "no more data need to delete now";
    }
  }
  if (_imgprocess != nullptr) {
    delete _imgprocess;
    _imgprocess = nullptr;
  }
  LOG(INFO) << "ImageTransformer::~ImageTransformer(id:" << _id << ") finished";
}

int ImageTransformer::init(const kv_conf_t &conf) {
  LOG(INFO) << "ImageTransformer::init(id:" << _id << ")";

  if (this->_state != "") {
    LOG(WARNING) << "transformer has already been inited";
    return -1;
  }
  KVConfHelper confhelper(conf);
  int thread_num = 0;
  if (!confhelper.get("thread_num", &thread_num, 0)) {
    LOG(WARNING) << "fail to get thread_num";
    return -2;
  }
  if (thread_num <= 0 || thread_num > 100) {
    LOG(WARNING) << "invalid thread_num param[" << thread_num << "]";
    return -3;
  }
  this->_workers.set_worker_num(thread_num);

  int worker_queue_limit = 0;
  if (!confhelper.get("worker_queue_limit", &worker_queue_limit, 0)) {
    LOG(WARNING) << "fail to get worker_queue_limit";
    return -4;
  }

  if (worker_queue_limit <= 0) {
    LOG(WARNING) << "invalid worker_queue_limit param[" << worker_queue_limit
                 << "]";
    return -5;
  }
  _output_queue.set_queue_limit(worker_queue_limit);
  this->_workers.set_queue_limit(worker_queue_limit);

  this->_state = "inited";
  return 0;
}

int ImageTransformer::set_processor(IProcessor *p) {
  LOG(INFO) << "ImageTransformer::set_processor";

  if (p == nullptr) {
    LOG(WARNING) << "ImageTransformer::set_processor(NULL) invalid pointer";
    return -1;
  }

  if (_imgprocess != nullptr) {
    delete _imgprocess;
    _imgprocess = nullptr;
  }
  _imgprocess = p;
  return 0;
}

int ImageTransformer::start() {
  LOG(INFO) << "ImageTransformer::start";
  if (this->_state != "inited") {
    LOG(WARNING) << "not allowed to start in this state[" << this->_state
                 << "]";
    return -1;
  }

  int ret = this->_workers.start();
  if (ret) {
    LOG(ERROR) << "failed to start workers in transformer";
    return -3;
  }
  this->_state = "started";
  return 0;
}

int ImageTransformer::stop() {
  LOG(INFO) << "ImageTransformer::stop";

  if (this->_state != "started") {
    LOG(ERROR) << "not allowed to stop in this state[" << this->_state << "]";
    return -1;
  }

  // do not exit the worker here for unfinished tasks
  this->_state = "stopped";
  return 0;
}

bool ImageTransformer::is_stopped() { return this->_state != "started"; }

int ImageTransformer::put(const transformer_input_data_t &input) {
  MyTask *t = MyTask::create(input, this);
  int ret = this->_put_task(t);
  if (ret) {
    MyTask::destroy(t);
  }
  return ret;
}

int ImageTransformer::put(int id,
                          const char *image,
                          int image_len,
                          const char *label,
                          int label_len) {
  MyTask *t = MyTask::create(this);
  transformer_input_data_t *input = t->get_input();
  input->id = id;
  input->data.resize(image_len);
  memcpy(&input->data[0], image, image_len);
  input->label.resize(label_len);
  memcpy(&input->label[0], label, label_len);

  int ret = this->_put_task(t);
  if (ret) {
    MyTask::destroy(t);
  }
  return ret;
}

int ImageTransformer::_put_task(ITask *t) {
  if (this->_state != "started") {
    LOG(WARNING) << "not allowed to put input in this state[" << this->_state
                 << "]";
    return -1;
  }

  int ret = this->_workers.append_task(t);

  if (ret) {
    LOG(FATAL) << "failed to append task to transformer";
    ret = -2;
  } else {
    _in_num++;
  }
  return ret;
}

/*
 * get data from transformed queue utill no data and stopped
 */
int ImageTransformer::get(transformer_output_data_t *output) {
  transformer_output_data_t *out = NULL;
  while (1) {
    out = this->_output_queue.get(100);
    if (out) {
      break;
    }
    if (is_stopped() && _unconsumed_num() == 0) {
      break;
    }
  }

  if (out) {
    _out_num++;
    *output = *out;
    delete out;
    return 0;
  } else {
    LOG(INFO) << "tranformer has stoped and got nothing";
    return 1;
  }
}

void ImageTransformer::process(const transformer_input_data_t &input) {
  transformer_output_data_t *output = new transformer_output_data_t;
  _imgprocess->process(input, *output);
  this->_output_queue.put(output);
  return;
}

};  // namespace vistool
