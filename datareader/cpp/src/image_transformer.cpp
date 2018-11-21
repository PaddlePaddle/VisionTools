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

#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include/image_transformer.h"
#include "include/image_util.h"
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

class KVConfHelper {
public:
  explicit KVConfHelper(const kv_conf_t &conf) : _conf(conf) {}

  virtual ~KVConfHelper() {}

  std::string get(const std::string &k, const std::string def = "") const {
    kv_conf_const_iter_t it = _conf.find(k);

    if (it != _conf.end()) {
      return it->second;
    } else {
      return def;
    }
  }

  bool get(const std::string &k, int *v, int def = -1) const {
    std::string value = get(k);
    try {
      if (value.size() > 0) {
        *v = std::stoi(value);
        return true;
      } else {
        *v = def;
        return false;
      }
    } catch (const std::exception &e) {
      std::string err_msg = string_format(
          "parse conf failed:with "
          "errmsg[%s] and input:[k:%s,v:%s]",
          e.what(),
          k.c_str(),
          value.c_str());
      LOG(WARNING) << err_msg;
      return false;
    }
  }

  bool get(const std::string &k, float *v, float def = -1.0) const {
    std::string value = get(k);
    if (value.size() > 0) {
      *v = std::stof(value);
      return true;
    } else {
      *v = def;
      return false;
    }
  }

  bool get(const std::string &k,
           std::vector<float> *v,
           std::string sep = ",") const {
    std::string value = get(k);

    v->clear();
    if (value.size() > 0) {
      for (auto &i : splitString(value, sep)) {
        v->push_back(std::stof(i));
      }
    }
    if (v->size() > 0) {
      return true;
    } else {
      return false;
    }
  }

  bool get(const std::string &k,
           std::vector<int> *v,
           std::string sep = ",") const {
    std::vector<float> fv;

    v->clear();
    if (!this->get(k, &fv, sep)) {
      return false;
    }
    for (size_t i = 0; i < fv.size(); i++) {
      v->push_back(static_cast<int>(fv[i]));
    }
    return true;
  }

private:
  const kv_conf_t _conf;
};

static int process_resize(const KVConfHelper &conf,
                          const cv::Mat &input,
                          cv::Mat *result,
                          std::string *errmsg,
                          BufLogger *logger) {
  int ret = 0;
  int img_w = input.cols;
  int img_h = input.rows;

  int resize_w = 0;
  int resize_h = 0;
  int short_size = 0;
  int interpo = -1;

  conf.get("interpolation", &interpo, cv::INTER_LINEAR);

  logger->append("[interpo:%d]", &interpo);
  if (conf.get("short_size", &short_size)) {
    float percent = static_cast<float>(short_size) / std::min(img_w, img_h);
    resize_w = static_cast<int>(round(img_w * percent));
    resize_h = static_cast<int>(round(img_h * percent));
    logger->append("[short_size:%d]", &short_size);
  } else if (!conf.get("resize_w", &resize_w) ||
             !conf.get("resize_h", &resize_h)) {
    ret = TRANS_ERR_RESIZE_INVALID_PARAM;
    *errmsg = string_format("not found valid 'resize_w' or 'resize_h'");
    return ret;
  }
  logger->append("[resize:{w:%d,h:%d}]", resize_w, resize_h);

  cv::Size size(resize_w, resize_h);
  ret = resize(input, size, result, interpo);
  if (ret || result->empty()) {
    *errmsg = string_format("failed to resize image with ret[%d]", ret);
    ret = TRANS_ERR_RESIZE_INVALID_PARAM;
    return ret;
  }
  return ret;
}

static int process_crop(const KVConfHelper &conf,
                        const cv::Mat &input,
                        cv::Mat *result,
                        std::string *errmsg,
                        BufLogger *logger) {
  int ret = 0;
  int crop_x = 0;
  int crop_y = 0;
  int crop_w = 0;
  int crop_h = 0;

  int crop_center = 0;

  int img_h = input.rows;
  int img_w = input.cols;
  if (conf.get("crop_center", &crop_center) && conf.get("crop_w", &crop_w) &&
      conf.get("crop_h", &crop_h)) {
    logger->append("[w:%d,h:%d,crop_center:%d]", crop_w, crop_h, crop_center);
    if (crop_center) {
      crop_x = (img_w - crop_w) / 2;
      crop_y = (img_h - crop_h) / 2;
    } else {
      crop_x = randInt(0, img_w - crop_w);
      crop_y = randInt(0, img_h - crop_h);
    }
  } else if (!conf.get("crop_x", &crop_x) || !conf.get("crop_y", &crop_y) ||
             !conf.get("crop_w", &crop_w) || !conf.get("crop_h", &crop_h)) {
    *errmsg = string_format("not found valid 'crop_[x|y|w|h]' params");
    ret = TRANS_ERR_CROP_INVALID_PARAM;
    return ret;
  }

  logger->append(
      "[crop:{x:%d,y:%d,w:%d,h:%d}]", crop_x, crop_y, crop_w, crop_h);
  cv::Rect rect(crop_x, crop_y, crop_w, crop_h);
  ret = crop(input, rect, result);
  if (ret || result->empty()) {
    *errmsg = string_format("failed to crop image with ret[%d]", ret);
    ret = TRANS_ERR_CROP_INVALID_PARAM;
  }
  return ret;
}

static int process_random_crop(const KVConfHelper &conf,
                               const cv::Mat &input,
                               cv::Mat *result,
                               std::string *errmsg,
                               BufLogger *logger) {
  int img_h = input.rows;
  int img_w = input.cols;
  int ret = 0;

  std::vector<int> final_size;  // final size for the cropped image
  std::vector<float> scale;
  std::vector<float> ratio;

  if (!conf.get("scale", &scale) || scale.size() != 2) {
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
    *errmsg = "not found 'scale' param";
    return ret;
  }
  logger->append("[scale:%.2f,%.2f]", scale[0], scale[1]);

  if (!conf.get("ratio", &ratio) || ratio.size() != 2) {
    *errmsg = "not found valid 'ratio' param";
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
    return ret;
  }
  logger->append("[ratio:%.2f,%.2f]", ratio[0], ratio[1]);

  if (!conf.get("final_size", &final_size) || final_size.size() != 2 ||
      final_size[0] * final_size[1] <= 0) {
    *errmsg = "not found valid 'final_size'";
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
    return ret;
  }
  int final_w = final_size[0];
  int final_h = final_size[1];
  logger->append("[final_size:%d,%d]", final_w, final_h);

  float aspect_ratio = sqrt(randFloat(ratio[0], ratio[1]));
  float w = 1.0 * aspect_ratio;
  float h = 1.0 / aspect_ratio;

  float bound = std::min((static_cast<float>(img_w) / img_h) / pow(w, 2),
                         ((static_cast<float>(img_h) / img_w)) / pow(h, 2));

  float scale_max = std::min(scale[1], bound);
  float scale_min = std::min(scale[0], bound);
  float target_area = img_w * img_h * randFloat(scale_min, scale_max);
  float target_size = sqrt(target_area);

  int int_w = static_cast<int>(floor(target_size * w));
  int int_h = static_cast<int>(floor(target_size * h));

  int i = randInt(0, img_w - int_w);
  int j = randInt(0, img_h - int_h);

  logger->append("[crop_rect:{x:%d,y:%d,w:%d,h:%d}", i, j, int_w, int_h);
  cv::Rect rect(i, j, int_w, int_h);
  cv::Mat cropped;
  ret = crop(input, rect, &cropped);
  if (ret || cropped.empty()) {
    *errmsg = string_format("rand_crop.crop failed with ret[%d]", ret);
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
    return ret;
  }

  cv::Size size(final_w, final_h);
  int interpo = -1;
  conf.get("interpolation", &interpo, cv::INTER_LANCZOS4);
  logger->append("[resize:w:%d,h:%d,interpo:%d]", final_w, final_h, interpo);
  ret = resize(cropped, size, result, interpo);
  if (ret || result->empty()) {
    *errmsg = string_format("rand_crop.resize failed with ret[%d]", ret);
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
    return ret;
  }

  return ret;
}

static int process_rotate(const KVConfHelper &conf,
                          const cv::Mat &input,
                          cv::Mat *result,
                          std::string *errmsg,
                          BufLogger *logger) {
  int ret = 0;
  int angle = 0;
  int random_range = 0;
  int resample = -1;
  if (conf.get("random_range", &random_range)) {
    angle = randInt(-random_range, random_range);
    logger->append("[random_range:%d]", random_range);
  } else if (!conf.get("angle", &angle)) {
    *errmsg = string_format(
        "rotate op not found valid "
        "'random_range' param");
    ret = TRANS_ERR_ROTATE_INVALID_PARAM;
    return ret;
  }
  logger->append("[angle:%d]", angle);

  conf.get("resample", &resample, cv::INTER_NEAREST);
  logger->append("[resample:%d]", resample);
  ret = rotate(input, static_cast<float>(angle), result, resample);
  if (ret || result->empty()) {
    *errmsg = string_format("failed to rotate image with ret[%d]", ret);
    ret = TRANS_ERR_RAND_CROP_INVALID_PARAM;
  }
  return ret;
}

static int process_flip(const KVConfHelper &conf,
                        const cv::Mat &input,
                        cv::Mat *result,
                        std::string *errmsg,
                        BufLogger *logger) {
  int ret = 0;
  int flip_code = 0;
  int random = 0;

  conf.get("random", &random, 0);
  if (!conf.get("flip_code", &flip_code)) {
    ret = TRANS_ERR_FLIP_INVALID_PARAM;
    *errmsg = string_format("not found valid 'flip_code'");
    return ret;
  }
  if (random && !randInt(0, 1)) {  // no need to flip
    *result = input;
    return ret;
  }

  logger->append("[random:%d, flip_code:%d]", random, flip_code);
  ret = flip(input, flip_code, result);
  if (ret || result->empty()) {
    *errmsg = string_format("failed to resize image with ret[%d]", ret);
    ret = TRANS_ERR_FLIP_INVALID_PARAM;
    return ret;
  }
  return ret;
}

typedef int processor_t(const KVConfHelper &conf,
                        const cv::Mat &input,
                        cv::Mat *result,
                        std::string *errmsg,
                        BufLogger *logger);

class ProcessorMgr {
public:
  ProcessorMgr() {
    _processors["resize"] = &process_resize;
    _processors["crop"] = &process_crop;
    _processors["random_crop"] = &process_random_crop;
    _processors["rotate"] = &process_rotate;
    _processors["flip"] = &process_flip;
  }
  ~ProcessorMgr() { _processors.clear(); }

  processor_t *get(const std::string &op_name) {
    for (auto &v : _processors) {
      if (v.first == op_name) {
        return v.second;
      }
    }
    return NULL;
  }

private:
  std::map<std::string, processor_t *> _processors;
};
static ProcessorMgr g_processor_mgr;

ImageTransformer::ImageTransformer()
    : _id(0),
      _in_num(0),
      _out_num(0),
      _ops(0),
      _state(""),
      _workers(),
      _output_queue(1000),
      _swapaxis(0) {
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
    return -1;
  }
  if (thread_num <= 0 || thread_num > 100) {
    LOG(WARNING) << "invalid thread_num param[" << thread_num << "]";
    return -2;
  }
  this->_workers.set_worker_num(thread_num);

  int worker_queue_limit = 0;
  if (!confhelper.get("worker_queue_limit", &worker_queue_limit, 0)) {
    LOG(WARNING) << "fail to get worker_queue_limit";
    return -1;
  }

  if (worker_queue_limit <= 0) {
    LOG(WARNING) << "invalid worker_queue_limit param[" << worker_queue_limit
                 << "]";
    return -3;
  }
  _output_queue.set_queue_limit(worker_queue_limit);
  this->_workers.set_queue_limit(worker_queue_limit);

  confhelper.get("swapaxis", &_swapaxis, 0);
  this->_state = "inited";
  return 0;
}

int ImageTransformer::add_op(const std::string &op_name,
                             const kv_conf_t &conf) {
  LOG(INFO) << "ImageTransformer::add_op(" << op_name << ")";

  if (this->_state != "inited") {
    LOG(WARNING) << "not allowed to add_op in this state[" << this->_state
                 << "]";
    return -1;
  }

  if (op_name != "decode" && !g_processor_mgr.get(op_name)) {
    LOG(WARNING) << "not support this op_name[" << op_name << "]";
    return -2;
  }

  kv_conf_const_iter_t it = conf.begin();
  kv_conf_t op_conf;
  op_conf["op_name"] = op_name;
  for (; it != conf.end(); ++it) {
    LOG(INFO) << "\"" << it->first << "\": \"" << it->second << "\"";
    op_conf[it->first] = it->second;
  }
  _ops.push_back(op_conf);
  LOG(INFO) << "succeed to add " << _ops.size() << " ops into this transformer";
  return 0;
}

int ImageTransformer::start() {
  LOG(INFO) << "ImageTransformer::start";
  if (this->_state != "inited") {
    LOG(WARNING) << "not allowed to start in this state[" << this->_state
                 << "]";
    return -1;
  }

  if (this->_ops.size() == 0) {
    LOG(ERROR) << "no operations added for this transformer";
    return -2;
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

static void swapaxis(const cv::Mat &mat, std::string *outstr) {
  if (mat.channels() == 1) {
    std::memcpy((void *)outstr->data(), mat.data, mat.total() * mat.elemSize());
  } else {
    std::vector<cv::Mat> channels(mat.channels());
    int imgsz = mat.rows * mat.cols;
    cv::split(mat, channels);
    for (size_t i = 0; i < channels.size(); i++) {
      const char *src = outstr->data() + i * imgsz;
      std::memcpy((void *)src,
                  channels[i].data,
                  channels[i].total() * channels[i].elemSize());
    }
  }
}
void ImageTransformer::process(const transformer_input_data_t &input) {
  int err_no = TRANS_ERR_OK;
  std::string err_msg = "";
  cv::Mat result;

  const char *input_img = &input.data[0];
  size_t input_len = input.data.size();
  BufLogger logger;
  logger.append("[process][input:{id:%d,size:%d}]", input.id, input_len);

  transformer_output_data_t *output = new transformer_output_data_t;
  output->id = input.id;
  output->label = input.label;

  try {
    for (size_t i = 0; i < _ops.size(); i++) {
      KVConfHelper conf(_ops[i]);
      const std::string op_name = conf.get("op_name");
      logger.append("{[op:%s]", op_name.c_str());
      int64_t start_ts = now_usec();
      if (op_name == "decode") {
        int mode = 0;
        conf.get("mode", &mode, cv::IMREAD_UNCHANGED);
        logger.append("[mode:%d]", mode);
        err_no = decode(input_img, input_len, &result, mode);
      } else {
        cv::Mat in = result;
        result = cv::Mat();
        processor_t *processor = g_processor_mgr.get(op_name);
        if (processor) {
          err_no = processor(conf, in, &result, &err_msg, &logger);
        } else {
          err_no = TRANS_ERR_INVALID_OP_NAME;
        }
      }

      int64_t op_cost = (now_usec() - start_ts) / 1000;
      logger.append(
          "ret:%d,elenum:%d,cost:%lums]}", err_no, result.total(), op_cost);
      if (err_no || result.empty()) {
        LOG(WARNING) << string_format(
            "failed to execute op[%s] "
            "with err_no[%d]",
            op_name.c_str(),
            err_no);
        if (!err_no) {
          err_no = TRANS_ERR_NO_OUTPUT;
        }
        break;
      }
    }
  } catch (const std::exception &e) {
    err_no = TRANS_ERR_LOGICERROR_EXCEPTION;
    err_msg = string_format("fatal logic error:[%s]", e.what());
    LOG(WARNING) << err_msg;
  }

  output->err_no = err_no;
  output->err_msg = err_msg;
  if (!err_no) {
    int size = result.total() * result.elemSize();
    output->data.resize(size);
    if (_swapaxis) {
      output->shape.push_back(result.channels());
      output->shape.push_back(result.rows);
      output->shape.push_back(result.cols);
      swapaxis(result, &output->data);
    } else {
      output->shape.push_back(result.rows);
      output->shape.push_back(result.cols);
      output->shape.push_back(result.channels());
      std::memcpy(&output->data[0], result.data, size);
    }
  } else {
    output->data.assign(input_img, input_len);
  }

  logger.append("[output:{size:%d,err_no:%d,err_msg:[%s]}]",
                output->data.size(),
                err_no,
                err_msg.c_str());

  this->_output_queue.put(output);
  return;
}

};  // namespace vistool
