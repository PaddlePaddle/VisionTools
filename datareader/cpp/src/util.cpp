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

#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>

#include <chrono>
#include <string>
#include <vector>

#include "include/logger.h"
#include "include/util.h"

namespace vistool {

int randInt(int min, int max) {
  if (min > max) {
    int t = min;
    min = max;
    max = t;
  }
  return (random() % (max - min + 1)) + min;
}

float randFloat(float min, float max) {
  if (min > max) {
    float t = min;
    min = max;
    max = t;
  }
  int64_t r = random();
  double d_r = static_cast<double>(r) / RAND_MAX;
  return static_cast<float>(min + (max - min) * d_r);
}

std::string string_format(const char* fmt, ...) {
  std::string str;
  va_list vl;
  va_start(vl, fmt);
  char tmp = '\0';
  int size = vsnprintf(&tmp, sizeof(tmp), fmt, vl);
  if (size <= 0) {
    str.clear();
  } else {
    str.resize(++size);
    va_start(vl, fmt);
    vsnprintf(&str[0], size, fmt, vl);
    str.resize(size - 1);
  }
  va_end(vl);
  return str;
}

std::vector<std::string> splitString(const std::string& s,
                                     const std::string& delim) {
  std::vector<std::string> ret;
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (index - last > 0) {
    ret.push_back(s.substr(last, index - last));
  }
  return ret;
}

int64_t now_usec() {
  std::chrono::time_point<std::chrono::system_clock> ts =
      std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             ts.time_since_epoch())
      .count();
}

BufLogger::BufLogger(bool log_out)
    : _start_ts(0), _log_out(log_out), _buffer("") {
  _start_ts = now_usec();
}

BufLogger::~BufLogger() {
  if (_log_out && this->_buffer.size() > 0) {
    this->append("[total_cost:%lums]", ((now_usec() - _start_ts) / 1000));
    LOG(INFO) << this->get();
  }
}

bool BufLogger::append(const char* fmt, ...) {
  std::string str;
  va_list vl;
  va_start(vl, fmt);
  char tmp = '\0';
  int size = vsnprintf(&tmp, sizeof(tmp), fmt, vl);
  if (size <= 0) {
    str.clear();
  } else {
    str.resize(++size);
    va_start(vl, fmt);
    vsnprintf(&str[0], size, fmt, vl);
    str.resize(size - 1);
  }
  va_end(vl);

  if (str.size() > 0) {
    this->_buffer.append(str);
    return true;
  } else {
    return false;
  }
}

};  // namespace vistool
