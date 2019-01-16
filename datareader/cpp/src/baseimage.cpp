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

#include "baseprocess.h"
#include "include/imageprocess.h"
#include "logger.h"
#include "util.h"
#ifdef WITH_LUA
#include "include/luacvprocess.h"
#endif

namespace vistool {

std::string KVConfHelper::get(const std::string &k,
                              const std::string def) const {
  kv_conf_const_iter_t it = _conf.find(k);

  if (it != _conf.end()) {
    return it->second;
  } else {
    return def;
  }
}

bool KVConfHelper::get(const std::string &k, int *v, int def) const {
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
    std::string err_msg = formatString(
        "parse conf failed:with "
        "errmsg[%s] and input:[k:%s,v:%s]",
        e.what(),
        k.c_str(),
        value.c_str());
    LOG(WARNING) << err_msg;
    return false;
  }
}

bool KVConfHelper::get(const std::string &k, float *v, float def) const {
  std::string value = get(k);
  if (value.size() > 0) {
    *v = std::stof(value);
    return true;
  } else {
    *v = def;
    return false;
  }
}

bool KVConfHelper::get(const std::string &k,
                       std::vector<float> *v,
                       std::string sep) const {
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

bool KVConfHelper::get(const std::string &k,
                       std::vector<int> *v,
                       std::string sep) const {
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

IProcessor *IProcessor::create(const std::string &classname,
                               const ops_conf_t &ops) {
  if (classname == "ImageProcess") {
    ImageProcess *p = new ImageProcess();
    if (!p->init(ops)) {
      return p;
    } else {
      delete p;
    }
  } else if (classname == "LuacvProcess") {
#ifdef WITH_LUA
    vistool::LuacvProcess *p = new LuacvProcess();
    if (!p->init(ops)) {
      return p;
    } else {
      delete p;
    }
#endif
  }
  LOG(FATAL) << "failed to create IProcessor with classname:" << classname;
  return nullptr;
}
};  // namespace vistool
