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

#include <string.h>
#include <fstream>

#include "image_util.h"
#include "logger.h"
#include "luacvprocess.h"
#include "opencv2/opencv.hpp"
#include "util.h"

namespace vistool {

LuacvProcess::LuacvProcess() : _lua_mgr(NULL), _tochw(0) {
  LOG(INFO) << "LuacvProcess::LuacvProcess";
}

LuacvProcess::~LuacvProcess() {
  LOG(INFO) << "LuacvProcess::LuacvProcess";
  if (_lua_mgr) {
    LuaStateMgr::destroy(_lua_mgr);
    _lua_mgr = NULL;
  }
}

int LuacvProcess::init(const ops_conf_t &ops) {
  KVConfHelper confhelper(ops[0]);

  if (ops.size() > 1) {
    LOG(FATAL) << "lua process can only accept one operator";
    return -1;
  }

  std::string opname = confhelper.get("op_name");
  if (opname != "lua_op") {
    LOG(FATAL) << "operator name[" << opname << "] is invalid";
    return -1;
  }

  std::string lua_script = confhelper.get("lua_fname");
  bool isfile = true;
  if (lua_script == "") {
    lua_script = confhelper.get("lua_code");
    if (lua_script == "") {
      LOG(FATAL) << "not found any 'lua_fname' or 'lua_code' param in 'lua_op'";
    } else {
      LOG(INFO) << "found 'lua_code' conf";
      isfile = false;
    }
  }

  confhelper.get("tochw", &_tochw, 0);
  LOG(INFO) << "set tochw to " << _tochw;

  int state_num = 1;
  confhelper.get("state_num", &state_num, state_num);
  LOG(INFO) << "create lua manager with state_num:" << state_num;
  _lua_mgr = LuaStateMgr::create(lua_script, isfile, state_num);
  if (_lua_mgr) {
    return 0;
  } else {
    return -1;
  }
}

typedef std::vector<cv::Mat> lua_param_type_t;
int LuacvProcess::process(const transformer_input_data_t &input,
                          transformer_output_data_t &output) {
  int err_no = TRANS_ERR_OK;
  std::string err_msg = "";

  size_t input_len = input.data.size();
  BufLogger logger;
  logger.append("[luacvprocess][input:{id:%d,size:%d}]", input.id, input_len);

  output.id = input.id;
  output.label = input.label;

  ScopedState scopped_state(_lua_mgr);
  kaguya::State state(scopped_state.value());

  cv::Mat inputmat(1, input.data.size(), CV_8U, (void *)input.data.c_str());
  cv::Mat labelmat(1, input.label.size(), CV_8U, (void *)input.label.c_str());
  lua_param_type_t lua_outputs;
  std::vector<cv::Mat> lua_inputs;
  lua_inputs.push_back(inputmat);
  lua_inputs.push_back(labelmat);

  cv::Mat result;
  try {
    lua_outputs = state["lua_main"].call<lua_param_type_t>(lua_inputs);
    if (lua_outputs.size() != 2) {
      err_no = TRANS_ERR_LUA_INVALID_OUTPUT;
      err_msg = formatString("invalid lua output[%u]", lua_outputs.size());
    } else {
      result = lua_outputs[0];
    }
  } catch (kaguya::LuaException &e) {
    err_no = TRANS_ERR_LUA_EXCEPTION;
    err_msg = formatString("lua exception:[%s]", e.what());
  } catch (kaguya::KaguyaException &e) {
    err_no = TRANS_ERR_LUA_KAGUYA_EXCEPTION;
    err_msg = formatString("kaguya exception:[%s]", e.what());
  }

  output.err_no = err_no;
  output.err_msg = err_msg;
  if (err_no || result.empty()) {
    LOG(WARNING) << formatString(
        "faield to execute lua script "
        "with err_no[%d] and err_msg[%s]",
        err_no,
        err_msg.c_str());
  } else {
    size_t totalsize = result.total() * result.elemSize();
    output.data.resize(totalsize);
    if (_tochw) {
      output.shape.push_back(result.channels());
      output.shape.push_back(result.rows);
      output.shape.push_back(result.cols);
      tochw(result, &output.data);
    } else {
      output.shape.push_back(result.rows);
      output.shape.push_back(result.cols);
      output.shape.push_back(result.channels());
      std::memcpy((void *)output.data.data(), result.data, totalsize);
    }
    output.label = mat2str(lua_outputs[1]);
  }

  state.garbageCollect();
  logger.append("[used_mem:%dKB]", state.useKBytes());
  logger.append("[output:{size:%d,err_no:%d,err_msg:[%s]}]",
                output.data.size(),
                err_no,
                err_msg.c_str());
  return 0;
}
};
