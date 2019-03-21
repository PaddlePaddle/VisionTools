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
#include "baseprocess.h"
#include "lua_util.h"
#include "transformer.h"

namespace vistool {

class LuacvProcess : public IProcessor {
public:
  LuacvProcess();
  ~LuacvProcess();

  virtual int init(const ops_conf_t &ops);

  virtual int process(const transformer_input_data_t &input,
                      transformer_output_data_t &output);

protected:
  LuaStateMgr *_lua_mgr;
  int _tochw;
};
};
