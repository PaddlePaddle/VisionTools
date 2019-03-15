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
#include <string>
#include "concurrent.h"

extern "C" {
#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"
}

#include "kaguya/kaguya.hpp"

extern "C" {
int luaopen_cv(lua_State *L);
}

namespace vistool {

class LuaStateMgr {
public:
  LuaStateMgr(){};
  virtual ~LuaStateMgr(){};

  static LuaStateMgr *create(const std::string &lua_script,
                             bool isfile,
                             int state_num = 1);

  static void destroy(LuaStateMgr *mgr) { delete mgr; }

  virtual lua_State *get() = 0;
  virtual void release(lua_State *state) = 0;
};

class ScopedState {
public:
  ScopedState(LuaStateMgr *mgr) : _state(NULL), _mgr(mgr) {}
  ~ScopedState() {
    if (_state) {
      _mgr->release(_state);
      _state = NULL;
    }
  }

  inline lua_State *value() {
    if (!_state) {
      _state = _mgr->get();
    }
    return this->_state;
  }

private:
  lua_State *_state;
  LuaStateMgr *_mgr;
};
};
