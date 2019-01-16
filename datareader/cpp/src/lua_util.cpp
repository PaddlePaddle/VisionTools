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

#include "lua_util.h"
#include <string>
#include <vector>
#include "logger.h"

#include <kaguya/another_binding_api.hpp>
#include "opencv2/opencv.hpp"

#include "image_util.h"

namespace vistool {

KAGUYA_BINDINGS(luac_basic) {
  kaguya::function("imencode", encodeImage);
  kaguya::function("imdecode", decodeImage);
  kaguya::function("mat2str", mat2str);
}

static int initLuaEnv(lua_State *lua_s, const std::string &lua_script) {
  kaguya::State state(lua_s);
  state.openlibs();
  state.openlib("luac_cv", luaopen_cv);
  state.openlib("luac_basic", vistool::luaopen_luac_basic);

  bool succeed = false;
  if (lua_script.find("\n") != std::string::npos) {
    LOG(INFO) << "load 'lua_main' from script code: " << lua_script.size();
    succeed = state.dostring(lua_script.c_str());
  } else {
    LOG(INFO) << "load 'lua_main' from script file:" << lua_script;
    succeed = state.dofile(lua_script.c_str());
  }
  if (!succeed) {
    LOG(FATAL) << "load 'lua_main' failed with error:"
               << lua_tostring(lua_s, -1);
    return -1;
  } else {
    return 0;
  }
}

class LuaStateMgrImpl : public LuaStateMgr {
public:
  LuaStateMgrImpl(const std::string &lua_script, int state_num = 1)
      : _lua_script(lua_script), _states(state_num) {
    for (int i = 0; i < state_num; i++) {
      lua_State *s = luaL_newstate();
      if (0 != initLuaEnv(s, _lua_script)) {
        lua_close(s);
        s = NULL;
        LOG(FATAL) << "failed to create lua states:" << i;
        break;
      } else {
        _states.put(s);
      }
    }
  }

  virtual ~LuaStateMgrImpl() {
    while (!_states.is_empty()) {
      lua_State *s = _states.get();
      lua_close(s);
    }
  }

  virtual lua_State *get() { return _states.get(); }

  void release(lua_State *s) { _states.put(s); }

private:
  std::string _lua_script;
  BlockingQueue<lua_State *> _states;
};

LuaStateMgr *LuaStateMgr::create(const std::string &lua_script, int state_num) {
  LuaStateMgrImpl *mgr = new LuaStateMgrImpl(lua_script, state_num);
  return mgr;
}
};
