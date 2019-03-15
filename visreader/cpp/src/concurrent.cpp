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

#include "include/concurrent.h"
//#include <pthread.h>
#include "include/logger.h"
#include "include/util.h"

namespace vistool {

ITask::ITask() : _finished(false), _cb(NULL), _arg(NULL), _result(-1000) {}

ITask::~ITask() {}

void ITask::wait() {
  std::unique_lock<std::mutex> lock(_mutex);

  _cond.wait(lock, [this]() { return _finished; });

  lock.unlock();
}

void ITask::notify() {
  std::lock_guard<std::mutex> lock(_mutex);
  _finished = true;
  _cond.notify_all();
}

int ITask::set_cb(task_cb_t cb, void *arg) {
  _cb = cb;
  _arg = arg;
  return 0;
}

void ITask::finish() {
  if (_cb) {
    _cb(_arg);
  } else {
    notify();
  }
}

void ThreadPool::run() {
  static int id = 0;
  id++;
  int order = id;
  LOG(INFO) << "run worker_" << order << " thread";

  while (!_exit_mark) {
    ITask *t = fetch_task(10);
    if (t) {
      t->execute();
    } else if (_exit_mark) {
      break;
    }
  }

  LOG(INFO) << "exit thread with id:" << order << " right now";
}

};  // namespace vistool
