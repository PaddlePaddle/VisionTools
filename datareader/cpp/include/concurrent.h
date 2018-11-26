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

/**
 * function:
 *  this module provides multi-threading interfaces
 **/

#ifndef DATAREADER_CPP_INCLUDE_CONCURRENT_H_
#define DATAREADER_CPP_INCLUDE_CONCURRENT_H_

#include <stdarg.h>
#include <stdint.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace vistool {

/**
 * @brief a thread-safe blocking queue which implements producer-consumer
 * pattern
 */
template <typename T>
class BlockingQueue {
public:
  explicit BlockingQueue(int queue_len = 100) : _queue_limit(queue_len) {}

  int set_queue_limit(int queue_len) {
    this->_queue_limit = queue_len;
    return 0;
  }

  inline bool is_full() { return _queue.size() >= _queue_limit; }

  inline bool is_empty() { return _queue.empty(); }

  inline int length() const { return _queue.size(); }

  /**
   * @brief get an element from this queue, which will be timed out
   *        if 'wait_ms' is greater than 0
   *
   * Return the element if succeed, otherwise NULL
   */
  T get(uint32_t wait_ms = 0) {
    std::unique_lock<std::mutex> lock(_mutex);
    bool cond_meet = false;
    if (!wait_ms) {
      _cond.wait(lock, [this]() { return !is_empty(); });
      cond_meet = true;
    } else {
      if (_cond.wait_for(lock, std::chrono::milliseconds(wait_ms), [this]() {
            return !is_empty();
          })) {
        cond_meet = true;
      } else {
        cond_meet = false;
      }
    }

    T ele = NULL;
    if (cond_meet) {
      ele = _queue.front();
      _queue.pop_front();
    }

    lock.unlock();
    _cond.notify_all();
    return ele;
  }

  /**
   * @brief get an element from this queue, which maybe blocked forever
   *        if no element is available
   */
  void get(T *res) {
    std::unique_lock<std::mutex> lock(_mutex);

    _cond.wait(lock, [this]() { return !is_empty(); });
    *res = _queue.front();
    _queue.pop_front();

    lock.unlock();
    _cond.notify_all();
    return;
  }

  /**
   * @brief put an element to this queue
   */
  void put(const T &ele) {
    std::unique_lock<std::mutex> lock(_mutex);

    _cond.wait(lock, [this]() { return !is_full(); });
    _queue.push_back(ele);

    lock.unlock();
    _cond.notify_all();
  }

  ~BlockingQueue() {}

private:
  std::mutex _mutex;
  std::condition_variable _cond;
  std::deque<T> _queue;
  int _queue_limit;
};

typedef void (*task_cb_t)(void *arg);

/**
 * @brief task interface which represents a job processed by class 'ThreadPool'
 */
class ITask {
public:
  ITask();
  virtual ~ITask();

  virtual void wait();  // wait for task finishing

  void notify();  // notify task finished

  int set_cb(task_cb_t cb, void *arg);

  void finish();

  inline void set_result(int result) { _result = result; }

  inline int get_result() { return _result; }

  virtual void execute() { finish(); }

private:
  // sync call interface
  std::mutex _mutex;
  std::condition_variable _cond;
  bool _finished;

  // async call interface
  task_cb_t _cb;
  void *_arg;
  int _result;
};

/**
 * @brief a class used to concurrently process multiple tasks,
 *        it contains one input-queue and mutiple workers which run in function
 * 'this->run'
 */
class ThreadPool {
public:
  explicit ThreadPool(int worker_num = 1, int queue_limit = 1000)
      : _worker_num(worker_num),
        _exit_mark(false),
        _task_queue(queue_limit),
        _threads_info(0) {}

  virtual ~ThreadPool() {
    while (!_task_queue.is_empty()) {
      ITask *t = _task_queue.get(1);
      if (t) {
        t->execute();
      }
    }
  }

  virtual int init(int workers = 1) {
    _worker_num = workers;
    return 0;
  }

  void set_worker_num(int num) { _worker_num = num; }

  void set_queue_limit(int limit) { _task_queue.set_queue_limit(limit); }

  virtual std::thread *_start() {
    std::thread *t = new std::thread(thread_routine, static_cast<void *>(this));
    return t;
  }

  virtual int start() {
    for (int i = 0; i < _worker_num; i++) {
      std::thread *t = _start();
      if (t) {
        _threads_info.push_back(t);
      } else {
        return -1;
      }
    }
    return 0;
  }

  int append_task(ITask *task) {
    if (!_exit_mark) {
      _task_queue.put(task);
      return 0;
    } else {
      return -1;
    }
  }

  int append_and_wait(ITask *task) {
    if (!_exit_mark) {
      _task_queue.put(task);
      task->wait();
      return 0;
    } else {
      return -1;
    }
  }

  void notify_exit() { _exit_mark = true; }

  virtual void join() {
    for (size_t i = 0; i < _threads_info.size(); i++) {
      _threads_info[i]->join();
      delete _threads_info[i];
    }
    _threads_info.clear();
    while (!_task_queue.is_empty()) {
      ITask *t = _task_queue.get(100);
      if (t) {
        delete t;
      }
    }
  }

protected:
  static void *thread_routine(void *args) {
    ThreadPool *pthr = static_cast<ThreadPool *>(args);
    pthr->run();
    return NULL;
  }

  virtual void run();

  ITask *fetch_task(uint32_t wait_ms = 0) { return _task_queue.get(wait_ms); }

protected:
  int _worker_num;
  volatile bool _exit_mark;
  BlockingQueue<ITask *> _task_queue;
  std::vector<std::thread *> _threads_info;
};

};      // namespace vistool
#endif  // DATAREADER_CPP_INCLUDE_CONCURRENT_H_
