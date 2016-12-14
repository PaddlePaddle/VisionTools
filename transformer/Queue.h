/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

template <typename T>
class BlockingQueue {
public:
  /**
   * @brief Construct Function.
   * @param[in] capacity the max numer of elements the queue can have.
   */
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}

  /**
   * @brief enqueue an element into Queue.
   * @param[in] x The enqueue element, pass by reference .
   * @note This method is thread-safe, and will wake up another thread
   * who was blocked because of the queue is empty.
   * @note If it's size() >= capacity before enqueue,
   * this method will block and wait until size() < capacity.
   */
  void enqueue(const T& x) {
    std::unique_lock<std::mutex> lock(mutex_);
    notFull_.wait(lock, [&] { return queue_.size() < capacity_; });
    queue_.push_back(x);
    notEmpty_.notify_one();
  }

  /**
   * Dequeue from a queue and return a element.
   * @note this method will be blocked until not empty.
   * @note this method will wake up another thread who was blocked because
   * of the queue is full.
   */
  T dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [&] { return !queue_.empty(); });

    T front(queue_.front());
    queue_.pop_front();
    notFull_.notify_one();
    return front;
  }

  /**
   * Return size of queue.
   *
   * @note This method is thread safe.
   * The size of the queue won't change until the method return.
   */
  size_t size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.size();
  }

  /**
   * @brief is empty or not.
   * @return true if empty.
   * @note This method is thread safe.
   */
  size_t empty() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.empty();
  }

private:
  std::mutex mutex_;
  std::condition_variable notEmpty_;
  std::condition_variable notFull_;
  std::deque<T> queue_;
  size_t capacity_;
};
