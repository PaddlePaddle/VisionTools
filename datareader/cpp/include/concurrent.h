/*
 * this module provides multi-threading interfaces
 */

#ifndef DATAREADER_CPP_INCLUDE_CONCURRENT_H
#define DATAREADER_CPP_INCLUDE_CONCURRENT_H

#include <deque>
#include <vector>
#include <chrono>
#include <stdarg.h>
#include <stdint.h>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace vis {

template<typename T>
class BlockingQueue {
public:
    BlockingQueue(int queue_len = 100)
        : _queue_limit(queue_len) {
    }

    int set_queue_limit (int queue_len) {
        this->_queue_limit = queue_len;
        return 0;
    }

    inline bool is_full() {
        return _queue.size() >= _queue_limit;
    }

    inline bool is_empty() {
        return _queue.empty();
    }

    inline int length() const {
        return _queue.size();
    }

    T get(uint32_t wait_ms = 0) {
        std::unique_lock<std::mutex> lock(_mutex);
        bool cond_meet = false;
        if (!wait_ms) {
            _cond.wait(lock, [this](){
                    return !is_empty();
                    });
            cond_meet = true;
        } else {
            if (_cond.wait_for(lock, std::chrono::milliseconds(wait_ms),
                        [this](){ return !is_empty(); })) {
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

    int get(T &res) {
        std::unique_lock<std::mutex> lock(_mutex);

        _cond.wait(lock, [this](){
                return !is_empty();
                });
        res = _queue.front();
        _queue.pop_front();

        lock.unlock();
        _cond.notify_all();
        return 0;
    }

    void put(const T &ele) {
        std::unique_lock<std::mutex> lock(_mutex);

        _cond.wait(lock, [this](){
                return !is_full();
                });
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

class ITask {
public:
    ITask();
    virtual ~ITask();

    virtual void wait(); //wait for task finishing

    void notify ();//notify task finished
  
    int set_cb(task_cb_t cb, void *arg);

    void finish ();

    inline void set_result (int result) {
        _result = result;
    }

    inline int get_result() {
        return _result;
    }

    virtual void execute() { 
        finish(); 
    }

private:
    //sync call interface
    std::mutex _mutex;
    std::condition_variable _cond;
    bool _finished;

    //async call interface
    task_cb_t _cb;
    void *_arg;
    int _result;
};

//a class provide one queue and mutiple consumers
class ThreadPool {
public:

    ThreadPool(int worker_num = 1, int queue_limit = 1000)
        : _worker_num(worker_num),
          _exit_mark(false),
          _task_queue(queue_limit),
          _threads_info(0) {
    }

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

    void set_worker_num(int num) {
        _worker_num = num;
    }

    void set_queue_limit(int limit) {
        _task_queue.set_queue_limit(limit);
    }

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

    int append_task(ITask* task) {
        if (!_exit_mark) {
            _task_queue.put(task);
            return 0;
        } else {
            return -1;
        }
    }

    int append_and_wait(ITask* task) {
        if (!_exit_mark) {
            _task_queue.put(task);
            task->wait();
            return 0;
        } else {
            return -1;
        }
    }

    void notify_exit() {
        _exit_mark = true;
    }

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
    static void* thread_routine(void *args) {
        ThreadPool *pthr = static_cast<ThreadPool *>(args);
        pthr->run();
        return (void *)0;
    }

    virtual void run();

    ITask *fetch_task(uint32_t wait_ms = 0) {
        return _task_queue.get(wait_ms);
    }

protected:
    int _worker_num;
    volatile bool _exit_mark;
    BlockingQueue<ITask *> _task_queue;
    std::vector<std::thread *> _threads_info;
};

};// end of namespace 'vis'

#endif

/* vim: set ts=4 sw=4: */

