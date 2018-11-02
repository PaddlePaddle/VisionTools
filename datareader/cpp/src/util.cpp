#include <chrono>
#include <stdlib.h>
#include <stdarg.h>
#include "logger.h"
#include "util.h"

namespace vis {

int int_rand_range(int min, int max) {
    if (min > max) {
        int t = min;
        min = max;
        max = t;
    }
    return (random() % (max - min + 1)) + min;
}

float float_rand_range(float min, float max) {
    if (min > max) {
        float t = min;
        min = max;
        max = t;
    }
    long r = random();
    double d_r = double(r) / RAND_MAX;
    return float(min + (max - min) * d_r);
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

std::vector<std::string> string_split(const std::string& s,
        const std::string& delim) {
    std::vector< std::string> ret;
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
    std::chrono::time_point<std::chrono::system_clock> ts 
        = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            ts.time_since_epoch()).count();
}

BufLogger::BufLogger(bool log_out)
    : _start_ts(0),
      _log_out(log_out),
      _buffer("") {
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

};// end of namespace 'vis'

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
