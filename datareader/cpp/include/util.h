#ifndef DATAREADER_CPP_INCLUDE_UTIL_H
#define DATAREADER_CPP_INCLUDE_UTIL_H

#include <string>
#include <vector>
#include <stdint.h>

namespace vis {

int int_rand_range(int min, int max);

float float_rand_range(float min, float max);

std::string string_format(const char* fmt, ...);

std::vector<std::string> string_split(
        const std::string& s,
        const std::string& delim);

int64_t now_usec();

class BufLogger {
public:
    BufLogger(bool log_out = true);
    ~BufLogger();

    bool append(const char* fmt, ...);

    inline const std::string& get() const {
        return this->_buffer;
    }

private:
    int64_t _start_ts;
    bool _log_out;
    std::string _buffer;
};

};// end of namespace "vis"

#endif  //__UTIL_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
