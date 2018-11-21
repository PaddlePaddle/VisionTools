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

#ifndef DATAREADER_CPP_INCLUDE_UTIL_H
#define DATAREADER_CPP_INCLUDE_UTIL_H

#include <string>
#include <vector>
#include <stdint.h>

namespace vistool {

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
};// end of namespace "vistool"

#endif  //__UTIL_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
