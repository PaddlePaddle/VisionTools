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

#ifndef DATAREADER_CPP_INCLUDE_LOGGER_H
#define DATAREADER_CPP_INCLUDE_LOGGER_H

#include <iostream>
#include <sstream>
namespace vistool {

//compatiable with glog
enum {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3,
};

struct NullStream : std::ostream {
    NullStream() : std::ios(0), std::ostream(0) {}
};

class Logger {
public:
    Logger(const char * filename, int lineno, int loglevel) {
        static const char * loglevelstrs[] = {"INFO ", "WARN ", "ERROR", "FATAL"};
        static NullStream nullstream;
        _loglevel = loglevel;
        _logstream = (_loglevel >= getloglevel()) ? &std::cerr : &nullstream;
        (*_logstream) << loglevelstrs[_loglevel] << ":" << filename << "[" << lineno << "]";
    }
    static inline int & getloglevel(){
        //default initialized with glog env
        static int globallevel = getgloglevel();
        return globallevel;
    }
    static inline void setloglevel(int loglevel){
        getloglevel() = loglevel;
    }
    static int getgloglevel(){
        char * env = getenv("GLOG_minloglevel");
        int level = WARNING;
        if (env != NULL) {
            int num = 0;
            std::istringstream istr(env);
            istr >> num;
            if (!istr.fail()){
                level = num;
            }
        }
        return level;
    }
    ~Logger() { *_logstream << std::endl; }
    std::ostream & getstream() { return *_logstream; }
protected:
    int _loglevel;
    std::ostream * _logstream;
};

}; // end of namespace 'vistool'

#define LOG(loglevel) Logger(__FILE__, __LINE__, loglevel).getstream()
#endif  //DATAREADER_CPP_INCLUDE_LOGGER_H
