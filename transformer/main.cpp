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

#include <iostream>
#include <fstream>
#include "DataTransformer.h"

int main(int argc, char** argv) {
  float mean[3] = {103.939, 116.779, 123.68};
  DataTransformer* trans = new DataTransformer(
      4, 1024, false, true, 224, 224, 256, false, true, mean);
  std::string src = argv[1];
  std::vector<std::string> files;
  std::vector<int> labels;

  std::ifstream infile(src.c_str());
  std::string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    files.push_back(line.substr(0, pos));
    labels.push_back(label);
  }
  std::cout << files.size() << std::endl;
  trans->processImgFile(files, labels.data());
  float* data = new float[3 * 224 * 224];
  int lab = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    trans->obtain(data, &lab);
    std::cout << lab << std::endl;
  }
  return 0;
}
