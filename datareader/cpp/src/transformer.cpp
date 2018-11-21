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

#include "transformer.h"
#include "image_transformer.h"
#include "logger.h"

namespace vistool {

Transformer::Transformer() {
}

Transformer::~Transformer() {
}

Transformer * Transformer::create(const std::string &type) {
    LOG(INFO) << "Transformer::create(" << type << ")";

    if (type == "ImageTransformer") {
        ImageTransformer *t = new ImageTransformer();
        return t;
    } else {
        LOG(WARNING) << "failed to create transformer with type[" 
            << type << "]";
        return NULL;
    }
}

void Transformer::destroy(Transformer *t) {
    LOG(INFO) << "Transformer::destroy";
    if (t) {
        delete t;
    }
}
};// end of namespace 'vistool'

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
